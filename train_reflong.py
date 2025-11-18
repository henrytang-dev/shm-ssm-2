"""Fine-tuning script for SMH-SSM adapters on the Ref-Long dataset."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from transformers import AutoTokenizer, get_scheduler

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf


LOGGER = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SMH-SSM on Ref-Long")
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--adapter_config", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--answers_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--logging_dir", type=str, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--max_seq_length", type=int, default=4096)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--logging_steps", type=int, default=25)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_total_limit", type=int, default=3)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", type=str, default="eval_accuracy")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--saliency_mode", choices=["none", "grad"], default="none")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    return parser.parse_args()


def configure_logging(log_dir: Optional[str]) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(Path(log_dir) / "training.log")
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logging.getLogger().addHandler(handler)


def load_adapter_config(path: Optional[str]) -> Optional[Dict]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def format_answer(doc_ids: Sequence[int]) -> str:
    docs = ", ".join(str(int(d)) for d in sorted(doc_ids))
    return f"Documents: {{{docs}}}"


def load_reflong(dataset_path: Path, answers_path: Path) -> Dict[str, Dict]:
    prompts = json.loads(dataset_path.read_text())
    answers = json.loads(answers_path.read_text())
    keys = sorted(set(prompts.keys()) & set(answers.keys()), key=lambda x: int(x))
    merged = {}
    for key in keys:
        merged[key] = {
            "prompt": prompts[key][1],
            "doc_ids": [int(v) for v in answers[key][1]],
            "answer_text": format_answer(answers[key][1]),
        }
    return merged


@dataclass
class TokenizedSample:
    input_ids: List[int]
    labels: List[int]
    prompt_text: str
    answer_text: str
    doc_ids: List[int]


class RefLongDataset(Dataset):
    def __init__(self, entries: Sequence[Dict], tokenizer: AutoTokenizer, max_seq_length: int) -> None:
        self.samples = [self._encode(tokenizer, entry, max_seq_length) for entry in entries]

    def _encode(self, tokenizer: AutoTokenizer, entry: Dict, max_seq_length: int) -> TokenizedSample:
        prompt_ids = tokenizer(entry["prompt"], add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(entry["answer_text"], add_special_tokens=False)["input_ids"]
        if tokenizer.eos_token_id is not None:
            answer_ids = answer_ids + [tokenizer.eos_token_id]
        total = len(prompt_ids) + len(answer_ids)
        if total > max_seq_length:
            overflow = total - max_seq_length
            drop = min(len(prompt_ids), overflow)
            prompt_ids = prompt_ids[drop:]
            total = len(prompt_ids) + len(answer_ids)
            if total > max_seq_length:
                keep = max_seq_length - len(answer_ids)
                prompt_ids = prompt_ids[-max(keep, 0):]
        input_ids = prompt_ids + answer_ids
        labels = [-100] * len(prompt_ids) + answer_ids
        return TokenizedSample(
            input_ids=input_ids,
            labels=labels,
            prompt_text=entry["prompt"],
            answer_text=entry["answer_text"],
            doc_ids=entry["doc_ids"],
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> TokenizedSample:
        return self.samples[idx]


class DataCollator:
    def __init__(self, pad_token_id: int, max_length: int) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, batch: Sequence[TokenizedSample]) -> Dict:
        max_len = min(self.max_length, max(len(sample.input_ids) for sample in batch))
        batch_size = len(batch)
        input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
        prompt_texts, answer_texts, doc_ids = [], [], []
        for i, sample in enumerate(batch):
            seq = sample.input_ids[-max_len:]
            tgt = sample.labels[-max_len:]
            length = len(seq)
            input_ids[i, :length] = torch.tensor(seq, dtype=torch.long)
            labels[i, :length] = torch.tensor(tgt, dtype=torch.long)
            attention_mask[i, :length] = 1
            prompt_texts.append(sample.prompt_text)
            answer_texts.append(sample.answer_text)
            doc_ids.append(sample.doc_ids)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_texts": prompt_texts,
            "answer_texts": answer_texts,
            "doc_ids": doc_ids,
        }


class EmbeddingSaliencyHelper:
    def __init__(self, model: MambaLMHeadModel) -> None:
        self.activations: Optional[torch.Tensor] = None
        self.hook = model.backbone.embedding.register_forward_hook(self._hook)

    def _hook(self, _module, _inputs, output):
        self.activations = output

    def compute(self, loss: torch.Tensor) -> Optional[torch.Tensor]:
        if self.activations is None:
            return None
        grads = torch.autograd.grad(loss, self.activations, retain_graph=False, allow_unused=True)[0]
        self.activations = None
        if grads is None:
            return None
        return grads.norm(dim=-1, keepdim=True).detach()

    def remove(self) -> None:
        self.hook.remove()


def extract_numbers(text: str) -> List[int]:
    return [int(n) for n in re.findall(r"\d+", text)]


def calculate_f1(pred: Sequence[int], target: Sequence[int]) -> float:
    pred_set, target_set = set(pred), set(target)
    if not pred_set and not target_set:
        return 1.0
    if not pred_set or not target_set:
        return 0.0
    true_pos = len(pred_set & target_set)
    precision = true_pos / len(pred_set)
    recall = true_pos / len(target_set)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def split_dataset(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
) -> Tuple[RefLongDataset, Optional[RefLongDataset]]:
    merged = load_reflong(Path(args.dataset_path), Path(args.answers_path))
    keys = list(merged.keys())
    random.shuffle(keys)
    val_count = int(len(keys) * args.val_split)
    val_keys = keys[:val_count] if val_count > 0 else []
    train_keys = keys[val_count:]
    if args.max_train_samples:
        train_keys = train_keys[: args.max_train_samples]
    if args.max_eval_samples and val_keys:
        val_keys = val_keys[: args.max_eval_samples]
    train_entries = [merged[k] for k in train_keys]
    val_entries = [merged[k] for k in val_keys]
    train_dataset = RefLongDataset(train_entries, tokenizer, args.max_seq_length)
    eval_dataset = RefLongDataset(val_entries, tokenizer, args.max_seq_length) if val_entries else None
    return train_dataset, eval_dataset


def create_model(args: argparse.Namespace) -> MambaLMHeadModel:
    config_dict = load_config_hf(args.model_name_or_path)
    allowed_fields = set(MambaConfig.__dataclass_fields__.keys())
    filtered = {k: v for k, v in config_dict.items() if k in allowed_fields}
    config = MambaConfig(**filtered)
    adapter_cfg = load_adapter_config(args.adapter_config)
    if adapter_cfg is not None:
        config.adapter_cfg = adapter_cfg
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    model = MambaLMHeadModel(config, dtype=dtype)
    state_dict = load_state_dict_hf(args.model_name_or_path, dtype=dtype)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        LOGGER.warning("Missing keys when loading weights: %s", missing)
    if unexpected:
        LOGGER.warning("Unexpected keys when loading weights: %s", unexpected)
    return model


def evaluate(
    accelerator: Accelerator,
    model: MambaLMHeadModel,
    tokenizer: AutoTokenizer,
    dataloader: Optional[DataLoader],
    loss_fn: nn.Module,
    args: argparse.Namespace,
    saliency_mode: str,
) -> Dict[str, float]:
    if dataloader is None:
        return {}
    model.eval()
    losses, acc_flags, f1_scores = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(accelerator.device)
            labels = batch["labels"].to(accelerator.device)
            saliency_scores = None
            if saliency_mode == "grad":
                helper = EmbeddingSaliencyHelper(accelerator.unwrap_model(model))
                with torch.enable_grad():
                    model.train()
                    outputs = model(input_ids=input_ids, saliency_scores=None)
                    sal_loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                    saliency_scores = helper.compute(sal_loss)
                helper.remove()
                model.eval()
                if saliency_scores is not None:
                    saliency_scores = saliency_scores.to(accelerator.device)
            outputs = model(input_ids=input_ids, saliency_scores=saliency_scores)
            loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
            losses.append(accelerator.gather_for_metrics(loss.repeat(input_ids.size(0))))
            pred_ids = outputs.logits.argmax(dim=-1)
            mask = labels != -100
            batch_acc, batch_f1 = [], []
            for i in range(input_ids.size(0)):
                tgt_tokens = torch.masked_select(labels[i], mask[i])
                pred_tokens = torch.masked_select(pred_ids[i], mask[i])
                pred_text = tokenizer.decode(pred_tokens.tolist(), skip_special_tokens=True)
                pred_nums = extract_numbers(pred_text)
                gold_nums = batch["doc_ids"][i]
                batch_acc.append(float(set(pred_nums) == set(gold_nums)))
                batch_f1.append(calculate_f1(pred_nums, gold_nums))
            acc_flags.append(accelerator.gather_for_metrics(torch.tensor(batch_acc, device=accelerator.device)))
            f1_scores.append(accelerator.gather_for_metrics(torch.tensor(batch_f1, device=accelerator.device)))
    eval_loss = torch.cat(losses).mean().item() if losses else 0.0
    eval_acc = torch.cat(acc_flags).mean().item() if acc_flags else 0.0
    eval_f1 = torch.cat(f1_scores).mean().item() if f1_scores else 0.0
    return {"eval_loss": eval_loss, "eval_accuracy": eval_acc, "eval_f1": eval_f1}


def main() -> None:
    args = parse_args()
    configure_logging(args.logging_dir)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=args.logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if args.bf16 else "no",
        project_config=project_config,
    )
    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        if args.logging_dir:
            Path(args.logging_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset, eval_dataset = split_dataset(args, tokenizer)
    collator = DataCollator(tokenizer.pad_token_id, args.max_seq_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=args.per_device_eval_batch_size,
            shuffle=False,
            num_workers=args.dataloader_num_workers,
            collate_fn=collator,
        )
        if eval_dataset
        else None
    )
    model = create_model(args)
    if args.gradient_checkpointing:
        LOGGER.warning("Gradient checkpointing requested but not implemented for this architecture.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = int(args.num_train_epochs * steps_per_epoch)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, eval_loader, scheduler
    )
    unwrapped = accelerator.unwrap_model(model)
    saliency_helper = EmbeddingSaliencyHelper(unwrapped) if args.saliency_mode == "grad" else None
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    global_step = 0
    best_metric = None
    saved_ckpts: List[Path] = []

    def save_checkpoint(step: int, is_best: bool = False) -> None:
        accelerator.wait_for_everyone()
        if not accelerator.is_main_process:
            return
        ckpt_dir = Path(args.output_dir) / f"checkpoint-{step}"
        unwrapped.save_pretrained(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        torch.save(optimizer.state_dict(), ckpt_dir / "optimizer.pt")
        torch.save(scheduler.state_dict(), ckpt_dir / "scheduler.pt")
        saved_ckpts.append(ckpt_dir)
        while len(saved_ckpts) > args.save_total_limit:
            old = saved_ckpts.pop(0)
            shutil.rmtree(old, ignore_errors=True)
        if is_best:
            best_dir = Path(args.output_dir) / "best"
            if best_dir.exists():
                shutil.rmtree(best_dir)
            shutil.copytree(ckpt_dir, best_dir)

    total_epochs = math.ceil(args.num_train_epochs)
    for epoch in range(total_epochs):
        model.train()
        for batch in train_loader:
            global_step += 1
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)
                saliency_scores = None
                if args.saliency_mode == "grad" and saliency_helper is not None:
                    model.eval()
                    with torch.enable_grad():
                        outputs = model(input_ids=input_ids, saliency_scores=None)
                        sal_loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                        saliency_scores = saliency_helper.compute(sal_loss)
                    model.train()
                    if saliency_scores is not None:
                        saliency_scores = saliency_scores.to(accelerator.device)
                outputs = model(input_ids=input_ids, saliency_scores=saliency_scores)
                loss = loss_fn(outputs.logits.view(-1, outputs.logits.size(-1)), labels.view(-1))
                accelerator.backward(loss)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if accelerator.is_main_process and global_step % args.logging_steps == 0:
                LOGGER.info("Step %s | Loss %.4f", global_step, loss.item())
            if args.save_steps and global_step % args.save_steps == 0:
                save_checkpoint(global_step)
            if eval_loader and args.eval_steps and global_step % args.eval_steps == 0:
                metrics = evaluate(accelerator, model, tokenizer, eval_loader, loss_fn, args, args.saliency_mode)
                accelerator.print(f"Eval @ step {global_step}: {metrics}")
                current_metric = metrics.get(args.metric_for_best_model)
                if args.load_best_model_at_end and current_metric is not None:
                    better = (
                        best_metric is None
                        or (args.greater_is_better and current_metric > best_metric)
                        or (not args.greater_is_better and current_metric < best_metric)
                    )
                    if better:
                        best_metric = current_metric
                        save_checkpoint(global_step, is_best=True)
        accelerator.print(f"Finished epoch {epoch + 1}")

    if eval_loader:
        final_metrics = evaluate(accelerator, model, tokenizer, eval_loader, loss_fn, args, args.saliency_mode)
        accelerator.print(f"Final evaluation: {final_metrics}")

    if saliency_helper is not None:
        saliency_helper.remove()


if __name__ == "__main__":
    main()
