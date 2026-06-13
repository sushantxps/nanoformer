"""Training utilities: a cosine learning-rate schedule with warmup, and a
compact trainer with gradient accumulation, gradient clipping, optional
mixed precision, and checkpointing.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Optional

import torch
import torch.nn as nn

from .model import Transformer, ModelConfig


@dataclass
class TrainConfig:
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    max_steps: int = 2000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 1
    eval_interval: int = 200
    eval_steps: int = 50
    beta1: float = 0.9
    beta2: float = 0.95
    amp: bool = False
    log_interval: int = 50


def cosine_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup followed by cosine decay to ``min_lr``."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def configure_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    """AdamW with weight decay on matmul weights only (not norms/biases)."""
    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2))


Batch = tuple[torch.Tensor, torch.Tensor]
BatchFn = Callable[[str], Batch]  # split name -> (inputs, targets)


class Trainer:
    def __init__(
        self,
        model: Transformer,
        cfg: TrainConfig,
        get_batch: BatchFn,
        device: str = "cpu",
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.get_batch = get_batch
        self.device = device
        self.opt = configure_optimizer(model, cfg)
        self.use_amp = cfg.amp and device == "cuda"
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.history: list[dict] = []

    @torch.no_grad()
    def evaluate(self) -> float:
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval_steps):
            x, y = self.get_batch("val")
            x, y = x.to(self.device), y.to(self.device)
            _, loss, _ = self.model(x, targets=y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def train(self) -> list[dict]:
        self.model.train()
        t0 = time.time()
        for step in range(self.cfg.max_steps):
            lr = cosine_lr(step, self.cfg)
            for g in self.opt.param_groups:
                g["lr"] = lr

            self.opt.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(self.cfg.grad_accum_steps):
                x, y = self.get_batch("train")
                x, y = x.to(self.device), y.to(self.device)
                with torch.autocast(
                    device_type="cuda" if self.use_amp else "cpu",
                    enabled=self.use_amp,
                ):
                    _, loss, _ = self.model(x, targets=y)
                    loss = loss / self.cfg.grad_accum_steps
                self.scaler.scale(loss).backward()
                accum_loss += loss.item()

            if self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.scaler.step(self.opt)
            self.scaler.update()

            if step % self.cfg.log_interval == 0:
                dt = time.time() - t0
                rec = {"step": step, "train_loss": accum_loss, "lr": lr, "elapsed": dt}
                self.history.append(rec)
                print(
                    f"step {step:5d} | loss {accum_loss:.4f} | "
                    f"lr {lr:.2e} | {dt:.1f}s"
                )

            if step > 0 and step % self.cfg.eval_interval == 0:
                val = self.evaluate()
                self.history.append({"step": step, "val_loss": val})
                print(f"  -> val loss {val:.4f}")

        return self.history

    def save_checkpoint(self, path: str | Path) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "model_config": self.model.cfg.to_dict(),
                "optimizer": self.opt.state_dict(),
                "history": self.history,
            },
            path,
        )

    @staticmethod
    def load_model(path: str | Path, device: str = "cpu") -> Transformer:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Transformer(ModelConfig(**ckpt["model_config"]))
        model.load_state_dict(ckpt["model"])
        return model.to(device)
