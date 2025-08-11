#!/usr/bin/env python3
# LoRa_ft.py — fairseq-only LoRA fine-tuning of REAL EAT, explicit CLS training, ONNX export (pipeline-compatible)
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import Dataset, DataLoader
from torchaudio.transforms import MelSpectrogram
from tqdm import tqdm

from fairseq import checkpoint_utils

# --- Register EAT tasks/models with Fairseq (import for side effects) ---
import EAT.tasks.pretraining_AS2M          # registers "mae_image_pretraining" task
import EAT.models.EAT_pretraining          # registers data2vec_multi model, Modality, etc.
import EAT.models.mae
import EAT.models.images
# -----------------------------------------------------------------------

from EAT.models.EAT_pretraining import Modality  # IMAGE modality for spectrogram "images"


# ───────────────────────── DATA ─────────────────────────
class MelSpecDataset(Dataset):
    """
    On-the-fly wav -> log-Mel (128, T_len) -> (1, 128, T_len).
    Matches your pipeline (n_fft=400, hop=160, n_mels=128) and normalization.
    """
    def __init__(
        self,
        wavs: List[torch.Tensor],
        sr: int = 16000,
        n_fft: int = 400,
        hop: int = 160,
        n_mels: int = 128,
        t_len: int = 1024,
        normalize_mean: float = -4.288,
        normalize_std: float = 4.469,
        device: torch.device = torch.device("cpu"),
    ):
        self.wavs = wavs
        self.t_len = t_len
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.mel = MelSpectrogram(
            sample_rate=sr,
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            center=True,
            pad_mode="reflect",
            power=2.0,
            normalized=False,
        ).to(device)

    def __len__(self):
        return len(self.wavs)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.wavs[idx]
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if x.dim() > 1:
            x = x.mean(dim=0)  # mono
        x = x.to(self.mel.mel_scale.fb.device, dtype=torch.float32)

        # (mel, time)
        spec = self.mel(x)
        spec = torch.log(torch.clamp(spec, min=1e-10))

        # pipeline normalization
        spec = (spec - self.normalize_mean) / (self.normalize_std + 1e-6)

        # pad/crop to t_len
        if spec.shape[1] < self.t_len:
            pad = self.t_len - spec.shape[1]
            spec = torch.nn.functional.pad(spec, (0, pad))
        else:
            spec = spec[:, : self.t_len]

        spec = spec.unsqueeze(0)  # (1, 128, T_len)
        return spec, spec


def collate_pad(batch):
    xs = [b[0] for b in batch]  # (1, 128, T)
    x = torch.stack(xs, dim=0)  # (B, 1, 128, T)
    return x, x


# ──────────────────────── LOADERS ────────────────────────
def load_eat_fairseq(ckpt_path: str, device: str = "cuda"):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].eval()
    if device and torch.cuda.is_available() and device.startswith("cuda"):
        model = model.cuda()
        device_t = torch.device("cuda")
    else:
        device_t = torch.device("cpu")
    return model, cfg, task, device_t


def load_fresh_eat_from_ckpt(ckpt_path: str, device: str = "cuda"):
    """Create a fresh Fairseq EAT model instance from checkpoint (avoids deepcopy recursion)."""
    models, _, _ = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0].eval()
    if device and torch.cuda.is_available() and device.startswith("cuda"):
        model = model.cuda()
    return model


# ──────────────────────── LoRA (PEFT) ────────────────────────
def discover_attn_linear_names(model: nn.Module):
    """Find attention qkv/proj Linear modules to target with LoRA."""
    targets = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            lname = name.lower()
            if "attn" in lname and ("qkv" in lname or "in_proj" in lname or lname.endswith(".proj") or ".proj." in lname):
                targets.add(name)
    if not targets:
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and "attn" in name.lower():
                targets.add(name)
    return sorted(targets)


def attach_lora(model: nn.Module, rank: int, alpha: int, dropout: float):
    from peft import LoraConfig, get_peft_model
    target_modules = discover_attn_linear_names(model)
    if not target_modules:
        raise RuntimeError("Could not find attention linear layers for LoRA.")
    print("LoRA targets:", target_modules[:6], "..." if len(target_modules) > 6 else "")

    lcfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        target_modules=target_modules
    )
    lora_model = get_peft_model(model, lcfg)
    lora_model.print_trainable_parameters()
    return lora_model


# ──────────────────────── LOSSES ────────────────────────
def cls_distill_loss(student: nn.Module, teacher: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """
    Explicitly train CLS: student CLS (index 0) → teacher mean over patch tokens.
    Student: keep extra tokens (CLS) -> x[:,0,:]
    Teacher: remove extra tokens (patches only) -> mean over time.
    """
    s_out = student.extract_features(
        xb, mode=Modality.IMAGE, padding_mask=None, mask=False, remove_extra_tokens=False
    )
    s_tokens = s_out["x"]             # (B, S, C)
    s_cls = s_tokens[:, 0, :]         # (B, C)

    with torch.no_grad():
        t_out = teacher.extract_features(
            xb, mode=Modality.IMAGE, padding_mask=None, mask=False, remove_extra_tokens=True
        )
        t_tokens = t_out["x"]         # (B, P, C)
        t_mean = t_tokens.mean(dim=1) # (B, C)

    return torch.nn.functional.mse_loss(s_cls, t_mean)


# ───────────────────────── TRAIN ─────────────────────────
def train(student, teacher, loader, epochs, lr, device):
    student.to(device)
    teacher.to(device).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    trainable = [p for p in student.parameters() if p.requires_grad]
    tot = sum(p.numel() for p in student.parameters())
    trn = sum(p.numel() for p in trainable)
    print(f"→ training {trn:,} / {tot:,} params ({100*trn/tot:.2f} %)")

    opt = optim.AdamW(trainable, lr=lr, weight_decay=1e-2)
    steps_per_epoch = max(1, len(loader))
    sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * steps_per_epoch)

    for ep in range(1, epochs + 1):
        student.train()
        running = 0.0
        pbar = tqdm(loader, desc=f"Epoch {ep}/{epochs}", unit="batch")
        for xb, _ in pbar:
            xb = xb.to(device)  # (B,1,128,T)

            loss = cls_distill_loss(student, teacher, xb)

            opt.zero_grad()
            loss.backward()
            opt.step()
            sch.step()
            running += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        print(f"Epoch {ep}: mean loss {running/len(loader):.4f}")
    return student.eval()


# ───────────────────────── SAVE / ONNX ─────────────────────────
class ExportableEAT(torch.nn.Module):
    """
    Wrapper that matches your sklearn pipeline:
      input:  (B, T, 1, 128)
      output: (B, S, C) with CLS at index 0 (we keep extra tokens)
    Internally permutes to (B, 1, 128, T) for the IMAGE modality.
    """
    def __init__(self, eat_model: nn.Module):
        super().__init__()
        self.m = eat_model

    def forward(self, x):
        # Expect (B, T, 1, 128)
        if x.dim() != 4 or x.shape[2] != 1:
            raise RuntimeError("Expected input shape (B, T, 1, 128)")
        # (B, T, 1, 128) -> (B, 1, 128, T)
        x = x.permute(0, 2, 3, 1)
        out = self.m.extract_features(
            x, mode=Modality.IMAGE, padding_mask=None, mask=False, remove_extra_tokens=False
        )  # keep CLS
        return out["x"]  # (B, S, C)


def save_all(student, out_dir, t_len=1024, device="cpu"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)

    # Save LoRA adapters if present
    try:
        student.save_pretrained(out / "lora_adapters")
    except Exception:
        pass

    # Merge adapters into base weights
    merged = getattr(student, "merge_and_unload", lambda: student)()
    torch.save({"model": merged.state_dict()}, out / "eat_fairseq_lora_cls_merged.pt")

    # ONNX export, pipeline-compatible I/O (B, T, 1, 128)
    try:
        dummy = torch.randn(1, t_len, 1, 128, device=device)
        wrapped = ExportableEAT(merged).eval()
        torch.onnx.export(
            wrapped, dummy, out / "eat_fairseq_lora_cls_merged.onnx",
            input_names=["input"], output_names=["features"], opset_version=17,
            dynamic_axes={"input": {0: "batch", 1: "time"},
                          "features": {0: "batch", 1: "time"}},
            do_constant_folding=False, keep_initializers_as_inputs=False
        )
        print("✓ saved adapters/, merged .pt, and ONNX")
    except Exception as e:
        print(f"ONNX export skipped: {e}")


# ───────────────────────── MAIN ─────────────────────────
def main():
    ap = argparse.ArgumentParser("LoRA fine-tune REAL EAT — explicit CLS training")
    ap.add_argument("--ckpt", required=True, help="Path to EAT fairseq checkpoint (.pt)")
    ap.add_argument("--dataset_root", default="./dataset")
    ap.add_argument("--machine", default="bearing")
    ap.add_argument("--batch",   type=int, default=8)
    ap.add_argument("--epochs",  type=int, default=2)
    ap.add_argument("--lr",      type=float, default=1e-4)
    ap.add_argument("--rank",    type=int, default=64)
    ap.add_argument("--alpha",   type=int, default=32)
    ap.add_argument("--drop",    type=float, default=0.05)
    ap.add_argument("--t_len",   type=int, default=1024)
    ap.add_argument("--n_mels",  type=int, default=128)
    ap.add_argument("--out",     default="./eat_fairseq_lora_cls_ft")
    ap.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    # Load base once to verify env / registry
    base_model, cfg, task, _ = load_eat_fairseq(args.ckpt, args.device)
    print("✓ REAL EAT (fairseq) loaded")

    # Teacher & student: load fresh copies from checkpoint (avoid deepcopy recursion)
    teacher = load_fresh_eat_from_ckpt(args.ckpt, args.device).to(dev).eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = load_fresh_eat_from_ckpt(args.ckpt, args.device).to(dev).eval()
    student = attach_lora(student, rank=args.rank, alpha=args.alpha, dropout=args.drop)

    # Gather wavs under dataset_root/machine/train/*.wav
    wavs = []
    train_dir = Path(args.dataset_root) / args.machine / "train"
    if train_dir.exists():
        for p in sorted(train_dir.glob("*.wav")):
            try:
                wav, sr = torchaudio.load(str(p))
                if sr != 16000:
                    wav = torchaudio.functional.resample(wav, sr, 16000)
                wavs.append(wav.squeeze(0))  # (N,)
            except Exception:
                continue
    if not wavs:
        raise RuntimeError(f"No wavs found under {train_dir}. Put training audio there.")

    ds = MelSpecDataset(
        wavs, sr=16000, n_fft=400, hop=160, n_mels=args.n_mels, t_len=args.t_len, device=dev
    )
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, drop_last=True,
                    num_workers=4, collate_fn=collate_pad)

    student = train(student, teacher, dl, args.epochs, args.lr, dev)

    save_all(student, args.out, t_len=args.t_len, device=dev)


if __name__ == "__main__":
    main()
