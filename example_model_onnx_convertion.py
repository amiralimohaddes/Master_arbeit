#!/usr/bin/env python3
import os
import torch

# From your uploaded files
from EAT.models.EAT_pretraining import (
    Data2VecMultiModel,
    Data2VecMultiConfig,
    D2vModalitiesConfig,
    Modality,
)
from EAT.models.images import D2vImageConfig

# ---------- Build EAT (image modality) ----------
def build_eat_image(embed_dim=768, depth=12, num_heads=12,
                    in_chans=1, img_size=128, patch_size=16):
    model_cfg = Data2VecMultiConfig(
        embed_dim=embed_dim,
        num_heads=num_heads,
        depth=depth,
        modalities=D2vModalitiesConfig(
            image=D2vImageConfig(
                # important bits for your log-mel spectrograms
                input_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                # make the first returned token a global CLS-like token
                num_extra_tokens=1,
            )
        ),
        supported_modality=Modality.IMAGE,
    )

    model = Data2VecMultiModel(
        model_cfg,
        modalities=[Modality.IMAGE],
        skip_ema=True,
    )
    return model


# ---------- Load checkpoint, ignore decoder weights ----------
def load_checkpoint_into_eat(eat_model, ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    new_sd = {}
    for k, v in sd.items():
        # strip common prefixes
        nk = k
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
        # fairseq/EAT uses uppercase modality names internally
        nk = nk.replace("modality_encoders.image", "modality_encoders.IMAGE")

        # we do NOT need the decoder for feature extraction; drop it to avoid shape mismatches
        if ".decoder." in nk:
            continue

        new_sd[nk] = v

    missing, unexpected = eat_model.load_state_dict(new_sd, strict=False)
    print(f"[load] missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print("  (first 10 missing):", *missing[:10], sep="\n  ")
    if unexpected:
        print("  (first 10 unexpected):", *unexpected[:10], sep="\n  ")


# ---------- Thin wrapper so ONNX returns tokens (CLS first) ----------
class ExportableEAT(torch.nn.Module):
    """
    Input expected by your pipeline's ONNXFeatureExtractor:
      x: (B, T, 1, H)   where H=128 (mel bins), T=frames
    We internally permute to (B, 1, H, T) for the EAT image encoder.
    Output:
      (B, S, C) tokens with CLS at index 0 -> pipeline's feature_mask takes it.
    """
    def __init__(self, eat_model):
        super().__init__()
        self.eat = eat_model

    def forward(self, x):
        # Accept (B, T, 1, H) or (B, T, H)
        if x.dim() == 3:  # (B, T, H) -> (B, T, 1, H)
            x = x.unsqueeze(2)
        # (B, T, 1, H) -> (B, 1, H, T)
        x = x.permute(0, 2, 3, 1)

        out = self.eat.extract_features(
            x,
            padding_mask=None,
            mask=False,
            # remove_masked=False,
            remove_extra_tokens=False,   # KEEP the CLS token at index 0
            mode=Modality.IMAGE,
            # features_only=True,
        )
        return out["x"]  # (B, S, 768), S = 1 (CLS) + patches


if __name__ == "__main__":
    # ---- paths ----
    onnx_out  = "onnx1_export.onnx"   # <- matches your pipeline
    ckpt_path = os.path.join("pt_files", "EAT-base_epoch30_pt.pt")

    # ---- build & load ----
    eat = build_eat_image(
        embed_dim=768, depth=12, num_heads=12,
        in_chans=1, img_size=128, patch_size=16,
    )
    load_checkpoint_into_eat(eat, ckpt_path)
    eat.eval()

    exportable = ExportableEAT(eat).eval()

    # ---- quick sanity ----
    B, T, H = 1, 1024, 128
    dummy = torch.randn(B, T, H)
    with torch.no_grad():
        y = exportable(dummy)
    print("[sanity] output shape:", tuple(y.shape))  # e.g., (1, 513, 768) for T=1024, H=128, patch=16

    # ---- ONNX export ----
    torch.onnx.export(
        exportable,
        dummy,
        onnx_out,
        input_names=["input"],
        output_names=["features"],
        opset_version=17,
        dynamic_axes={
            "input":    {0: "batch", 1: "time"},
            "features": {0: "batch", 1: "time"},  # time here = token length (1 + #patches)
        },
        do_constant_folding=False,
        keep_initializers_as_inputs=False,
    )
    print(f"Saved ONNX to: {onnx_out}")
