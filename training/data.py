"""Unified data utilities supporting three modalities:
1. TSV image-text pairs (legacy metadata.tsv)
2. 🤗 Datasets (image+text **or** text‑only with on‑the‑fly rendering)
3. Synthetic or pre‑baked HDF5 corpora for OCR training

Call `make_dataloader(kind=..., ...)` from train.py - nothing else changes.
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, Iterable, Dict, Callable, Any
import random, math, io

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset, get_worker_info
from torchvision import transforms
from PIL import Image, ImageDraw, ImageFont

from tokenizer import BPETokenizer

# ──────────────────────────────────────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────────────────────────────────────
PATCH = 16                             # patch size for CNN stem
IMG_SIZE = 224                         # default resize

def _patchify(img: torch.Tensor) -> torch.Tensor:
    """img [C,H,W] → [N_patches, flat_dim]"""
    patches = F.unfold(img.unsqueeze(0), kernel_size=PATCH, stride=PATCH)
    return patches.squeeze(0).transpose(0, 1)

# ----------------------------------------------------------------------------
# Dataset registry
# ----------------------------------------------------------------------------
_DATASET_REG: Dict[str, Callable[..., Dataset]] = {}

def register(name: str):
    def _decorator(cls):
        _DATASET_REG[name] = cls
        return cls
    return _decorator

import inspect

def build_dataset(kind: str, **kw):
    if kind not in _DATASET_REG:
        raise ValueError(f"Unknown dataset kind: {kind}. Options: {list(_DATASET_REG)}")
    cls = _DATASET_REG[kind]
    # Filter kw to only args the dataset constructor accepts
    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kw.items() if k in valid}
    return cls(**filtered)

# ----------------------------------------------------------------------------
# 1. Legacy TSV image-text dataset
# ----------------------------------------------------------------------------
@register("tsv")
class VisionTextDataset(Dataset):
    """metadata.tsv lines: <image_path>	<text>"""
    def __init__(self, meta_file: str, tokenizer: BPETokenizer, img_size: int = IMG_SIZE, max_txt_len: int = 256):
        self.entries = [line.strip().split("	", 1) for line in Path(meta_file).read_text().splitlines()]
        self.tk = tokenizer
        self.max_txt_len = max_txt_len
        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size), antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        img_path, text = self.entries[idx]
        img = self.img_tf(Image.open(img_path).convert("RGB"))
        patches = _patchify(img)
        ids = self.tk.encode(text)[: self.max_txt_len - 1] + [self.tk.eos_id]
        return patches, torch.tensor(ids, dtype=torch.long)

# ----------------------------------------------------------------------------
# 2. Hugging Face dataset wrapper (image+text or text‑only)
# ----------------------------------------------------------------------------
@register("hf")
class HFDataset(IterableDataset):
    def __init__(
        self,
        hf_id: str,
        split: str,
        tokenizer: BPETokenizer,
        text_col="text",
        image_col="image",
        streaming=False,
        max_txt_len: int = 256,
    ):
        from datasets import load_dataset  # lazy
        self.tk = tokenizer
        self.text_col, self.image_col = text_col, image_col
        self.max_txt_len = max_txt_len

        # Load once per worker; will be sharded below
        self.ds = load_dataset(hf_id, split=split, streaming=streaming)
        self.ds = self.ds.with_format("torch")

        self.img_tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
            transforms.ToTensor(),
        ])

    def _render_text(self, txt: str) -> Image.Image:
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), "white")
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()
        w, h = draw.textsize(txt, font=font)
        draw.text(((IMG_SIZE - w)//2, (IMG_SIZE - h)//2), txt,
                  fill="black", font=font)
        return img

    def __iter__(self):
        ds_iter = self.ds
        worker_info = get_worker_info()
        if worker_info is not None:
            ds_iter = ds_iter.shard(
                num_shards=worker_info.num_workers,
                index=worker_info.id,
            )

        # 3) Now yield examples from _your_ slice
        for i, ex in enumerate(ds_iter):
            text = ex[self.text_col]

            # image or synth
            if self.image_col in ex and ex[self.image_col] is not None:
                img = ex[self.image_col]
                if not isinstance(img, Image.Image):
                    img = transforms.ToPILImage()(img)
            else:
                img = self._render_text(text)

            # to tensor & patchify
            img = self.img_tf(img)
            patches = _patchify(img)
            if patches.shape[1] != 3 * PATCH**2:
                # print(f'Example {i} has shape {patches.shape} which is not {(IMG_SIZE**2 / PATCH**2, 3 * PATCH**2)}. Skipping.')
                continue

            # tokenize & yield
            ids = self.tk.encode(text)[: self.max_txt_len - 1] + [self.tk.eos_id]
            yield patches, torch.tensor(ids, dtype=torch.long)


# ----------------------------------------------------------------------------
# 3. On‑the‑fly synthetic text renderer (OCR pre‑training)
# ----------------------------------------------------------------------------
@register("synthetic")
class SyntheticDataset(IterableDataset):
    def __init__(self, corpus_path: str, tokenizer: BPETokenizer, fonts_dir: str | None = None, max_txt_len: int = 256):
        self.lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
        self.tk = tokenizer
        self.max_txt_len = max_txt_len
        self.fonts = list(Path(fonts_dir).glob("*.ttf")) if fonts_dir else []
        self.img_tf = transforms.ToTensor()

    def _render(self, txt: str) -> Image.Image:
        bg = Image.new("L", (IMG_SIZE, IMG_SIZE), 255)
        draw = ImageDraw.Draw(bg)
        font = ImageFont.truetype(str(random.choice(self.fonts)), 32) if self.fonts else ImageFont.load_default()
        w, h = draw.textsize(txt, font=font)
        draw.text(((IMG_SIZE - w)//2, (IMG_SIZE - h)//2), txt, fill=0, font=font)
        return bg.convert("RGB")

    def __iter__(self):
        while True:  # infinite stream - epoch control handled by DataLoader worker init seed
            txt = random.choice(self.lines)
            img = self.img_tf(self._render(txt))
            patches = _patchify(img)
            ids = self.tk.encode(txt)[: self.max_txt_len - 1] + [self.tk.eos_id]
            yield patches, torch.tensor(ids, dtype=torch.long)

# ----------------------------------------------------------------------------
# 4. Pre‑baked HDF5 dataset (PyTables)
# ----------------------------------------------------------------------------
@register("h5")
class H5Dataset(Dataset):
    def __init__(self, h5_path: str, tokenizer: BPETokenizer, max_txt_len: int = 256):
        import tables as tb
        self.h5 = tb.open_file(h5_path, mode="r")
        self.imgs = self.h5.root.images  # Array (N, 3, H, W) uint8
        self.caps = self.h5.root.captions  # VLArray utf‑8 strings
        self.tk = tokenizer
        self.max_txt_len = max_txt_len

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        img = torch.tensor(self.imgs[idx]).float() / 255.0  # [3,H,W]
        patches = _patchify(img)
        txt = self.caps[idx].decode("utf-8")
        ids = self.tk.encode(txt)[: self.max_txt_len - 1] + [self.tk.eos_id]
        return patches, torch.tensor(ids, dtype=torch.long)

# ----------------------------------------------------------------------------
# Collate function (shared by all datasets)
# ----------------------------------------------------------------------------

def collate(batch):
    """
    batch: list of (patches, token_ids)
      - patches: Tensor [N_patches, D]  (same D for all samples!)
      - token_ids: LongTensor [L_i]
    Returns:
      - patches: FloatTensor [B, N_patches, D]
      - token_ids: LongTensor [B, L_max] (padded)
    """
    patch_seqs, txt_seqs = zip(*batch)

    # stack patches directly (they all share the same shape [N, D])
    patches = torch.stack(patch_seqs, dim=0)  # → [B, N_patches, D]

    # pad text sequences to the max length in this batch
    max_len = max(t.shape[0] for t in txt_seqs)
    padded_txt = torch.stack([
        F.pad(t, (0, max_len - t.shape[0]), value=0)
        for t in txt_seqs
    ], dim=0)  # → [B, max_len]

    return patches, padded_txt, None

# def collate(batch, max_txt_len: int = 256, pad_id: int = 0):
#     """
#     batch: list of (patches[L,D], text_ids[T_i])
#     returns:
#         patches           : [B, L, D]
#         texts_padded      : [B, max_txt_len]           (full 256, no drop)
#         key_padding_mask  : [B, L + max_txt_len - 1]   (last text slot masked)
#     """
#     # ── split & basic shapes ──────────────────────────────────────────────
#     patches_list, texts_list = zip(*batch)
#     B        = len(patches_list)
#     L, D     = patches_list[0].shape                   # image-token count & dim
#     S_in     = L + max_txt_len - 1                    # sequence length seen by MHA
#     device   = patches_list[0].device

#     # ── 1. stack image patches ────────────────────────────────────────────
#     patches = torch.stack(patches_list, 0)             # [B, L, D]

#     # ── 2. pad / truncate text   (kept @ 256) ─────────────────────────────
#     texts_padded = torch.full((B, max_txt_len),
#                               pad_id, dtype=torch.long, device=device)
#     lengths = torch.zeros(B, dtype=torch.long, device=device)   # real lengths ≤ 256
#     for i, seq in enumerate(texts_list):
#         n = min(len(seq), max_txt_len)
#         texts_padded[i, :n] = seq[:n]
#         lengths[i] = n

#     # ── 3. build key-padding mask  [B, S_in] ─────────────────────────────
#     #     last text position (label) is *always* masked.
#     arange       = torch.arange(S_in, device=device)          # [S_in]
#     eff_lengths  = torch.clamp(lengths, max=max_txt_len-1)    # what the model *sees*
#     pad_starts   = L + eff_lengths.unsqueeze(1)               # [B,1]
#     key_pad_mask = arange.unsqueeze(0) >= pad_starts          # broadcast → [B,S_in]

#     return patches, texts_padded, key_pad_mask




# ----------------------------------------------------------------------------
# Public loader factory
# ----------------------------------------------------------------------------

def make_dataloader(kind: str, batch_size: int, num_workers: int, shuffle: bool, **dataset_kw):
    ds = build_dataset(kind, **dataset_kw)
    if isinstance(ds, IterableDataset):
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate, persistent_workers=True)