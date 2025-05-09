"""Distributed training entry‑point.
Now supports any dataset kind registered in data.py.

Example (single node 8×GPU, HF OpenImages micro):
    torchrun --standalone -n 8 src/train.py \
        --kind hf \
        --hf_id sizhkhy/open-images-captions-micro \
        --tokenizer tokenizer.model \
        --batch 4 --epochs 1
"""
import functools
import argparse, os, math, inspect
import time
import debugpy
import wandb

import torch
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from functools import partial
from model import TransformerBlock  # for FSDP auto-wrap policy
from torch.distributed import init_process_group
from torch.utils.data import DistributedSampler

from tokenizer import BPETokenizer
from data import make_dataloader
from model import MultiModalLM

# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def ddp_setup(rank, world_size):
    init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

# def evaluate(model, dl, loss_fn, rank):
#     """Return mean loss over `dl` (DDP-aware)."""
#     model.eval()
#     with torch.no_grad():
#         total, n = 0.0, 0
#         for patches, txt, mask in dl:
#             patches = patches.cuda(rank, non_blocking=True)
#             txt     = txt.cuda(rank, non_blocking=True)
#             logits  = model(patches, txt[:, :-1], mask)

#             loss = loss_fn(
#                 logits.reshape(-1, logits.size(-1)),
#                 txt[:, 1:].reshape(-1),
#             )
#             bs = txt.size(0)
#             total += loss.item() * bs
#             n     += bs

#     # Average across GPUs
#     total = torch.tensor([total], device=rank)
#     n     = torch.tensor([n    ], device=rank)
#     torch.distributed.all_reduce(total, op=torch.distributed.ReduceOp.SUM)
#     torch.distributed.all_reduce(n,     op=torch.distributed.ReduceOp.SUM)
#     model.train()
#     return (total / n).item()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main(cfg):
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    ddp_setup(rank, world)

    # ----------------------------- data -------------------------------------
    tk = BPETokenizer(cfg.tokenizer)

    ds_kwargs = {k: v for k, v in vars(cfg).items() if k not in {
        "kind", "batch", "num_workers", "shuffle", "tokenizer", "epochs", "lr"}}

    dl = make_dataloader(kind=cfg.kind,
                         batch_size=cfg.batch,
                         num_workers=cfg.num_workers,
                         shuffle=(rank == 0 and cfg.shuffle),
                         tokenizer=tk,
                         **ds_kwargs)

    # when using map‑style dataset we need sampler for DDP
    if isinstance(dl.dataset, torch.utils.data.Dataset) and not isinstance(dl.dataset, torch.utils.data.IterableDataset):
        dl.sampler = DistributedSampler(dl.dataset, num_replicas=world, rank=rank)

    # ----------------------------- model ------------------------------------
    model = MultiModalLM(tk.vocab_size).to(rank)
    # Wrap transformer blocks via auto-wrap policy
    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls=(TransformerBlock,),
    )
    model = FSDP(model, auto_wrap_policy=wrap_policy,
                    cpu_offload=CPUOffload(offload_params=False))

    if rank == 0:
        wandb.init(
            name=f'{time.strftime("%m%d")}-{cfg.run_name}',
            project="sanskrit-ocr",
            config=vars(cfg),
            tags=["hf-dataset", cfg.kind],
        )
        # optional: watch gradients & parameter histograms
        wandb.watch(model, log="all", log_freq=20)

        total_params = sum(p.numel() for p in model.parameters())
        print(f"Params: {total_params/1e6:.1f}M")

    opt = optim.AdamW(model.parameters(), lr=cfg.lr)

    try:
        steps_per_epoch = math.ceil(len(dl) / world)
    except TypeError:          # IterableDataset has no len()
        # fallback:  one epoch == this many gradient steps
        steps_per_epoch = 40          # ← or load from cfg
        print(f"Setting {steps_per_epoch=}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs * steps_per_epoch)
    loss_fn = nn.CrossEntropyLoss(ignore_index=tk.padding_idx)

    # ----------------------------- train loop -------------------------------
    for epoch in range(cfg.epochs):
        if hasattr(dl, "sampler") and isinstance(dl.sampler, DistributedSampler):
            dl.sampler.set_epoch(epoch)
        for step, (patches, txt, mask) in enumerate(dl):
            opt.zero_grad()

            patches, txt = patches.cuda(rank, non_blocking=True), txt.cuda(rank, non_blocking=True)
            logits = model(patches, txt[:, :-1], mask)

            loss = loss_fn(logits.reshape(-1, tk.vocab_size), txt[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            scheduler.step()
            torch.cuda.empty_cache() # NOTE: try padding all batches to same max len and turn this off

            if rank == 0:
                print(f'{step=} {patches.shape=} {txt.shape=}')
                print(f'{txt[:10, 5]}')
                print(f'{loss.item()=}')
                wandb.log({
                    "train/loss": loss.item(),
                    "train/step": epoch * steps_per_epoch + step,
                    "train/lr": opt.param_groups[0]["lr"],
                })

        if rank == 0:
            print(f"Epoch {epoch}/{cfg.epochs} loss={loss.item():.4f}")

            # # 4) Log per‐epoch metrics (e.g. validation)
            # val_loss = evaluate(model, val_dl, loss_fn, rank)
            # wandb.log({
            #     "val/loss": val_loss,
            #     "epoch": epoch,
            # })
    

    if rank == 0:
        wandb.finish()
        torch.save(model.state_dict(), "ckpt.pt")

# -----------------------------------------------------------------------------
# arg parsing
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # generic
    p.add_argument("--kind", default="hf", choices=["hf", "tsv", "synthetic", "h5"], help="Which dataset backend to use")
    p.add_argument("--tokenizer", default="tokenizer.model")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--shuffle", action="store_true", default=False, help="Shuffle for map‑style datasets")

    # hf specific
    p.add_argument("--hf_id", default=None, help="🤗 dataset id, e.g. sizhkhy/open-images-captions-micro")
    p.add_argument("--split", default="train")
    p.add_argument("--text_col", default="text")
    p.add_argument("--image_col", default="image")
    p.add_argument("--streaming", action="store_true")

    # tsv / h5 / synthetic paths
    p.add_argument("--meta", help="metadata.tsv path (kind=tsv)")
    p.add_argument("--h5_path")
    p.add_argument("--corpus_path")
    p.add_argument("--fonts_dir")

    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--debug", type=bool, default=False)

    cfg = p.parse_args()

    if cfg.debug:
        rank = int(os.getenv("RANK", "-1"))
        port = rank + 5678
        print(f'Initializing debugpy on {port}')
        debugpy.listen(("0.0.0.0", port))
        debugpy.wait_for_client()

    main(cfg)