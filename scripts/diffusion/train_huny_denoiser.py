# scripts/diffusion/train_stage2_v2v.py
# Stage2: video-to-video (condition=image2 -> target=image1) denoise training
# Scheme A (STRICT):
#   - Use MMDiT cond pathway: inp["cond"]
#   - inp["img"] is ONLY noisy tokens x_t (dim = in_channels, e.g. 64)
#   - DO NOT concat img with cond; DO NOT modify img_in
# v-pred target (MovieGen):
#   v = (1 - sigma_min) * x1 - x0
# W&B logging aligned to stage1:
#   iter/acc_step/epoch/loss/avg_loss/lr/eps/global_grad_norm, step=actual_update_step
# Validation:
#   keep 3x3 media grid: (cond, pred, gt) stacked vertically, 3 samples horizontally
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.cpu\.amp\.autocast.*deprecated.*",
)

import os
import gc
import math
import random
import subprocess
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from einops import rearrange

from colossalai.booster import Booster
from colossalai.utils import set_seed
from peft import LoraConfig

from opensora.acceleration.checkpoint import GLOBAL_ACTIVATION_MANAGER, set_grad_checkpoint
from opensora.models.mmdit.distributed import MMDiTPolicy
from opensora.registry import MODELS, build_module
from opensora.utils.ckpt import (
    CheckpointIO,
    model_sharding,
    record_model_param_shape,
    rm_checkpoints,
)
from opensora.utils.config import config_to_name, create_experiment_workspace, parse_configs
from opensora.utils.logger import create_logger
from opensora.utils.misc import (
    NsysProfiler,
    Timers,
    all_reduce_mean,
    create_tensorboard_writer,
    is_log_process,
    is_pipeline_enabled,
    log_cuda_max_memory,
    log_cuda_memory,
    log_model_params,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.sampling import get_res_lin_function, pack, prepare_ids, time_shift
from opensora.utils.train import (
    create_colossalai_plugin,
    set_eps,
    set_lr,
    setup_device,
    update_ema,
    warmup_ae,
)

# your denoise loader (the one you pasted)
# make sure this import path matches your repo layout
from opensora.datasets.denoiser_video_dataloader import build_denoise_dataloader  # noqa


# basic env
torch.backends.cudnn.benchmark = False
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


# ======================================================
# Helpers
# ======================================================
def _is_master() -> bool:
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


def unpack_ct_tokens(tokens: torch.Tensor, T: int, H: int, W: int, patch_size: int):
    """
    tokens: [B, N, D], where N = T * (H/ps) * (W/ps)
    returns: [B, C, T, H, W]
    """
    B, N, D = tokens.shape
    ph = pw = patch_size
    hw = N // T
    h = w = int(hw ** 0.5)
    assert h * w == hw, f"non-square grid: hw={hw}, h={h}, w={w}"
    C = D // (ph * pw)
    x = rearrange(tokens, "b (t h w) (c ph pw) -> b c t (h ph) (w pw)", t=T, h=h, w=w, ph=ph, pw=pw)
    return x


@torch.no_grad()
def decode_volume(lat_unpacked: torch.Tensor, ae):
    """
    lat_unpacked: [B, C, T, H, W] (latent)
    return: decoded tensor (usually [B, 3, T, H', W'] in [-1,1] or [0,1])
    """
    return ae.decode(lat_unpacked)


def _to_uint8_hwc(img_chw: torch.Tensor) -> np.ndarray:
    """
    img_chw: [C,H,W] in [0,1]
    returns uint8 [H,W,3]
    """
    x = img_chw.clamp(0, 1)
    x = (x * 255).byte().cpu().numpy()
    if x.shape[0] == 1:
        x = np.repeat(x, 3, axis=0)
    x = np.transpose(x, (1, 2, 0))
    return x


def make_3x3_grid(cond_vid, pred_vid, gt_vid, t_index_list):
    """
    cond_vid/pred_vid/gt_vid: torch [B, C, T, H, W] in [0,1]
    each sample column = (cond, pred, gt) stacked vertically
    """
    B = cond_vid.shape[0]
    cols = []
    for i in range(B):
        t0 = int(t_index_list[i])
        cond_img = _to_uint8_hwc(cond_vid[i, :, t0])
        pred_img = _to_uint8_hwc(pred_vid[i, :, t0])
        gt_img = _to_uint8_hwc(gt_vid[i, :, t0])
        col = np.concatenate([cond_img, pred_img, gt_img], axis=0)
        cols.append(col)
    return np.concatenate(cols, axis=1)


# ======================================================
# Strict Stage2 prepare_inputs (Scheme A)
# ======================================================
@torch.no_grad()
def prepare_inputs_stage2(batch, model, model_ae, cfg, device, dtype, nsys=None, timers=None):
    """
    batch must provide:
      - "video"     : target (image1), [B,1,T,H,W] or [B,3,T,H,W]
      - "condition" : condition (image2), [B,1,T,H,W] or [B,3,T,H,W]

    returns:
      inp: dict for model(**inp)
      x0:  packed x0 tokens [B,N,Ctok]
      x1:  packed x1 tokens [B,N,Ctok]
      x0_unpacked, cond_lat_unpacked: [B,C_lat,T,H_lat,W_lat]
    """
    inp = {}

    x_gt = batch["video"]      # target
    x_in = batch["condition"]  # condition
    bs = x_gt.shape[0]

    # VAE channel fix: 1->3
    if x_gt.shape[1] == 1:
        x_gt = x_gt.repeat(1, 3, 1, 1, 1)
    if x_in.shape[1] == 1:
        x_in = x_in.repeat(1, 3, 1, 1, 1)

    # encode (latent space)
    if nsys is not None and timers is not None:
        with nsys.range("encode_video"), timers["encode_video"]:
            x0_unpacked = model_ae.encode(x_gt.to(device=device, dtype=dtype))
            cond_lat_unpacked = model_ae.encode(x_in.to(device=device, dtype=dtype))
    else:
        x0_unpacked = model_ae.encode(x_gt.to(device=device, dtype=dtype))
        cond_lat_unpacked = model_ae.encode(x_in.to(device=device, dtype=dtype))

    sigma_min = cfg.get("sigma_min", 1e-5)
    patch_size = cfg.get("patch_size", 2)

    # timestep (same pattern as stage1)
    shift_alpha = get_res_lin_function()((x0_unpacked.shape[-1] * x0_unpacked.shape[-2]) // 4)
    shift_alpha *= math.sqrt(x0_unpacked.shape[-3])
    t = torch.sigmoid(torch.randn((bs,), device=device))
    t = time_shift(shift_alpha, t).to(dtype)

    # ids/text placeholders (cached_text pattern)
    t5_embedding = torch.zeros((bs, 1, 1), dtype=dtype, device=device)
    clip_embedding = torch.zeros((bs, 1), dtype=dtype, device=device)
    inp.update(prepare_ids(x0_unpacked, t5_embedding, clip_embedding))

    # pack
    x0 = pack(x0_unpacked, patch_size=patch_size)                 # [B,N,Ctok]
    cond_tokens = pack(cond_lat_unpacked, patch_size=patch_size)  # [B,N,Ctok]

    # noise x1 and x_t
    x1 = torch.randn_like(x0, dtype=torch.float32).to(device, dtype)
    t_rev = 1.0 - t
    x_t = t_rev[:, None, None] * x0 + (1.0 - (1.0 - sigma_min) * t_rev[:, None, None]) * x1  # [B,N,Ctok]

    # ---- STRICT scheme A checks ----
    m = model.unwrap() if hasattr(model, "unwrap") else model
    img_expected = m.img_in.in_features
    cond_embed = bool(getattr(getattr(m, "config", None), "cond_embed", False))

    assert cond_embed, "Scheme A requires model.config.cond_embed=True (so model uses inp['cond'])."
    assert img_expected == x_t.shape[-1], (
        f"Scheme A requires img token dim == img_in.in_features. "
        f"Got img_in={img_expected}, x_t_dim={x_t.shape[-1]}. "
        f"Do NOT concat. Fix model checkpoint/config to make in_channels match token dim."
    )

    inp["img"] = x_t.to(dtype)

    # MMDiT cond_in expects: in_channels + patch_size^2
    B, N, Ctok = cond_tokens.shape
    extra = torch.ones((B, N, patch_size * patch_size), device=device, dtype=dtype)
    inp["cond"] = torch.cat([cond_tokens.to(dtype), extra], dim=-1)

    inp["timesteps"] = t.to(dtype)
    # guidance is still required by some wrappers; guidance_embed can be False (then Identity)
    inp["guidance"] = torch.full((bs,), cfg.get("guidance", 4), device=device, dtype=dtype)

    return inp, x0, x1, x0_unpacked, cond_lat_unpacked


# ======================================================
# Sampling (MovieGen bridge inversion, conditional)
# ======================================================
@torch.no_grad()
def run_sampling_moviegen_v2v(model, x0_unpacked, cond_latent_unpacked, cfg, device, dtype):
    """
    Conditional MovieGen v-pred multi-step sampling, aligned with training:
      x_t = a(t)*x0 + b(t)*x1
      v   = (1-sigma_min)*x1 - x0
    Model inputs:
      img:  [B,N,Ctok]
      cond: [B,N,Ctok+p^2]
    """
    bs, C_lat, T, H, W = x0_unpacked.shape
    patch_size = cfg.get("patch_size", 2)
    num_steps = cfg.get("num_steps", 20)
    sigma_min = cfg.get("sigma_min", 1e-5)

    # ids placeholders
    t5_embedding = torch.zeros((bs, 1, 1), dtype=dtype, device=device)
    clip_embedding = torch.zeros((bs, 1), dtype=dtype, device=device)
    dummy_x0 = torch.zeros_like(x0_unpacked, dtype=dtype, device=device)
    ids = prepare_ids(dummy_x0, t5_embedding, clip_embedding)

    img_ids = ids["img_ids"]
    txt = ids["txt"]
    txt_ids = ids["txt_ids"]
    y_vec = ids["y_vec"]

    # cond tokens -> cond_inp
    cond_tokens = pack(cond_latent_unpacked, patch_size=patch_size)  # [B,N,Ctok]
    B, N, Ctok = cond_tokens.shape
    extra = torch.ones((B, N, patch_size * patch_size), device=device, dtype=dtype)
    cond_inp = torch.cat([cond_tokens.to(dtype), extra], dim=-1)

    # init x_T
    x_t = torch.randn((B, N, Ctok), device=device, dtype=torch.float32).to(dtype)

    # schedule (stage1 style)
    t_sigmoid = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)
    shift_alpha = get_res_lin_function()((H * W) // 4)
    shift_alpha *= math.sqrt(T)

    for i in range(num_steps):
        t_s = t_sigmoid[i]
        t_e = t_sigmoid[i + 1]

        t_curr = time_shift(shift_alpha, t_s.view(1)).repeat(B).to(dtype)
        t_next = time_shift(shift_alpha, t_e.view(1)).repeat(B).to(dtype)

        t_rev = 1.0 - t_curr
        a = t_rev
        b = 1.0 - (1.0 - sigma_min) * t_rev

        inp = {
            "img": x_t.to(device=device, dtype=dtype),
            "cond": cond_inp.to(device=device, dtype=dtype),
            "img_ids": img_ids.to(device=device, dtype=dtype),
            "txt": txt.to(device=device, dtype=dtype),
            "txt_ids": txt_ids.to(device=device, dtype=dtype),
            "y_vec": y_vec.to(device=device, dtype=dtype),
            "timesteps": t_curr.to(device=device, dtype=dtype),
            "guidance": torch.full((B,), cfg.get("guidance", 4), device=device, dtype=dtype),
        }

        v = model(**inp)  # [B,N,Ctok]

        a_b = a[:, None, None]
        b_b = b[:, None, None]
        c_v = b_b / (1.0 - sigma_min)
        d_v = a_b + c_v

        x0_hat = (x_t - c_v * v) / d_v
        x1_hat = (v + x0_hat) / (1.0 - sigma_min)

        t_next_rev = 1.0 - t_next
        a_next = t_next_rev
        b_next = 1.0 - (1.0 - sigma_min) * t_next_rev

        a_next = a_next[:, None, None]
        b_next = b_next[:, None, None]
        x_t = a_next * x0_hat + b_next * x1_hat

    x_pred_unpacked = unpack_ct_tokens(x_t, T=T, H=H, W=W, patch_size=patch_size)
    return x_pred_unpacked.to(dtype)


# ======================================================
# Build VALID loader (stage1-like: random pick 3)
# ======================================================
def build_valid_dataloader_stage2(cfg):
    """
    Build a validation dataloader with batch_size=3 for (video, condition) pairs.
    Uses the same dataset as training, but:
      - selects 3 random samples once
      - no shuffle
      - only used on master rank
    """
    import torch
    from torch.utils.data import DataLoader, Dataset

    data_dir = cfg.dataset.data_path  # your folder containing *_input.npy / *_gt.npy
    target_hw = cfg.get("valid_target_hw", 256)

    # scan input files
    all_inputs = sorted([f for f in os.listdir(data_dir) if f.endswith("_input.npy")])
    if len(all_inputs) < 3:
        raise ValueError(f"[VALID] need >=3 samples, found {len(all_inputs)} in {data_dir}")

    selected = random.sample(all_inputs, 3)

    class _ValidPairs(Dataset):
        def __init__(self, data_dir, names, clip_len=16, target_hw=256):
            self.data_dir = data_dir
            self.names = names
            self.clip_len = clip_len
            self.target_hw = target_hw

        def __len__(self):
            return len(self.names)

        def __getitem__(self, idx):
            inp_name = self.names[idx]
            inp_path = os.path.join(self.data_dir, inp_name)
            gt_path = inp_path.replace("_input.npy", "_gt.npy")

            vol_in = np.load(inp_path)  # [T,H,W] or [D,H,W]
            vol_gt = np.load(gt_path)

            T0 = vol_in.shape[0]
            L = self.clip_len
            if T0 >= L:
                indices = list(range(0, L))
            else:
                indices = list(range(T0)) + [T0 - 1] * (L - T0)

            vol_in = vol_in[indices]
            vol_gt = vol_gt[indices]

            t_in = torch.from_numpy(vol_in).unsqueeze(0).float()  # [1,L,H,W]
            t_gt = torch.from_numpy(vol_gt).unsqueeze(0).float()

            if t_in.shape[-1] != self.target_hw:
                t_in = F.interpolate(
                    t_in.unsqueeze(0),
                    size=(L, self.target_hw, self.target_hw),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
                t_gt = F.interpolate(
                    t_gt.unsqueeze(0),
                    size=(L, self.target_hw, self.target_hw),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)

            # normalize to [-1,1] (same as your dataset)
            t_in = (t_in / 127.5) - 1.0
            t_gt = (t_gt / 127.5) - 1.0

            return {"video": t_gt, "condition": t_in, "num_frames": L, "path": inp_path}

    ds = _ValidPairs(data_dir, selected, clip_len=16, target_hw=target_hw)
    loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    return loader


# ======================================================
# Validation: keep 3x3 media
# ======================================================
@torch.no_grad()
def run_validation_3x3(model, model_ae, valid_loader, cfg, device, dtype, global_step, actual_update_step, wandb_enabled=False):
    model.eval()

    try:
        batch = next(iter(valid_loader))
    except StopIteration:
        print("[VALID] valid_loader empty, skip")
        model.train()
        return None

    # move to device
    batch["video"] = batch["video"].to(device=device, dtype=dtype, non_blocking=True)
    batch["condition"] = batch["condition"].to(device=device, dtype=dtype, non_blocking=True)

    inp, x0, x1, x0_unpacked, cond_lat_unpacked = prepare_inputs_stage2(
        batch, model, model_ae, cfg, device, dtype
    )

    x_pred_unpacked = run_sampling_moviegen_v2v(
        model=model,
        x0_unpacked=x0_unpacked,
        cond_latent_unpacked=cond_lat_unpacked,
        cfg=cfg,
        device=device,
        dtype=dtype,
    )

    gt_vid = decode_volume(x0_unpacked, model_ae)
    cond_vid = decode_volume(cond_lat_unpacked, model_ae)
    pred_vid = decode_volume(x_pred_unpacked, model_ae)

    # assume decode is [-1,1] -> convert to [0,1]
    gt_vid = (gt_vid + 1) / 2
    cond_vid = (cond_vid + 1) / 2
    pred_vid = (pred_vid + 1) / 2

    # pick frame 0 for each of 3 samples
    t_list = [0, 0, 0]
    grid = make_3x3_grid(cond_vid, gt_vid, pred_vid, t_list)

    if wandb_enabled:
        wandb.log({"valid/preview": wandb.Image(grid)}, step=actual_update_step)

    print(f"[VALID] 3x3 preview logged at global_step={global_step}, update_step={actual_update_step}")
    model.train()
    return grid

# ======================================================
# Main
# ======================================================
def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    cfg = parse_configs()
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()

    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)

    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))

    plugin_type = cfg.get("plugin", "zero2")
    plugin_config = cfg.get("plugin_config", {})
    plugin_kwargs = {}
    if plugin_type == "hybrid":
        plugin_kwargs["custom_policy"] = MMDiTPolicy

    plugin = create_colossalai_plugin(
        plugin=plugin_type,
        dtype=cfg.get("dtype", "bf16"),
        grad_clip=cfg.get("grad_clip", 0),
        **plugin_config,
        **plugin_kwargs,
    )
    booster = Booster(plugin=plugin)

    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),
    )

    if is_log_process(plugin_type, plugin_config):
        os.system(f"chgrp -R share {exp_dir}")

    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))

    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "Open-Sora"),
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
                settings=wandb.Settings(start_method="thread"),
            )

    # ======================================================
    # Dataloader (your denoise loader)
    # ======================================================
    logger.info("Building stage2 denoise dataloader...")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1

    dataloader, sampler = build_denoise_dataloader(
        data_dir=cfg.dataset.data_path,
        batch_sizes=cfg.batch_sizes,
        clip_lens=cfg.video_clip_lens,
        num_workers=cfg.get("num_workers", 4),
        rank=rank,
        world_size=world,
        seed=cfg.get("seed", 42),
    )
    num_steps_per_epoch = len(dataloader)
    logger.info("Dataloader ready. Steps per epoch: %s", num_steps_per_epoch)

    # build valid loader on master only
    valid_loader = None
    if coordinator.is_master():
        valid_loader = build_valid_dataloader_stage2(cfg)

    # ======================================================
    # Build models
    # ======================================================
    logger.info("Building models...")

    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    log_cuda_memory("diffusion")
    log_model_params(model)

    # Strict scheme A sanity check before boosting
    assert getattr(model.config, "cond_embed", False), "Scheme A needs cond_embed=True in your cfg.model."
    assert model.img_in.in_features == getattr(model.config, "in_channels", model.img_in.in_features), \
        "img_in.in_features mismatch with config.in_channels. Likely wrong checkpoint/config."

    # EMA
    use_lora = cfg.get("lora_config", None) is not None
    if cfg.get("ema_decay", None) is not None and not use_lora:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = None
        ema_shape_dict = None
        logger.info("No EMA model created.")
    log_cuda_memory("EMA")

    # LoRA
    if use_lora:
        lora_config = LoraConfig(**cfg.get("lora_config", None))
        model = booster.enable_lora(
            model=model,
            lora_config=lora_config,
            pretrained_dir=cfg.get("lora_checkpoint", None),
        )
        log_cuda_memory("lora")
        log_model_params(model)

    # AE
    if not cfg.get("cached_video", False):
        model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory("autoencoder")
        log_model_params(model_ae)
        model_ae.encode = torch.compile(model_ae.encoder, dynamic=True)
    else:
        raise ValueError("Stage2 v2v requires AE encoding; please set cached_video=False.")

    # Optim & LR
    optimizer = create_optimizer(model, cfg.optim)
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )
    log_cuda_memory("optimizer")

    # ======================================================
    # Boost
    # ======================================================
    logger.info("Preparing for distributed training...")
    torch.set_default_dtype(dtype)
    model, optimizer, _, dataloader, lr_scheduler = booster.boost(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        dataloader=dataloader,
    )
    torch.set_default_dtype(torch.float)
    logger.info("Boosted model for distributed training")
    log_cuda_memory("boost")

    # timers / profiler
    timers = Timers(record_time=cfg.get("record_time", False), record_barrier=cfg.get("record_barrier", False))
    nsys = NsysProfiler(
        warmup_steps=cfg.get("nsys_warmup_steps", 2),
        num_steps=cfg.get("nsys_num_steps", 2),
        enabled=cfg.get("nsys", False),
    )

    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)

    # warmup AE
    if cfg.get("warmup_ae", False):
        # If you don't have bucket_config in stage2, you can skip this entirely
        try:
            from opensora.datasets.denoiser_video_dataloader import bucket_to_shapes
            shapes = bucket_to_shapes(cfg.get("bucket_config", None), batch_size=cfg.ae.batch_size)
            warmup_ae(model_ae, shapes, device, dtype)
        except Exception:
            if coordinator.is_master():
                print("[WARN] warmup_ae skipped (bucket_to_shapes not available or config missing).")

    # resume
    start_epoch = cfg.get("start_epoch", 0) or 0
    start_step = cfg.get("start_step", 0) or 0
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)

    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)
        lr_scheduler_to_load = lr_scheduler
        # ret = checkpoint_io.load(
        #     booster,
        #     cfg.load,
        #     model=model,
        #     ema=ema,
        #     optimizer=optimizer,
        #     lr_scheduler=lr_scheduler_to_load,
        #     sampler=None,
        #     include_master_weights=load_master_weights,
        # )
        # # best-effort
        # try:
        #     start_epoch, start_step = ret[0], ret[1]
        # except Exception:
        #     start_epoch, start_step = 0, 0

        # set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        # set_eps(optimizer, cfg.optim.eps)
        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=None,
            lr_scheduler=None,
            sampler=None,
            include_master_weights=False,  # stage2 typically does not need master weights
        )
        start_epoch, start_step = 0, 0  # more appropriate to restart from scratch for stage2
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        set_eps(optimizer, cfg.optim.eps)


    # shard EMA
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
    # ======================================================
    # Train loop (global-step tqdm, stage1-style logging)
    # ======================================================
    max_train_steps = cfg.get("max_train_steps", 200000)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    log_every = cfg.get("log_every", 50)
    eval_every = cfg.get("eval_every", 0)
    ckpt_every = cfg.get("ckpt_every", 0)
    sigma_min = cfg.get("sigma_min", 1e-5)

    # start points
    global_step = cfg.get("start_global_step", 0) or 0
    epoch = cfg.get("start_epoch", 0) or 0

    # important: set sampler epoch consistently at start
    if hasattr(sampler, "set_epoch"):
        sampler.set_epoch(epoch)

    dataloader_iter = iter(dataloader)

    running_loss = 0.0
    log_step = 0
    acc_step = 0

    # pbar: global-step based
    pbar = tqdm(
        range(global_step, max_train_steps),
        desc="train",
        disable=not is_log_process(plugin_type, plugin_config),
    )

    for gs in pbar:
        nsys.step()

        # ------------------------------------------------------
        # Fetch batch (DDP-safe reset)
        # ------------------------------------------------------
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            epoch_idx = gs // num_steps_per_epoch
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch_idx)
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)


        # move to device (pin_memory=True in loader)
        batch["video"] = batch["video"].to(device, dtype, non_blocking=True)
        batch["condition"] = batch["condition"].to(device, dtype, non_blocking=True)

        # ------------------------------------------------------
        # Forward / Loss
        # ------------------------------------------------------
        with nsys.range("iter"), timers["iter"]:
            inp, x0, x1, _, _ = prepare_inputs_stage2(
                batch, model, model_ae, cfg, device, dtype, nsys, timers
            )

            v_t = (1.0 - sigma_min) * x1 - x0

            with nsys.range("forward"), timers["forward"]:
                model_pred = model(**inp)
                loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")

            loss_item = all_reduce_mean(loss.detach()).item()

            # backward
            with nsys.range("backward"), timers["backward"]:
                ctx = (
                    booster.no_sync(model, optimizer)
                    if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq")
                    and ((gs + 1) % accumulation_steps != 0)
                    else nullcontext()
                )
                with ctx:
                    booster.backward(loss=(loss / accumulation_steps), optimizer=optimizer)

        # ------------------------------------------------------
        # Optim step
        # ------------------------------------------------------
        actual_update_step = (gs + 1) // accumulation_steps

        with nsys.range("optim"), timers["optim"]:
            if (gs + 1) % accumulation_steps == 0:
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()

            if lr_scheduler is not None:
                lr_scheduler.step()

        # ------------------------------------------------------
        # EMA
        # ------------------------------------------------------
        if ema is not None and (gs + 1) % accumulation_steps == 0:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(
                    ema,
                    model.unwrap(),
                    optimizer=optimizer,
                    decay=cfg.get("ema_decay", 0.9999),
                )

        # ------------------------------------------------------
        # Logging
        # ------------------------------------------------------
        running_loss += loss_item
        log_step += 1
        acc_step += 1

        if (gs + 1) % accumulation_steps == 0:
            if actual_update_step % log_every == 0:
                avg_loss = running_loss / max(1, log_step)

                # tqdm postfix
                if not pbar.disable:
                    pbar.set_postfix(
                        {
                            "loss": f"{avg_loss:.4f}",
                            "gs": gs,
                            "lr": optimizer.param_groups[0]["lr"],
                            "gn": optimizer.get_grad_norm() if hasattr(optimizer, "get_grad_norm") else None,
                            "epoch": epoch,
                        }
                    )

                # tb + wandb only on master
                if coordinator.is_master():
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss", loss_item, actual_update_step)

                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": gs,
                            "acc_step": acc_step,
                            "epoch": epoch,
                            "loss": loss_item,
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "eps": optimizer.param_groups[0].get("eps", None),
                            "global_grad_norm": optimizer.get_grad_norm()
                            if hasattr(optimizer, "get_grad_norm")
                            else None,
                        }
                        if cfg.get("record_time", False):
                            wandb_dict.update(timers.to_dict())
                        wandb.log(wandb_dict, step=actual_update_step)

                running_loss = 0.0
                log_step = 0

        # ------------------------------------------------------
        # Validation (master only)
        # ------------------------------------------------------
        if eval_every and (gs > 0) and (gs % eval_every == 0) and ((gs + 1) % accumulation_steps == 0):
            if dist.is_initialized():
                dist.barrier()
            if coordinator.is_master():
                run_validation_3x3(
                    model.unwrap(),
                    model_ae,
                    valid_loader,
                    cfg,
                    device,
                    dtype,
                    global_step=gs,
                    actual_update_step=actual_update_step,
                    wandb_enabled=cfg.get("wandb", False),
                )
            if dist.is_initialized():
                dist.barrier()

        # ------------------------------------------------------
        # Checkpoint
        # ------------------------------------------------------
        if ckpt_every and ((gs + 1) % accumulation_steps == 0) and (actual_update_step % ckpt_every == 0):
            if dist.is_initialized():
                dist.barrier()
            gc.collect()
            save_dir = checkpoint_io.save(
                booster,
                exp_dir,
                model=model,
                ema=ema,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                sampler=sampler,
                epoch=epoch,
                step=0,
                global_step=gs + 1,
                batch_size=cfg.get("batch_size", None),
                lora=use_lora,
                actual_update_step=actual_update_step,
                ema_shape_dict=ema_shape_dict,
                async_io=cfg.get("async_io", False),
                include_master_weights=save_master_weights,
            )
            if coordinator.is_master():
                logger.info("Saved checkpoint at gs=%s (update=%s) to %s", gs + 1, actual_update_step, save_dir)
            rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
            if dist.is_initialized():
                dist.barrier()

    pbar.close()



if __name__ == "__main__":
    main()

