import gc
import math
import os
import subprocess
import warnings
from contextlib import nullcontext
from copy import deepcopy
from pprint import pformat

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
gc.disable()


import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from colossalai.booster import Booster
from colossalai.utils import set_seed
from peft import LoraConfig
from tqdm import tqdm
from einops import rearrange

from opensora.acceleration.checkpoint import (
    GLOBAL_ACTIVATION_MANAGER,
    set_grad_checkpoint,
)
from opensora.acceleration.parallel_states import get_data_parallel_group
#from opensora.datasets.aspect import bucket_to_shapes
#from opensora.datasets.dataloader import prepare_dataloader
from opensora.datasets.pin_memory_cache import PinMemoryCache
from opensora.datasets.custom_video_dataloader import *
from opensora.models.mmdit.distributed import MMDiTPolicy
from opensora.registry import DATASETS, MODELS, build_module
from opensora.utils.ckpt import (
    CheckpointIO,
    model_sharding,
    record_model_param_shape,
    rm_checkpoints,
    load_json,
)
from opensora.utils.config import (
    config_to_name,
    create_experiment_workspace,
    parse_configs,
)
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
    print_mem,
    to_torch_dtype,
)
from opensora.utils.optimizer import create_lr_scheduler, create_optimizer
from opensora.utils.sampling import (
    get_res_lin_function,
    pack,
    prepare,
    prepare_ids,
    time_shift,
)
from opensora.utils.train import (
    create_colossalai_plugin,
    dropout_condition,
    get_batch_loss,
    prepare_visual_condition_causal,
    prepare_visual_condition_uncausal,
    set_eps,
    set_lr,
    setup_device,
    update_ema,
    warmup_ae,
)

torch.backends.cudnn.benchmark = False  # True leads to slow down in conv3d
import numpy as np
import cv2

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def build_valid_dataloader(cfg):
    """
    Returns a validation dataloader with B=3.
    Only called on the master rank:
      - not involved in bucketing
      - not involved in shuffling
    Dataset returns:
      {
        "video": [C,T,H,W],
        "path":  str
      }
    """
    import os
    import random
    import torch
    from torch.utils.data import DataLoader, Dataset
    import decord
    decord.bridge.set_bridge('torch')

    valid_dir = cfg.valid_data_path

    # collect all mp4 files
    all_videos = [
        os.path.join(valid_dir, name)
        for name in os.listdir(valid_dir)
        if name.endswith(".mp4")
    ]

    if len(all_videos) < 3:
        raise ValueError(f"[VALID] need >=3 videos, found {len(all_videos)}")

    # randomly select 3
    selected = random.sample(all_videos, 3)

    class ValidVideoDataset(Dataset):
        def __init__(self, paths):
            self.paths = paths

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, idx):
            path = self.paths[idx]

            vr = decord.VideoReader(path)
            num_frames = len(vr)

            T = 16
            frames = vr.get_batch(range(num_frames))  # [T0,H,W,3]

            if num_frames >= T:
                frames = frames[:T]
            else:
                last = frames[-1:].repeat(T - num_frames, 1, 1, 1)
                frames = torch.cat([frames, last], dim=0)

            # [T,3,H,W]
            frames = frames.permute(0, 3, 1, 2).float() / 255.0
            # model expects [C,T,H,W]
            frames = frames.permute(1, 0, 2, 3)

            return {
                "video": frames,  # [C,T,H,W]
                "path": path,
            }

    dataset = ValidVideoDataset(selected)

    loader = DataLoader(
        dataset,
        batch_size=3,      # load 3 videos at a time
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=False,
    )
    return loader



# ====================================================================
# ========== Core function 1: VAE decode ==========
# ====================================================================
def scale_latents(latents):
    latents -= 0.1159
    latents *= 0.3611
    return latents


def unscale_latents(latents):
    latents /= 0.3611
    latents += 0.1159
    return latents

def decode_volume(lat, vae):
    # lat: [B, C, T, H, W]

    # call decode directly, do not use .sample
    out = vae.decode(lat)     # returns [B, C, T, H, W]

    # convert back to [B, T, C, H, W] (consistent with training input)
    #out = out.permute(0, 2, 1, 3, 4)
    return out




# ====================================================================
# ========== Core function 2: model sampling (with input logic integrated) ==========
# ====================================================================
def unpack_ct_tokens(tokens, T, H, W, patch_size):
    """
    tokens: [B, N_tokens, D], where N_tokens = T * (H/ps) * (W/ps)
    returns: [B, C, T, H, W]
    """
    B, N, D = tokens.shape
    ph = pw = patch_size

    # number of patches per frame
    hw = N // T
    h = w = int((hw) ** 0.5)   # assume square
    assert h * w == hw, f"non-square grid: hw={hw}"

    C = D // (ph * pw)

    x = rearrange(
        tokens,
        'b (t h w) (c ph pw) -> b c t (h ph) (w pw)',
        t=T, h=h, w=w, ph=ph, pw=pw
    )
    return x

@torch.no_grad()
def run_sampling_moviegen(model, x0_unpacked, cfg, device, dtype):
    """
    Fully unconditional MovieGen v-pred multi-step sampling:
    - does not depend on cond_latent / text / image conditions
    - uses only random noise + time schedule + the v-pred formula from training
    """
    # ------- get shape info -------
    bs, C, T, H, W = x0_unpacked.shape
    patch_size = cfg.get("patch_size", 2)
    num_steps  = cfg.get("num_steps", 20)
    sigma_min  = cfg.get("sigma_min", 1e-5)

    # ------- 1. prepare dummy text / vector embeddings (unconditional: all-zero placeholders) -------
    # just to provide a valid shape to prepare_ids
    t5_embedding   = torch.zeros((bs, 1, 1), dtype=dtype, device=device)
    clip_embedding = torch.zeros((bs, 1),     dtype=dtype, device=device)

    # use x0_unpacked shape to build ids here
    dummy_x0 = torch.zeros_like(x0_unpacked, dtype=dtype, device=device)
    ids_dict = prepare_ids(dummy_x0, t5_embedding, clip_embedding)

    img_ids = ids_dict["img_ids"]
    txt     = ids_dict["txt"]
    txt_ids = ids_dict["txt_ids"]
    y_vec   = ids_dict["y_vec"]

    # ------- 2. initial x_T: Gaussian noise in token space -------
    # img token: [B, N_tokens, D]
    x_t = torch.randn_like(ids_dict["img"], dtype=torch.float32, device=device)
    x_t = x_t.to(dtype)

    # ------- 3. time schedule, consistent with training -------
    t_sigmoid = torch.linspace(1.0, 0.0, num_steps + 1,
                               device=device, dtype=dtype)

    shift_alpha = get_res_lin_function()((H * W) // 4)
    shift_alpha *= math.sqrt(T)  # same as training: add temporal influence

    # ========================================================
    #                    Sampling Loop
    # ========================================================
    for i in range(num_steps):
        # --- current / next sigmoidal t ---
        t_s = t_sigmoid[i]
        t_e = t_sigmoid[i + 1]

        # --- get actual t_curr, t_next via time_shift ---
        t_curr = time_shift(shift_alpha, t_s.view(1)).repeat(bs).to(dtype)  # [B]
        t_next = time_shift(shift_alpha, t_e.view(1)).repeat(bs).to(dtype)  # [B]

        # a(t), b(t) consistent with training
        t_rev = 1.0 - t_curr                   # [B]
        a = t_rev                              # [B]
        b = 1.0 - (1.0 - sigma_min) * t_rev    # [B]

        # Assemble model input (note: **no cond** here)
        inp = {
            "img":       x_t.to(device=device, dtype=dtype),
            "img_ids":   img_ids.to(device=device, dtype=dtype),
            "txt":       txt.to(device=device, dtype=dtype),
            "txt_ids":   txt_ids.to(device=device, dtype=dtype),
            "y_vec":     y_vec.to(device=device, dtype=dtype),
            "timesteps": t_curr.to(device=device, dtype=dtype),
            "guidance": torch.full(
                (bs,),
                cfg.get("guidance", 1.0),  # for unconditional, set to 1.0 or omit from model
                device=device,
                dtype=dtype,
            ),
        }

        # ---- model predicts v = (1-σ)x1 - x0 ----
        v = model(**inp)     # shape: [B, N_tokens, D]

        # ======================================================
        #          recover x0_hat, x1_hat from the training definition
        # ======================================================
        a_broadcast = a[:, None, None]                       # [B,1,1]
        b_broadcast = b[:, None, None]                       # [B,1,1]

        # c_v = b / (1 - sigma_min)
        c_v = b_broadcast / (1.0 - sigma_min)
        # d_v = a + c_v
        d_v = a_broadcast + c_v

        # x0_hat = (x_t - c_v * v) / d_v
        x0_hat = (x_t - c_v * v) / d_v

        # x1_hat = (v + x0_hat) / (1 - sigma_min)
        x1_hat = (v + x0_hat) / (1.0 - sigma_min)

        # ======================================================
        #         step to next timestep t_next using the same bridge formula
        # ======================================================
        t_next_rev = 1.0 - t_next                      # [B]
        a_next = t_next_rev                            # [B]
        b_next = 1.0 - (1.0 - sigma_min) * t_next_rev  # [B]

        a_next = a_next[:, None, None]    # [B,1,1]
        b_next = b_next[:, None, None]    # [B,1,1]

        # same formula as training: x_t = a(t)*x0 + b(t)*x1
        x_t = a_next * x0_hat + b_next * x1_hat

    # ======================================================
    #         final x_t (t≈0): x0_hat is the denoised output
    # ======================================================
    x_tokens = x_t
    x_pred_unpacked = unpack_ct_tokens(
        x_tokens, T=T, H=H, W=W, patch_size=patch_size
    )

    return x_pred_unpacked.to(dtype)


@torch.no_grad()
def run_sampling_moviegen_v(model, x0_unpacked, cond_latent, masks, cfg, device, dtype):
    """
    Correct MovieGen v-pred multi-step sampling:
    - During training:
        x_t = a(t) * x_0 + b(t) * x_1
        v   = (1 - sigma_min) * x_1 - x_0
        loss = MSE(model_pred, v)
    - During sampling:
        given cond_latent, start from pure noise and walk back to t=0 along the same bridge.
    """
    # ------- get shape info -------
    bs, C, T, H, W = x0_unpacked.shape
    patch_size = cfg.get("patch_size", 2)
    num_steps  = cfg.get("num_steps", 20)
    sigma_min  = cfg.get("sigma_min", 1e-5)

    # ------- 1. prepare dummy text embeddings (no text condition) -------
    t5_embedding   = torch.zeros((bs, 1, 1), dtype=dtype, device=device)
    clip_embedding = torch.zeros((bs, 1),     dtype=dtype, device=device)

    # only use shape to create ids (img_ids / txt_ids / y_vec), content does not matter
    dummy_x0 = torch.zeros_like(x0_unpacked, dtype=dtype, device=device)
    ids_dict = prepare_ids(dummy_x0, t5_embedding, clip_embedding)

    img_ids = ids_dict["img_ids"]
    txt     = ids_dict["txt"]
    txt_ids = ids_dict["txt_ids"]
    y_vec   = ids_dict["y_vec"]

    # ------- 2. initial x_T: Gaussian noise in token space -------
    # "img" token is packed: [B, N_tokens, D]
    x_t = torch.randn_like(ids_dict["img"], dtype=torch.float32, device=device)
    x_t = x_t.to(dtype)

    # ------- 3. Patchify cond latent -------
    cond_tokens = pack(cond_latent, patch_size=patch_size)  # [B, N_tokens_cond, D]

    # ------- 4. time schedule, consistent with training -------
    # t_sigmoid goes from 1->0, then apply time_shift to get actual t
    t_sigmoid = torch.linspace(1.0, 0.0, num_steps + 1, device=device, dtype=dtype)

    shift_alpha = get_res_lin_function()((H * W) // 4)
    shift_alpha *= math.sqrt(T)  # same as training: add temporal influence

    # ========================================================
    #                    Sampling Loop
    # ========================================================
    for i in range(num_steps):
        # --- current / next sigmoidal t ---
        t_s = t_sigmoid[i]
        t_e = t_sigmoid[i + 1]

        # --- get actual t_curr, t_next via time_shift ---
        t_curr = time_shift(shift_alpha, t_s.view(1)).repeat(bs).to(dtype)  # [B]
        t_next = time_shift(shift_alpha, t_e.view(1)).repeat(bs).to(dtype)  # [B]

        # a(t), b(t) consistent with training
        t_rev = 1.0 - t_curr                   # [B]
        a = t_rev                              # [B]
        b = 1.0 - (1.0 - sigma_min) * t_rev    # [B]

        # Assemble model input
        inp = {
            "cond":      cond_tokens.to(device=device, dtype=dtype),
            "img":       x_t.to(device=device, dtype=dtype),
            "img_ids":   img_ids.to(device=device, dtype=dtype),
            "txt":       txt.to(device=device, dtype=dtype),
            "txt_ids":   txt_ids.to(device=device, dtype=dtype),
            "y_vec":     y_vec.to(device=device, dtype=dtype),
            "timesteps": t_curr.to(device=device, dtype=dtype),
            # optionally add guidance if used in cfg:
            "guidance": torch.full(
                (bs,),
                cfg.get("guidance", 4),
                device=device,
                dtype=dtype,
            ),
        }

        # ---- model predicts v = (1-σ)x1 - x0 ----
        v = model(**inp)     # shape: [B, N_tokens, D]

        # ======================================================
        #          recover x0_hat, x1_hat from the training definition
        # ======================================================
        # expand dims to [B,1,1] for broadcasting
        a_broadcast = a[:, None, None]                       # [B,1,1]
        b_broadcast = b[:, None, None]                       # [B,1,1]

        # c_v = b / (1 - sigma_min)
        c_v = b_broadcast / (1.0 - sigma_min)
        # d_v = a + c_v
        d_v = a_broadcast + c_v

        # x0_hat = (x_t - c_v * v) / d_v
        x0_hat = (x_t - c_v * v) / d_v

        # x1_hat = (v + x0_hat) / (1 - sigma_min)
        x1_hat = (v + x0_hat) / (1.0 - sigma_min)

        # ======================================================
        #         step to next timestep t_next using the same bridge formula
        # ======================================================
        t_next_rev = 1.0 - t_next                      # [B]
        a_next = t_next_rev                            # [B]
        b_next = 1.0 - (1.0 - sigma_min) * t_next_rev  # [B]

        a_next = a_next[:, None, None]    # [B,1,1]
        b_next = b_next[:, None, None]    # [B,1,1]

        # same formula as training: x_t = a(t)*x0 + b(t)*x1
        x_t = a_next * x0_hat + b_next * x1_hat

    # ======================================================
    #         final x_t (t≈0): x0_hat is the denoised output
    # ======================================================
    x_tokens = x_t
    x_pred_unpacked = unpack_ct_tokens(
        x_tokens, T=T, H=H, W=W, patch_size=patch_size
    )

    return x_pred_unpacked.to(dtype)


@torch.no_grad()
def run_validation_from_dataloader(model, model_ae, dataloader, cfg, device, dtype, wandb_enabled=False):
    """
    Simple version: take one batch from the training dataloader, use the first 16 frames,
    construct x_0 and cond the same way as training, then run one sampling & visualization pass.
    """

    # 1. Fetch one batch from the dataloader (does not affect the training iterator)
    try:
        batch = next(iter(dataloader))
    except StopIteration:
        print("[VALID] Dataloader is empty, skip validation.")
        return

    x = batch["video"].to(device=device, dtype=dtype)  # [B,C,T,H,W]
    B, C, T, H, W = x.shape

    x_0 = model_ae.encode(x)
    x_rec = model_ae.decode(x_0)

    # Crop to first 16 frames
    MAX_FRAMES = 16
    if T > MAX_FRAMES:
        x = x[:, :, :MAX_FRAMES, :, :].contiguous()
        T = MAX_FRAMES
        print(f"[VALID] Cropped video frames to {T}")
    else:
        print(f"[VALID] Using all {T} frames")

    # 2. Same as training: choose function based on is_causal_vae,
    #    pass the original video x into prepare_visual_condition_*
    prepare_visual_condition = (
        prepare_visual_condition_causal
        if cfg.get("is_causal_vae", False)
        else prepare_visual_condition_uncausal
    )

    if cfg.get("condition_config", None) is not None:
        # ⚠️ Key: pass x (video), not latent
        x_0, cond = prepare_visual_condition(x, cfg.condition_config, model_ae)
    else:
        # No condition, encode directly
        x_0 = model_ae.encode(x)
        # Skip cond visualization here; if cond is needed, construct an all-zero mask manually
        cond = None

    # 3. Extract mask and cond_latent (if cond exists)
    if cond is not None:
        masks = cond[:, 0:1]   # [B,1,T,H_l,W_l]
        cond_latent = cond[:, 1:]
    else:
        masks = None
        cond_latent = None

    # 4. Sampling (using the same interface as before)
    # x_pred = run_sampling_moviegen_v(model, x_0, cond, masks, cfg, device, dtype)
    if cond is None or cond == {}:
        # unconditional sampling
        x_pred = run_sampling_moviegen(model, x_0, cfg, device, dtype)
    else:
        x_pred = run_sampling_moviegen_v(model, x_0, cond, masks, cfg, device, dtype)


    # 5. Decode GT & Pred
    video_gt   = decode_volume(x_0, model_ae)      # expected [B,T,C,H,W]
    video_pred = decode_volume(x_pred, model_ae)   # same [B,T,C,H,W]
    video_pred = (video_pred + 1) / 2

    # 6. Draw grid of cond / pred / gt
    def to_numpy(img):
        arr = (img.clamp(0,1)*255).byte().cpu().numpy()
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        return np.transpose(arr, (1, 2, 0))  # arr must be [C,H,W]


    cond_imgs, pred_imgs, gt_imgs = [], [], []

    if cond is not None:
        # Find the condition timestep cond_t for each sample
        cond_t_list = []
        _, _, T_lat, _, _ = masks.shape
        for i in range(B):
            flag = (masks[i, 0] > 0.5).view(T_lat, -1).any(dim=1)
            idx = torch.nonzero(flag).view(-1)
            cond_t = int(idx[0]) if idx.numel() > 0 else 0
            cond_t_list.append(cond_t)
    else:
        # No cond, use t=0 for all
        cond_t_list = [0] * B

    for i in range(B):
        t0 = cond_t_list[i]

        pred_img = to_numpy(video_pred[i, :, t0])   # FIXED
        gt_img   = to_numpy(video_gt[i, :, t0])     # FIXED

        if cond_latent is not None:
            cond_lat = cond_latent[i, :, :, :, :]    # [C_lat, T_lat, H_lat, W_lat]
            cond_dec = decode_volume(cond_lat.unsqueeze(0), model_ae)[0]  # [3,13,H,W]
            cond_img = to_numpy(cond_dec[:, t0])     # FIXED
        else:
            cond_img = to_numpy(video_gt[i, :, 0])   # e.g. first frame

        cond_imgs.append(cond_img)
        pred_imgs.append(pred_img)
        gt_imgs.append(gt_img)


    cols = [np.concatenate([cond_imgs[i], pred_imgs[i], gt_imgs[i]], axis=0) for i in range(B)]
    grid = np.concatenate(cols, axis=1)

    if wandb_enabled:
        wandb.log({"valid/preview": wandb.Image(grid)})

    print("[VALID] Step validation done.")
    return grid


def main():
    # ======================================================
    # 1. configs & runtime variables
    # ======================================================
    # == parse configs ==
    cfg = parse_configs()

    # == get dtype & device ==
    dtype = to_torch_dtype(cfg.get("dtype", "bf16"))
    device, coordinator = setup_device()
    grad_ckpt_buffer_size = cfg.get("grad_ckpt_buffer_size", 0)
    if grad_ckpt_buffer_size > 0:
        GLOBAL_ACTIVATION_MANAGER.setup_buffer(grad_ckpt_buffer_size, dtype)
    checkpoint_io = CheckpointIO()
    set_seed(cfg.get("seed", 1024))
    PinMemoryCache.force_dtype = dtype
    pin_memory_cache_pre_alloc_numels = cfg.get("pin_memory_cache_pre_alloc_numels", None)
    PinMemoryCache.pre_alloc_numels = pin_memory_cache_pre_alloc_numels

    # == init ColossalAI booster ==
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

    seq_align = plugin_config.get("sp_size", 1)

    # == init exp_dir ==
    exp_name, exp_dir = create_experiment_workspace(
        cfg.get("outputs", "./outputs"),
        model_name=config_to_name(cfg),
        config=cfg.to_dict(),
        exp_name=cfg.get("exp_name", None),  # useful for automatic restart to specify the exp_name
    )

    if is_log_process(plugin_type, plugin_config):
        os.system(f"chgrp -R share {exp_dir}")

    # == init logger, tensorboard & wandb ==
    logger = create_logger(exp_dir)
    logger.info("Training configuration:\n %s", pformat(cfg.to_dict()))
    tb_writer = None
    if coordinator.is_master():
        tb_writer = create_tensorboard_writer(exp_dir)
        if cfg.get("wandb", False):
            wandb.init(
                project=cfg.get("wandb_project", "Open-Sora-huny"),
                name=exp_name,
                config=cfg.to_dict(),
                dir=exp_dir,
                settings=wandb.Settings(start_method="thread"),
            )
    num_gpus = dist.get_world_size() if dist.is_initialized() else 1
    tp_size = cfg["plugin_config"].get("tp_size", 1)
    sp_size = cfg["plugin_config"].get("sp_size", 1)
    pp_size = cfg["plugin_config"].get("pp_size", 1)
    num_groups = num_gpus // (tp_size * sp_size * pp_size)
    logger.info("Number of GPUs: %s", num_gpus)
    logger.info("Number of groups: %s", num_groups)

    # ======================================================
    # 2. Build dataset & dataloader
    # ======================================================

    logger.info("Building dataset and dataloader...")

    dataloader, sampler = build_custom_video_dataloader(
        csv_path=cfg.dataset.data_path,
        clip_lens=cfg.video_clip_lens,
        batch_sizes=cfg.batch_sizes,
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        seed=cfg.get("seed", 42),
        rank=dist.get_rank() if dist.is_initialized() else 0,
        world_size=dist.get_world_size() if dist.is_initialized() else 1,
    )

    num_steps_per_epoch = len(dataloader)
    logger.info(f"Dataloader ready. Steps per epoch (for bookkeeping only): {num_steps_per_epoch}")

    # ======================================================
    # 3. build model
    # ======================================================
    logger.info("Building models...")

    # == build diffusion model ==
    model = build_module(cfg.model, MODELS, device_map=device, torch_dtype=dtype).train()
    if cfg.get("grad_checkpoint", True):
        set_grad_checkpoint(model)
    log_cuda_memory("diffusion")
    log_model_params(model)

    # == build EMA model ==
    use_lora = cfg.get("lora_config", None) is not None
    if cfg.get("ema_decay", None) is not None and not use_lora:
        ema = deepcopy(model).cpu().eval().requires_grad_(False)
        ema_shape_dict = record_model_param_shape(ema)
        logger.info("EMA model created.")
    else:
        ema = ema_shape_dict = None
        logger.info("No EMA model created.")
    log_cuda_memory("EMA")

    # == enable LoRA ==
    if use_lora:
        lora_config = LoraConfig(**cfg.get("lora_config", None))
        model = booster.enable_lora(
            model=model,
            lora_config=lora_config,
            pretrained_dir=cfg.get("lora_checkpoint", None),
        )
        log_cuda_memory("lora")
        log_model_params(model)

    # == build autoencoder ==
    if not cfg.get("cached_video", False):
        model_ae = build_module(cfg.ae, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
        log_cuda_memory("autoencoder")
        log_model_params(model_ae)
        # Compile encode only
        model_ae.encode = torch.compile(model_ae.encoder, dynamic=True)
    else:
        model_ae = None

    # # == build text encoders (optional, originally commented out, kept as-is) ==
    # if not cfg.get("cached_text", False):
    #     model_t5 = build_module(cfg.t5, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    #     log_cuda_memory("t5")
    #     log_model_params(model_t5)
    #
    #     model_clip = build_module(cfg.clip, MODELS, device_map=device, torch_dtype=dtype).eval().requires_grad_(False)
    #     log_cuda_memory("clip")
    #     log_model_params(model_clip)

    # == setup optimizer ==
    optimizer = create_optimizer(model, cfg.optim)

    # == setup lr scheduler ==
    lr_scheduler = create_lr_scheduler(
        optimizer=optimizer,
        num_steps_per_epoch=num_steps_per_epoch,
        epochs=cfg.get("epochs", 1000),
        warmup_steps=cfg.get("warmup_steps", None),
        use_cosine_scheduler=cfg.get("use_cosine_scheduler", False),
    )
    log_cuda_memory("optimizer")

    # =======================================================
    # 4. distributed training preparation with colossalai
    # =======================================================
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

    # == global variables ==
    log_step = 0
    acc_step = 0
    running_loss = 0.0
    timers = Timers(record_time=cfg.get("record_time", False), record_barrier=cfg.get("record_barrier", False))
    nsys = NsysProfiler(
        warmup_steps=cfg.get("nsys_warmup_steps", 2),
        num_steps=cfg.get("nsys_num_steps", 2),
        enabled=cfg.get("nsys", False),
    )
    valid_loader = None

    # =======================================================
    # 5. resume (GLOBAL STEP ONLY)
    # =======================================================
    load_master_weights = cfg.get("load_master_weights", False)
    save_master_weights = cfg.get("save_master_weights", False)

    # Use global_step as the sole recovery checkpoint
    start_global_step = cfg.get("start_step", None)  # if manually specified, treat as global_step override

    if cfg.get("load", None) is not None:
        logger.info("Loading checkpoint from %s", cfg.load)

        lr_scheduler_to_load = lr_scheduler
        if cfg.get("update_warmup_steps", False):
            lr_scheduler_to_load = None

        ret = checkpoint_io.load(
            booster,
            cfg.load,
            model=model,
            ema=ema,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler_to_load,
            sampler=None,  # no longer restoring sampler internal state, only restoring global_step
            include_master_weights=load_master_weights,
        )

        # Compatible with both tuple and dict return formats
        if isinstance(ret, dict):
            ckpt_global_step = ret.get("global_step", 0)
        else:
            # Legacy format: (epoch, step) or (epoch, step, global_step)
            if len(ret) >= 3:
                ckpt_global_step = ret[2]
            else:
                ckpt_epoch, ckpt_step = ret[0], ret[1]
                ckpt_global_step = ckpt_epoch * num_steps_per_epoch + ckpt_step

        if start_global_step is None:
            start_global_step = ckpt_global_step

        logger.info(
            "Loaded checkpoint %s, resume from global_step=%s (ckpt_global_step=%s)",
            cfg.load,
            start_global_step,
            ckpt_global_step,
        )

        # load optimizer and scheduler will overwrite some of the hyperparameters, so we need to reset them
        set_lr(optimizer, lr_scheduler, cfg.optim.lr, cfg.get("initial_lr", None))
        set_eps(optimizer, cfg.optim.eps)

        if cfg.get("update_warmup_steps", False):
            assert (
                cfg.get("warmup_steps", None) is not None
            ), "you need to set warmup_steps in order to pass --update-warmup-steps True"
            lr_scheduler.step(start_global_step)
            logger.info("The learning rate starts from %s", optimizer.param_groups[0]["lr"])

    if start_global_step is None:
        start_global_step = 0

    logger.info("Starting from global_step=%s", start_global_step)

    # == sharding EMA model ==
    if ema is not None:
        model_sharding(ema)
        ema = ema.to(device)
        log_cuda_memory("sharding EMA")

    # == warmup autoencoder ==
    if cfg.get("warmup_ae", False) and (model_ae is not None):
        shapes = bucket_to_shapes(cfg.get("bucket_config", None), batch_size=cfg.ae.batch_size)
        warmup_ae(model_ae, shapes, device, dtype)

    # =======================================================
    # 6. Prepare iteration helpers
    # =======================================================
    sigma_min = cfg.get("sigma_min", 1e-5)
    accumulation_steps = cfg.get("accumulation_steps", 1)
    ckpt_every = cfg.get("ckpt_every", 0)

    if cfg.get("is_causal_vae", False):
        prepare_visual_condition = prepare_visual_condition_causal
    else:
        prepare_visual_condition = prepare_visual_condition_uncausal

    @torch.no_grad()
    def prepare_inputs(batch):
        inp = dict()
        x = batch.pop("video")
        # Input is originally (B, T, C, H, W), VAE expects (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        bs = x.shape[0]

        # == encode video ==
        with nsys.range("encode_video"), timers["encode_video"]:
            if cfg.get("condition_config", None) is not None:
                # condition for i2v & v2v
                x_0, cond = prepare_visual_condition(x, cfg.condition_config, model_ae)
                cond = pack(cond, patch_size=cfg.get("patch_size", 2))
                inp["cond"] = cond
            else:
                if cfg.get("cached_video", False):
                    x_0 = batch.pop("video_latents").to(device=device, dtype=dtype)
                else:
                    x_0 = model_ae.encode(x)

        # == prepare timestep ==
        shift_alpha = get_res_lin_function()((x_0.shape[-1] * x_0.shape[-2]) // 4)
        shift_alpha *= math.sqrt(x_0.shape[-3])  # for image, T=1 so no effect
        t = torch.sigmoid(torch.randn((bs), device=device))
        t = time_shift(shift_alpha, t).to(dtype)

        # == text part currently uses zero placeholders (cached_text == True branch) ==
        if cfg.get("cached_text", False):
            t5_embedding = torch.zeros((bs, 1, 1), dtype=dtype, device=device)
            clip_embedding = torch.zeros((bs, 1), dtype=dtype, device=device)
            with nsys.range("encode_text"), timers["encode_text"]:
                inp_ = prepare_ids(x_0, t5_embedding, clip_embedding)
                inp.update(inp_)
                x_0 = pack(x_0, patch_size=cfg.get("patch_size", 2))
        else:
            # To enable a real text encoder in the future, revert to the original logic here
            raise NotImplementedError("Non-cached_text path is currently disabled in this main().")

        # == prepare noise vector ==
        x_1 = torch.randn_like(x_0, dtype=torch.float32).to(device, dtype)
        t_rev = 1 - t
        x_t = t_rev[:, None, None] * x_0 + (1 - (1 - sigma_min) * t_rev[:, None, None]) * x_1
        inp["img"] = x_t
        inp["timesteps"] = t.to(dtype)
        inp["guidance"] = torch.full((x_t.shape[0],), cfg.get("guidance", 4), device=x_t.device, dtype=x_t.dtype)

        return inp, x_0, x_1

    def run_iter(inp, x_0, x_1, step_for_acc):
        """
        step_for_acc: current global_step, used to determine whether to trigger optimizer.step()
        """
        if is_pipeline_enabled(plugin_type, plugin_config):
            inp["target"] = (1 - sigma_min) * x_1 - x_0  # follow MovieGen, modify V_t accordingly
            with nsys.range("forward-backward"), timers["forward-backward"]:
                data_iter = iter([inp])
                if cfg.get("no_i2v_ref_loss", False):
                    loss_fn = (
                        lambda out, input_: get_batch_loss(out, input_["target"], input_.pop("masks", None))
                        / accumulation_steps
                    )
                else:
                    loss_fn = (
                        lambda out, input_: F.mse_loss(out.float(), input_["target"].float(), reduction="mean")
                        / accumulation_steps
                    )
                loss = booster.execute_pipeline(data_iter, model, loss_fn, optimizer)["loss"]
                loss = loss * accumulation_steps if loss is not None else loss
                loss_item = all_reduce_mean(loss.data.clone().detach())
        else:
            with nsys.range("forward"), timers["forward"]:
                model_pred = model(**inp)  # B, T, L
                v_t = (1 - sigma_min) * x_1 - x_0
                if cfg.get("no_i2v_ref_loss", False):
                    loss = get_batch_loss(model_pred, v_t, inp.pop("masks", None))
                else:
                    loss = F.mse_loss(model_pred.float(), v_t.float(), reduction="mean")

            loss_item = all_reduce_mean(loss.data.clone().detach()).item()

            # == backward & update ==
            dist.barrier()
            with nsys.range("backward"), timers["backward"]:
                ctx = (
                    booster.no_sync(model, optimizer)
                    if cfg.get("plugin", "zero2") in ("zero1", "zero1-seq")
                    and (step_for_acc + 1) % accumulation_steps != 0
                    else nullcontext()
                )
                with ctx:
                    booster.backward(loss=(loss / accumulation_steps), optimizer=optimizer)

        with nsys.range("optim"), timers["optim"]:
            if (step_for_acc + 1) % accumulation_steps == 0:
                booster.checkpoint_io.synchronize()
                optimizer.step()
                optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step()

        # == update EMA ==
        if ema is not None:
            with nsys.range("update_ema"), timers["update_ema"]:
                update_ema(
                    ema,
                    model.unwrap(),
                    optimizer=optimizer,
                    decay=cfg.get("ema_decay", 0.9999),
                )

        return loss_item

    # =======================================================
    # 7. Global-step ONLY training loop
    # =======================================================
    dist.barrier()

    # total_steps is controlled by your config; cfg.total_steps must be set
    if not hasattr(cfg, "total_steps"):
        raise ValueError("Please set cfg.total_steps for global-step training.")
    max_train_steps = cfg.total_steps

    global_step = start_global_step
    logger.info(f"Training with global_step-only loop, from global_step={global_step}, total={max_train_steps}")

    pbar = tqdm(
        range(global_step, max_train_steps),
        desc="train",
        disable=not is_log_process(plugin_type, plugin_config),
    )

    dataloader_iter = iter(dataloader)

    while global_step < max_train_steps:
        nsys.step()

        # ---- load data ----
        with nsys.range("load_data"), timers["load_data"]:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                # dataloader exhausted → advance to next epoch (for sampler random seed only)
                epoch_idx = global_step // num_steps_per_epoch
                sampler.set_epoch(epoch_idx)
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)

            pinned_video = batch["video"]
            batch["video"] = pinned_video.to(device, dtype, non_blocking=True)

        # ---- run one iter ----
        with nsys.range("iter"), timers["iter"]:
            inp, x_0, x_1 = prepare_inputs(batch)
            loss = run_iter(inp, x_0, x_1, step_for_acc=global_step)

        # ---- update loss stats ----
        if loss is not None:
            running_loss += loss
            log_step += 1
            acc_step += 1

        actual_update_step = (global_step + 1) // accumulation_steps

        # ---- logging ----
        if (global_step + 1) % accumulation_steps == 0:
            if actual_update_step % cfg.get("log_every", 1) == 0:
                if is_log_process(plugin_type, plugin_config):
                    avg_loss = running_loss / log_step
                    pbar.set_postfix(
                        {
                            "loss": avg_loss,
                            "global_step": global_step,
                            "lr": optimizer.param_groups[0]["lr"],
                            "grad_norm": optimizer.get_grad_norm(),
                        }
                    )
                    if tb_writer is not None:
                        tb_writer.add_scalar("loss", loss, actual_update_step)

                    if cfg.get("wandb", False):
                        wandb_dict = {
                            "iter": global_step,
                            "acc_step": acc_step,
                            "loss": loss,
                            "avg_loss": avg_loss,
                            "lr": optimizer.param_groups[0]["lr"],
                            "eps": optimizer.param_groups[0]["eps"],
                            "global_grad_norm": optimizer.get_grad_norm(),
                        }
                        if cfg.get("record_time", False):
                            wandb_dict.update(timers.to_dict())
                        wandb.log(wandb_dict, step=actual_update_step)

                running_loss = 0.0
                log_step = 0

        # ---- validation ----
        if cfg.get("eval_every", 0) > 0 and global_step > 0 and global_step % cfg.eval_every == 0:
            dist.barrier()
            if coordinator.is_master():
                print(f"[VALID] Running validation at global_step={global_step}...")
                if valid_loader is None:
                    valid_loader = build_valid_dataloader(cfg)

                run_validation_from_dataloader(
                    model.unwrap(),
                    model_ae,
                    valid_loader,
                    cfg,
                    device,
                    dtype,
                    wandb_enabled=cfg.get("wandb", False),
                )
            dist.barrier()

        # ---- checkpoint & cache cleaning ----
        with nsys.range("clean_cache"), timers["clean_cache"]:
            if ckpt_every > 0 and actual_update_step % ckpt_every == 0 and coordinator.is_master():
                subprocess.run("sudo drop_cache", shell=True)

        with nsys.range("checkpoint"), timers["checkpoint"]:
            if ckpt_every > 0 and actual_update_step % ckpt_every == 0:
                gc.collect()

                # epoch / step here are only for legacy checkpoint naming, not used in training logic
                epoch_idx = global_step // num_steps_per_epoch
                step_idx = global_step % num_steps_per_epoch

                save_dir = checkpoint_io.save(
                    booster,
                    exp_dir,
                    model=model,
                    ema=ema,
                    optimizer=optimizer,
                    lr_scheduler=lr_scheduler,
                    sampler=sampler,
                    epoch=epoch_idx,
                    step=step_idx,
                    global_step=global_step,
                    batch_size=cfg.get("batch_size", None),
                    lora=use_lora,
                    actual_update_step=actual_update_step,
                    ema_shape_dict=ema_shape_dict,
                    async_io=cfg.get("async_io", False),
                    include_master_weights=save_master_weights,
                )

                if is_log_process(plugin_type, plugin_config):
                    os.system(f"chgrp -R share {save_dir}")

                logger.info(
                    "Saved checkpoint at global_step=%s (epoch_idx=%s, step_idx=%s) to %s",
                    global_step,
                    epoch_idx,
                    step_idx,
                    save_dir,
                )

                rm_checkpoints(exp_dir, keep_n_latest=cfg.get("keep_n_latest", -1))
                logger.info("Removed old checkpoints and kept %s latest ones.", cfg.get("keep_n_latest", -1))

        if cfg.get("record_time", False):
            # Previously used epoch, step directly; now use pseudo-epoch + local-step
            epoch_idx = global_step // num_steps_per_epoch
            step_idx = global_step % num_steps_per_epoch
            print(timers.to_str(epoch_idx, step_idx))

        pbar.update(1)
        global_step += 1

    pbar.close()
    log_cuda_max_memory("final")


if __name__ == "__main__":
    main()


# torchrun --nproc_per_node 4 scripts/diffusion/train_huny.py configs/diffusion/train/stage1_new.py --dataset.data-path /path/to/ct_videos.csv

"""
torchrun --nproc_per_node=1 --master_port=29505 \
    scripts/diffusion/train_huny.py \
    configs/diffusion/train/stage1_new.py \
    --dataset.data-path /path/to/ct_videos.csv
"""