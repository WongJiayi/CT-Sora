# Dataset settings
# dataset = dict(
#     type="ct-video",
#     root_dir="/path/to/your/latents/train/",
#     list_file="/path/to/your/latents/train/train.txt",
#     random_start=True,
#     return_name=True,
#     transform=None,   
# )
dataset = dict(
    type="video_text",
    transform_name="resize_crop",
    fps_max=24,  # the desired fps for training
    vmaf=True,  # load vmaf scores into text
)


grad_ckpt_settings = (8, 100)  # set the grad checkpoint settings
# bucket_config = {
#     "256px": {1: (1.0, 50)},
#     "768px": {1: (0.5, 11)},
#     "1024px": {1: (0.5, 7)},
# }

# Define model components
# model = dict(
#     type="flux",
#     from_pretrained=None,
#     strict_load=False,
#     guidance_embed=False,
#     fused_qkv=False,
#     use_liger_rope=True,
#     grad_ckpt_settings=grad_ckpt_settings,
#     # model architecture
#     in_channels=64,
#     vec_in_dim=768,
#     context_in_dim=768,
#     hidden_size=1536,
#     mlp_ratio=4.0,
#     num_heads=16,
#     depth=12,
#     depth_single_blocks=12,
#     axes_dim=[64, 16, 16],
#     theta=10_000,
#     qkv_bias=True,
# )
model = dict(
    type="flux",
    from_pretrained=None,
    strict_load=False,
    guidance_embed=False,
    fused_qkv=False,
    use_liger_rope=True,
    grad_ckpt_settings=grad_ckpt_settings,
    # model architecture
    in_channels=64,
    vec_in_dim=1,
    context_in_dim=1,
    hidden_size=2016,
    mlp_ratio=4.0,
    num_heads=24,
    depth=13,
    depth_single_blocks=15,
    axes_dim=[12, 36, 36],
    theta=10_000,
    qkv_bias=True,
    cond_embed=True,
)
# model = dict(
#     type="flux",
#     from_pretrained=None,
#     strict_load=False,
#     guidance_embed=False,
#     fused_qkv=False,
#     use_liger_rope=True,
#     grad_ckpt_settings=grad_ckpt_settings,
#     # model architecture
#     in_channels=64,
#     vec_in_dim=768,
#     context_in_dim=768,
#     hidden_size=3072,
#     mlp_ratio=4.0,
#     num_heads=24,
#     depth=19,
#     depth_single_blocks=38,
#     axes_dim=[16, 56, 56],
#     theta=10_000,
#     qkv_bias=True,
# )
dropout_ratio = {  # probability for dropout text embedding
    "t5": 0.31622777,
    "clip": 0.31622777,
}
ae = dict(
    type="hunyuan_vae",
    from_pretrained="ckpt/hunyuan_vae.safetensors",
    in_channels=3,
    out_channels=3,
    layers_per_block=2,
    latent_channels=16,
    use_spatial_tiling=True,
    use_temporal_tiling=False,
)
#is_causal_vae = True
t5 = dict(
    type="text_embedder",
    from_pretrained="google/t5-v1_1-xxl",
    cache_dir=None,
    max_length=512,
    shardformer=True,
)
clip = dict(
    type="text_embedder",
    from_pretrained="openai/clip-vit-large-patch14",
    cache_dir=None,
    max_length=77,
)

# Optimization settings
lr = 1e-5
eps = 1e-15
optim = dict(
    cls="HybridAdam",
    lr=lr,
    eps=eps,
    weight_decay=0.0,
    adamw_mode=True,
)
# optim = dict(
#     cls="AdamW",
#     lr=lr,
#     betas=(0.9, 0.999),
#     weight_decay=0.01,
# )
# optim = dict(
#     cls="muon",
#     # -------- Muon applies to hidden weights (ndim >= 2) --------
#     lr_muon=1e-3,
#     weight_decay_muon=1e-2,
#     # -------- AdamW applies to bias/gain/head/embedding --------
#     lr=3e-4,
#     betas=(0.9, 0.95),
#     weight_decay=1e-2,
#     grad_clip=1.0,
# )
warmup_steps = 0
update_warmup_steps = True

grad_clip = 1.0
accumulation_steps = 1
ema_decay = None

# Acceleration settings
prefetch_factor = 2
num_workers = 8
num_bucket_build_workers = 8
dtype = "bf16"
plugin = "zero2"
grad_checkpoint = True
plugin_config = dict(
    reduce_bucket_size_in_m=128,
    overlap_allgather=False,
)
pin_memory_cache_pre_alloc_numels = [(260 + 20) * 1024 * 1024] * 24 + [
    (34 + 20) * 1024 * 1024
] * 4
async_io = False

# Other settings
seed = 42
outputs = "outputs/"
epochs = 1000
log_every = 10
ckpt_every = 100
keep_n_latest = 20
wandb_project = "Open-Sora-Helma-test"

save_master_weights = False
load_master_weights = False

# For debugging
# record_time = True
# record_barrier = True
