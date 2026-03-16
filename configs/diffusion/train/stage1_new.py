_base_ = ["image_new.py"]

# dataset = dict(memory_efficient=False)

# new config
grad_ckpt_settings = (8, 100)
# bucket_config = {
#     "_delete_": True,
#     "256px": {
#         1: (1.0, 96),
#         8: (1.0, 32),
#         #9: (1.0, 24),
#         16: (1.0, 16),
#         #17: (1.0, 12),
#         #24: (1.0, 10),
#         32: (1.0, 10),
#         64: (1.0, 6),
#         96: (1.0, 4),
#         128: (1.0, 3),
#         #1: (1.0, 240),
#         #6: (1.0, 48),
#         #12: (1.0, 24),
#         #24: (1.0, 12),
#         #32: (1.0, 8),
#         #64: (1.0, 4),
#         #90: (1.0, 3),
#         #128: (1.0,2),
#         #200: (0.5, 1)
#         #480: (1.0, 1),
#         #800: (1.0, 1),
#         #360: (1.0, 1),
#         #1024: (1.0, 1),
#     },
# }

video_clip_lens = [1, 8, 16, 32, 64, 96, 128]
batch_sizes = {
    1: 96,
    8: 32,
    16: 16,
    32: 10,
    64: 6,
    96: 4,
    128: 3,
}


model = dict(grad_ckpt_settings=grad_ckpt_settings)
lr = 5e-5
optim = dict(lr=lr)
ckpt_every = 2000
keep_n_latest = 20

#lr = 1e-4
#optim = dict(lr=lr)
cached_text = True
#cached_video = True
wandb = True
eval_every = 500
ckpt_every = 5000 #2000
epochs = 10000
async_io = False
guidance_scale = 4
valid_data_path = "./tmp"
total_steps = 200000
exp_name = "stage2"

# load = "/path/to/checkpoint/epoch-global_step"
load = None