from .shinepost import MyTransformer as MyT
from .shinepost_v2 import MyTransformer as MyTv2
from .shinepost_v3 import MyTransformer as MyTv3

# first : swin처럼 7,7,7,7에 pos는 1,1,1,0씩 넣어보기(depth도 동일) 단, 1/2로 나눠야 하므로 num_heads는 cswin을 따라 (2,4,8,16)
def build_model(args, is_pretrain = False):
    model_type = args.model

    import torch.nn as nn

    if model_type == 'shinepost_v1':
        model = MyTv3(
            img_size=args.img_size,
            in_chans=3,
            num_classes=1000,
            embed_dim=96,
            depth=[2, 2, 6, 2],
            # split_size=[2, 2, 2, 7],
            split_size=[7, 7, 7, 7],
            num_heads=[2, 4, 8, 16],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            use_chk=False,
            # pos=[2, 2, 2, 0]
            pos=[1, 1, 1, 0]
        )
        
    # v2 : embedding이 96 -> 64
    elif model_type == 'shinepost_v2':
        model = MyTv3(
            img_size = args.img_size,
            in_chans = 3,
            num_classes=1000,
            embed_dim=64,
            depth=[1,2,21,1],
            # split_size=[2, 2, 2, 7],
            split_size=[7, 7, 7, 7],
            num_heads=[2,4,8,16],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            use_chk=False,
            # pos=[2, 2, 2, 0]
            pos=[1, 1, 1, 0]
        )

    # v3 : embedding은 64, split_size와 pos를 변경
    elif model_type == 'shinepost_v3':
        model = MyTv3(
            img_size = args.img_size,
            in_chans = 3,
            num_classes=1000,
            embed_dim=64,
            depth=[2,2,6,2],
            # split_size=[7, 7, 7, 7],
            split_size=[2, 2, 2, 7],
            num_heads=[2,4,8,16],
            mlp_ratio=4.,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.,
            attn_drop_rate=0.,
            drop_path_rate=0.,
            norm_layer=nn.LayerNorm,
            use_chk=False,
            pos=[2, 2, 2, 0]
            # pos=[1, 1, 1, 0]
        )

    else :
        raise NotImplementedError(f"Unknown model : {model_type}")

    return model


'''
Namespace(
aa='rand-m9-mstd0.5-inc1', 
amp=True, 
apex_amp=False, 
aug_splits=0, 
batch_size=16, 
bn_eps=None, 
bn_momentum=None, 
bn_tf=False, 
channels_last=False, 
clip_grad=None, 
color_jitter=0.4, 
cooldown_epochs=10, 
crop_pct=None, 
cutmix=1.0, 
cutmix_minmax=None, 
data='/data/imagenet_small/', 
decay_epochs=30, 
decay_rate=0.1, 
device='cuda:1', 
dist_bn='', 
distributed=True, 
drop=0.0, 
drop_block=None, 
drop_connect=None, 
drop_path=0.2, 
epochs=300, 
eval_checkpoint='', 
eval_metric='top1', 
gp=None, hflip=0.5, 
img_size=224, 
initial_checkpoint='', 
interpolation='', 
jsd=False, 
local_rank=1, 
log_interval=50, 
lr=0.002, 
lr_cycle_limit=1, 
lr_cycle_mul=1.0, 
lr_noise=None, 
lr_noise_pct=0.67, 
lr_noise_std=1.0, 
mean=None, 
min_lr=1e-05, 
mixup=0.8, m
ixup_mode='batch', 
mixup_off_epoch=0, 
mixup_prob=1.0, 
mixup_switch_prob=0.5, 
model='CSWin_64_12211_tiny_224', 
model_ema=True, 
model_ema_decay=0.99984, 
model_ema_force_cpu=False, 
momentum=0.9, 
native_amp=False, 
no_aug=False, 
no_prefetcher=False, 
no_resume_opt=False, 
num_classes=1000, 
num_gpu=1, 
opt='adamw', 
opt_betas=None, 
opt_eps=None, 
output='', 
patience_epochs=10, 
pin_mem=False,
prefetcher=True, 
pretrained=False, 
rank=1, 
ratio=[0.75, 1.3333333333333333], 
recount=1, 
recovery_interval=0, 
remode='pixel', 
reprob=0.25, 
resplit=False, 
resume='', 
save_images=False, 
scale=[0.08, 1.0], 
sched='cosine', 
seed=42, 
smoothing=0.1, 
split_bn=False, 
start_epoch=None, 
std=None, 
sync_bn=False, 
train_interpolation='random', 
tta=0, 
use_chk=False, 
use_multi_epochs_loader=False, 
validation_batch_size_multiplier=1, 
vflip=0.0, 
warmup_epochs=20, 
warmup_lr=1e-06, 
weight_decay=0.05, workers=8, 
world_size=2)Namespace(aa='rand-m9-mstd0.5-inc1', 
amp=True, apex_amp=False, aug_splits=0, batch_size=16, bn_eps=None, 
bn_momentum=None, 
bn_tf=False, channels_last=False, clip_grad=None, color_jitter=0.4, 
cooldown_epochs=10, crop_pct=None, cutmix=1.0, 
cutmix_minmax=None, data='/data/imagenet_small/', 
decay_epochs=30, decay_rate=0.1, device='cuda:0', dist_bn='', 
distributed=True, drop=0.0, drop_block=None, drop_connect=None, drop_path=0.2, 
epochs=300, eval_checkpoint='', eval_metric='top1', gp=None, hflip=0.5, img_size=224, 
initial_checkpoint='', interpolation='', jsd=False, local_rank=0, log_interval=50, lr=0.002, 
lr_cycle_limit=1, lr_cycle_mul=1.0, lr_noise=None, lr_noise_pct=0.67, lr_noise_std=1.0, mean=None, 
min_lr=1e-05, mixup=0.8, mixup_mode='batch', 
mixup_off_epoch=0, mixup_prob=1.0, mixup_switch_prob=0.5, 
model='CSWin_64_12211_tiny_224', model_ema=True, model_ema_decay=0.99984, model_ema_force_cpu=False, 
momentum=0.9, native_amp=False, no_aug=False, no_prefetcher=False, no_resume_opt=False, num_classes=1000, 
num_gpu=1, opt='adamw', opt_betas=None, opt_eps=None, output='', 
patience_epochs=10, pin_mem=False, 
prefetcher=True, pretrained=False, rank=0, ratio=[0.75, 1.3333333333333333], 
recount=1, 
recovery_interval=0, 
remode='pixel', 
reprob=0.25, 
resplit=False, resume='', 
save_images=False, 
scale=[0.08, 1.0], 
sched='cosine', seed=42, smoothing=0.1, split_bn=False, start_epoch=None, 
std=None, 
sync_bn=False, 
train_interpolation='random', tta=0, use_chk=False, use_multi_epochs_loader=False, 
validation_batch_size_multiplier=1, vflip=0.0, warmup_epochs=20, warmup_lr=1e-06, 
weight_decay=0.05, 
workers=8, 
world_size=2
)

'''
