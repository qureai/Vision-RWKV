import torch
from config import get_config
from models.vrwkv import VRWKV_Classification
import argparse

def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'vrwkv':
        model = VRWKV_Classification(
            img_size=config.MODEL.VRWKV.IMG_SIZE,
            patch_size=config.MODEL.VRWKV.PATCH_SIZE,
            in_channels=config.MODEL.VRWKV.IN_CHANNELS,
            out_indices=config.MODEL.VRWKV.OUT_INDICES,
            drop_rate=config.MODEL.VRWKV.DROP_RATE,  # drop after pos
            embed_dims=config.MODEL.VRWKV.EMBED_DIMS,
            depth=config.MODEL.VRWKV.DEPTH,
            drop_path_rate=config.MODEL.VRWKV.DROP_PATH_RATE,  # drop in blocks
            channel_gamma=config.MODEL.VRWKV.CHANNEL_GAMMA,
            shift_pixel=config.MODEL.VRWKV.SHIFT_PIXEL,
            shift_mode=config.MODEL.VRWKV.SHIFT_MODE,
            init_mode=config.MODEL.VRWKV.INIT_MODE,
            post_norm=config.MODEL.VRWKV.POST_NORM,
            k_norm=config.MODEL.VRWKV.K_NORM,
            init_values=config.MODEL.VRWKV.INIT_VALUES,
            hidden_rate=config.MODEL.VRWKV.HIDDEN_RATE,
            final_norm=config.MODEL.VRWKV.FINAL_NORM,
            interpolate_mode=config.MODEL.VRWKV.INTERPOLATE_MODE,

            with_cp=config.TRAIN.USE_CHECKPOINT,

            hidden_dims=config.MODEL.VRWKV.HIDDEN_DIMS,
            num_classes=config.MODEL.NUM_CLASSES
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/vrwkv_l_22kto1k_384.yaml')
    return parser.parse_args()

def main():
    args = parse_args()
    config = get_config(args)
    model = build_model(config)
    return model.eval()

if __name__ == '__main__':
    model = main()
    device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # calculate the number of parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f'Number of parameters: {num_params}')
    
    with torch.no_grad():
        out = model(torch.randn(128, 1, 1024, 1024).to(device))
        print(f"Output shape: {out.shape}")
        print(torch.cuda.memory_summary(device=device, abbreviated=False))
