"""
SAR-DDPM Inference on real SAR images.
"""

import argparse
import torch
import os
import cv2
import numpy as np

import torch.nn.functional as F

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    sr_model_and_diffusion_defaults,
    sr_create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
from torch.utils.data import DataLoader
from torch.optim import AdamW

from scripts.valdata import  ValData, ValDataNew, ValDataNewReal
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

val_dir = r'/content/resized_patches/resized_patches'
base_path = r'/content/resized_patches/pred'
resume_checkpoint_clean = r'/content/ddpmtrainedmodel/ddpmtrainedmodel/model000409.pt'


def main():
    args = create_argparser().parse_args()

    print(args)
    
    model_clean, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    val_data = DataLoader(ValDataNewReal(dataset_path=val_dir), batch_size=64, shuffle=False, num_workers=16)  #load_superres_dataval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model_clean.load_state_dict(torch.load(resume_checkpoint_clean, map_location=device))

    model_clean.to(device)
    
    if args.use_fp16:
        model_clean.half()
    else:
        model_clean.float()

    params =  list(model_clean.parameters())

    print('model clean device:')
    print(next(model_clean.parameters()).device)

    with torch.no_grad(): 
        number = 0

        for batch_id1, data_var in enumerate(val_data):
            number = number+1 
            clean_batch, model_kwargs1 = data_var

            single_img = model_kwargs1['SR'].to(dist_util.dev())

            count = 0
            [t1,t2,max_r,max_c] = single_img.size()
            
            N = 9
            
            val_inputv = single_img.clone()

            
            for row in range(0,max_r,100):
                for col in range(0,max_c,100):
                    
                    val_inputv[:,:,:row,:col] = single_img[:,:,max_r-row:,max_c-col:]
                    val_inputv[:,:,row:,col:] = single_img[:,:,:max_r-row,:max_c-col]
                    val_inputv[:,:,row:,:col] = single_img[:,:,:max_r-row,max_c-col:]
                    val_inputv[:,:,:row,col:] = single_img[:,:,max_r-row:,:max_c-col]

                    model_kwargs = {}
                    val_inputv = val_inputv.half() if args.use_fp16 else val_inputv.float()
                    for k, v in model_kwargs1.items():
                        if "Index" in k:
                            img_name = v
                        elif "SR" in k:
                            model_kwargs[k] = val_inputv
                        else:
                            model_kwargs[k] = v.half() if args.use_fp16 else v.float()

                    with torch.amp.autocast(device_type='cuda', enabled=args.use_fp16):  # Apply autocast here
                        sample = diffusion.p_sample_loop(
                            model_clean,
                            (clean_batch.shape[0], 3, 256, 256),
                            clip_denoised=True,
                            model_kwargs=model_kwargs,
                            device=device,
                            progress=True,
                        )

                    if args.use_fp16:
                        sample = sample.half()
                    else:
                        sample = sample.float()

                    if count==0:
                        sample_new = (1.0/N)*sample
                    else : 
                        sample_new[:,:,max_r-row:,max_c-col:] = sample_new[:,:,max_r-row:,max_c-col:] + (1.0/N)*sample[:,:,:row,:col]
                        sample_new[:,:,:max_r-row,:max_c-col] = sample_new[:,:,:max_r-row,:max_c-col] + (1.0/N)*sample[:,:,row:,col:]
                        sample_new[:,:,:max_r-row,max_c-col:] = sample_new[:,:,:max_r-row,max_c-col:] + (1.0/N)*sample[:,:,row:,:col]
                        sample_new[:,:,max_r-row:,:max_c-col] = sample_new[:,:,max_r-row:,:max_c-col] + (1.0/N)*sample[:,:,:row,col:]
                        
                    count += 1

            sample_new = ((sample_new + 1) * 127.5)
            sample_new = sample_new.clamp(0, 255).to(torch.uint8)
            sample_new = sample_new.permute(0, 2, 3, 1)
            sample_new = sample_new.contiguous().cpu().numpy() # might have to put to CUDAs
            
            for i in range(sample_new.shape[0]):
                snew = sample_new[i][:,:,::-1]
                snew = cv2.cvtColor(snew, cv2.COLOR_BGR2GRAY)
                print(img_name[i])
                cv2.imwrite(base_path+'pred_'+img_name[i],snew)

def create_argparser():
    defaults = dict(
        data_dir= val_dir,
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=1,
        ema_rate="0.9999",
        log_interval=5, #100 earlier
        save_interval=10, #200 earlier
        use_fp16=False,
        fp16_scale_growth=1e-3
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()