"""
Train SAR-DDPM model.
"""

import argparse

import torch.nn.functional as F

from guided_diffusion import logger
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
import torch
# from train_dataset import TrainData
from scripts.valdata import  ValData, ValDataNew

train_dir = r'/content/data/twofivesixx/train'
   
val_dir = r'/content/data/twofivesixx/val'

pretrained_weight_path = r'/content/data/weights/64_256_upsampler.pt'


def main():
    args = create_argparser().parse_args()

    logger.configure()

    logger.log("creating model...")
    model, diffusion = sr_create_model_and_diffusion(
        **args_to_dict(args, sr_model_and_diffusion_defaults().keys())
    )
    model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")

    val_data = DataLoader(ValDataNew(dataset_path=val_dir), batch_size=1, shuffle=False, num_workers=1) 

    print(args)
    data = load_sar_data(
        args.data_dir,
        train_dir,
        args.batch_size,
        large_size=256,
        small_size=256, # was 256 earlier ~Vishrut
        class_cond=False,
    )
    

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        val_dat=val_data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        args = args,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def load_sar_data(data_dir,gt_dirs, batch_size, large_size, small_size, class_cond=False):
    data = load_data(
        data_dir=data_dir,
        gt_dir=gt_dirs,
        batch_size=batch_size,
        image_size=large_size,
        class_cond=False,
    )
    for large_batch, model_kwargs in data:
        yield large_batch, model_kwargs


def create_argparser():
    defaults = dict(
        data_dir = train_dir,
        schedule_sampler="uniform",
        lr=1e-4,
        # lr=5e-5,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=2,
        microbatch=1,
        ema_rate="0.9999",
        log_interval=50,
        save_interval=25,
        resume_checkpoint=pretrained_weight_path,
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(sr_model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()
