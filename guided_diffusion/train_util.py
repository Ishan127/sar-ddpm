import copy
import functools
import os

import blobfile as bf
import torch as th
from torch.optim import AdamW
import cv2
from . import logger
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
import numpy as np
import skimage
from skimage.metrics import peak_signal_noise_ratio as psnr
import math

INITIAL_LOG_LOSS_SCALE = 15.0

class TrainLoop:
    def __init__(
        self,
        *,
        model,
        diffusion,
        data,
        val_dat,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        args,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        self.model = model.to('cuda' if th.cuda.is_available() else 'cpu')
        self.diffusion = diffusion
        self.data = data
        self.val_data=val_dat
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.resume_checkpoint = resume_checkpoint
        self.args = args
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0

        self.sync_cuda = th.cuda.is_available()

        self._load_and_sync_parameters()
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=fp16_scale_growth,
        )

        self.opt = AdamW(
            self.mp_trainer.master_params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self._load_optimizer_state()
            self.ema_params = [
                self._load_ema_parameters(rate) for rate in self.ema_rate
            ]
        else:
            self.ema_params = [
                copy.deepcopy(self.mp_trainer.master_params)
                for _ in range(len(self.ema_rate))
            ]

    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            dict_load = th.load(resume_checkpoint, map_location='cuda' if th.cuda.is_available() else 'cpu')
            self.model.load_state_dict(dict_load, strict=False)

    def _load_ema_parameters(self, rate):
        ema_params = copy.deepcopy(self.mp_trainer.master_params)

        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        ema_checkpoint = find_ema_checkpoint(main_checkpoint, self.resume_step, rate)
        if ema_checkpoint:
            logger.log(f"loading EMA from checkpoint: {ema_checkpoint}...")
            state_dict = th.load(ema_checkpoint, map_location='cuda' if th.cuda.is_available() else 'cpu')
            ema_params = self.mp_trainer.state_dict_to_master_params(state_dict)

        return ema_params

    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = th.load(opt_checkpoint, map_location='cuda' if th.cuda.is_available() else 'cpu')
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        val_idx=0
        best_psnr = 0
        c = 0

        while (
            not self.lr_anneal_steps
            or self.step + self.resume_step < self.lr_anneal_steps
        ):
            batch, cond = next(self.data)
            self.run_step(batch, cond)

            if (self.step+1) % self.save_interval == 0:
                number=0
                print('validation')
                
                with th.no_grad():
                        c += 1
                        if c == 5:
                            break
                        val_idx=val_idx+1
                        psnr_val = 0
                        for batch_id1, data_var in enumerate(self.val_data):
                            clean_batch, model_kwargs1 = data_var
                            model_kwargs={}
                            for k, v in model_kwargs1.items():
                                if('Index' in k):
                                    img_name=v
                                else:
                                    model_kwargs[k]= v.to('cuda' if th.cuda.is_available() else 'cpu')

                            sample = self.diffusion.p_sample_loop(
                                self.model,
                                (clean_batch.shape[0], 3, 256,256),
                                clip_denoised=True,
                                model_kwargs=model_kwargs,
                            )

                            sample = ((sample + 1) * 127.5)
                            sample = sample.clamp(0, 255).to(th.uint8)
                            sample = sample.permute(0, 2, 3, 1)
                            sample = sample.contiguous().cpu().numpy()

                            number=number+1
                            
                            clean_image = ((model_kwargs['HR']+1)* 127.5).clamp(0, 255).to(th.uint8)
                            clean_image= clean_image.permute(0, 2, 3, 1)
                            clean_image= clean_image.contiguous().cpu().numpy()

                            clean_image = clean_image[0][:,:,::-1]
                            sample = sample[0][:,:,::-1]
                            clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2GRAY)
                            sample = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY)
                            
                            psnr_im = psnr(clean_image,sample)
                            psnr_val = psnr_val + psnr_im

                        psnr_val = psnr_val/number

                        print('psnr =')
                        print(psnr_val)

                        if best_psnr < psnr_val:
                            best_psnr = psnr_val
                            self.save_val()

            self.step += 1

    def run_step(self, batch, cond):
        self.forward_backward(batch, cond)
        took_step = self.mp_trainer.optimize(self.opt)
        if took_step:
            self._update_ema()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond):
        self.mp_trainer.zero_grad()
        num_im = 0
        loss_wandb = 0
        for i in range(0, batch.shape[0], self.microbatch):
            num_im = num_im + 1
            
            micro = batch[i : i + self.microbatch].to('cuda' if th.cuda.is_available() else 'cpu')
            micro_cond = {
                k: v[i : i + self.microbatch].to('cuda' if th.cuda.is_available() else 'cpu')
                for k, v in cond.items()
            }
            last_batch = (i + self.microbatch) >= batch.shape[0]
            t, weights = self.schedule_sampler.sample(micro.shape[0], 'cuda' if th.cuda.is_available() else 'cpu')

            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                micro,
                t,
                model_kwargs=micro_cond,
            )
            if last_batch:
                losses = compute_losses()
            else:
                losses = compute_losses()

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            loss = (losses["loss"] * weights).mean()
            loss_wandb = th.log10(loss) + loss_wandb

            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            self.mp_trainer.backward(loss)
        loss_wandb_f = loss_wandb/num_im

    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join("./weights", filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

    def save_val(self):
        def save_checkpoint_val(rate, params):
            state_dict = self.mp_trainer.master_params_to_state_dict(params)
            logger.log(f"saving model {rate}...")
            if not rate:
                filename = f"model{(self.step+self.resume_step):06d}.pt"
            else:
                filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
            with bf.BlobFile(bf.join("./weights", filename), "wb") as f:
                th.save(state_dict, f)

        save_checkpoint_val(0, self.mp_trainer.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint_val(rate, params)

        with bf.BlobFile(
            bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
            "wb",
        ) as f:
            th.save(self.opt.state_dict(), f)

def parse_resume_step_from_filename(filename):
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_blob_logdir():
    return logger.get_dir()

def find_resume_checkpoint():
    return None

def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None

def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
