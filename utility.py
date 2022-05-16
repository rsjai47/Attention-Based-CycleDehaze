import torch
import torch.nn as nn
import os
import numpy as np
from option import opt


def save_checkpoint(gen_C, gen_H, disc_C, disc_H, opt_gen, opt_disc, start_step, max_ssim, max_psnr, ssims, psnrs, filename=opt.checkpoint_model):
    print("=> Saving checkpoint")
    checkpoint = {
        "gen_C": gen_C.state_dict(),
        "gen_H": gen_H.state_dict(),
        "disc_C": disc_C.state_dict(),
        "disc_H": disc_H.state_dict(),
        "opt_gen": opt_gen.state_dict(),
        "opt_disc": opt_disc.state_dict(),
        "start_step": start_step,
        "max_psnr": max_psnr,
        "max_ssim": max_ssim,
        "ssims": ssims,
        "psnrs": psnrs
    }
    torch.save(checkpoint, filename)
