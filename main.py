import torch
from dataset import RESIDE_Dataset, Eval_Dataset
from dataset import tensorShow, UnNormalize
import sys
import math
from models.generator import Generator
from models.discriminator import Discriminator
from PIL import Image
from utility import save_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.backends import cudnn
import torchvision
import torchvision.datasets as datasets
import torchvision.utils as vutils
from torchvision.utils import save_image
from matplotlib import pyplot as plt
from option import opt
from metrics import psnr, ssim
import numpy as np
import argparse
from torchvision.utils import make_grid
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter


def lr_schedule_cosdecay(t, T=opt.steps, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr


def train_fn(
    disc_C,
    disc_H,
    gen_H,
    gen_C,
    train_loader,
    test_loader,
    opt_disc,
    opt_gen,
    l1,
    mse,
    just_eval=False,
):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []
    g_scaler = GradScaler()
    d_scaler = GradScaler()
    writer = SummaryWriter(f"/runs")
    if opt.load_model:
        print("Loading checkpoint ---")
        checkpoint = torch.load(opt.checkpoint_model, map_location=opt.device)
        gen_C.load_state_dict(checkpoint["gen_C"])
        gen_H.load_state_dict(checkpoint["gen_H"])
        disc_C.load_state_dict(checkpoint["disc_C"])
        disc_H.load_state_dict(checkpoint["disc_H"])
        opt_gen.load_state_dict(checkpoint["opt_gen"])
        opt_disc.load_state_dict(checkpoint["opt_disc"])
        start_step = checkpoint["start_step"]
        max_ssim = checkpoint["max_ssim"]
        max_psnr = checkpoint["max_psnr"]
        psnrs = checkpoint["psnrs"]
        ssims = checkpoint["ssims"]

    if just_eval:
        print(f"Just Eval ---")
        eval_loader = DataLoader(
            dataset=Eval_Dataset("inputs", train=False, size="whole img"),
            batch_size=1,
            shuffle=False,
            pin_memory=True,
        )
        Eval(gen_C, eval_loader)
        return

    if opt.load_model:

        print(f"Starting training at {start_step} ---")

    else:
        print("Train from scratch ---")

    C_reals = 0
    C_fakes = 0
    train_iterator = iter(train_loader)
    loop = tqdm(range(start_step + 1, opt.steps), leave=True)
    for idx in loop:

        lr = lr_schedule_cosdecay(idx)

        for param_group in opt_gen.param_groups:
            param_group["lr"] = lr

        haze, clean = next(train_iterator)
        haze = haze.to(opt.device)
        clean = clean.to(opt.device)

        fake_clean = gen_C(haze)
        D_C_real = disc_C(clean)
        D_C_fake = disc_C(fake_clean.detach())
        C_reals += D_C_real.mean().item()
        C_fakes += D_C_fake.mean().item()
        D_C_real_loss = mse(D_C_real, torch.ones_like(D_C_real))
        D_C_fake_loss = mse(D_C_fake, torch.zeros_like(D_C_fake))
        D_C_loss = D_C_real_loss + D_C_fake_loss

        fake_haze = gen_H(clean)
        D_H_real = disc_H(haze)
        D_H_fake = disc_H(fake_haze.detach())
        D_H_real_loss = mse(D_H_real, torch.ones_like(D_H_real))
        D_H_fake_loss = mse(D_H_fake, torch.zeros_like(D_H_fake))
        D_H_loss = D_H_real_loss + D_H_fake_loss

        D_loss = (D_H_loss + D_C_loss) / 2

        opt_disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # adversarial loss for both generators
        D_C_fake = disc_C(fake_clean)
        D_H_fake = disc_H(fake_haze)
        loss_G_C = mse(D_C_fake, torch.ones_like(D_C_fake))
        loss_G_H = mse(D_H_fake, torch.ones_like(D_H_fake))

        # cycle loss
        cycle_haze = gen_H(fake_clean)
        cycle_clean = gen_C(fake_haze)
        cycle_haze_loss = l1(haze, cycle_haze)
        cycle_clean_loss = l1(clean, cycle_clean)

        # identity loss (remove these for efficiency if you set lambda_identity=0)
        identity_haze = gen_H(haze)
        identity_clean = gen_C(clean)
        identity_haze_loss = l1(haze, identity_haze)
        identity_clean_loss = l1(clean, identity_clean)

        # add all togethor
        G_loss = (
            loss_G_C
            + loss_G_H
            + cycle_haze_loss * opt.lambda_cycle
            + cycle_clean_loss * opt.lambda_cycle
            + identity_clean_loss * opt.lambda_identity
            + identity_haze_loss * opt.lambda_identity
        )

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()
        losses.append(G_loss.item())
        writer.add_scalar("data/loss_G", G_loss, idx)
        writer.add_scalar("data/loss_Gen_C", loss_G_C, idx)
        writer.add_scalar("data/loss_cycle_clean", cycle_clean_loss, idx)
        writer.add_scalar("data/loss_clean_identity", identity_clean_loss, idx)
        writer.add_scalar("data/loss_Disc", D_loss, idx)

        if idx % opt.eval_step == 0:

            unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))

            img_grid = vutils.make_grid(
                [
                    torch.squeeze(unorm(haze.detach()).cpu()[0]),
                    torch.squeeze(unorm(fake_clean.detach()).cpu()[0]),
                    torch.squeeze(unorm(clean.detach()).cpu()[0]),
                    torch.squeeze(unorm(fake_haze.detach()).cpu()[0]),
                ]
            )
            save_image(img_grid, f"saved_images/img_{idx}.png")

        if idx % (10 * opt.eval_step) == 0 and idx != 0:
            with torch.no_grad():
                ssim_eval, psnr_eval = test(gen_C, test_loader, max_psnr, max_ssim, idx)
            print(f"\n iter :{idx} |ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}")

            writer.add_scalar("data/ssim", ssim_eval, idx)
            writer.add_scalar("data/psnr", psnr_eval, idx)
            writer.add_scalars(
                "group", {"ssim": ssim_eval, "psnr": psnr_eval, "loss": G_loss}, idx
            )

            ssims.append(ssim_eval)
            psnrs.append(psnr_eval)

            if True:
                print("sim eval")
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                save_checkpoint(
                    gen_C,
                    gen_H,
                    disc_C,
                    disc_H,
                    opt_gen,
                    opt_disc,
                    idx,
                    max_ssim,
                    max_psnr,
                    ssims,
                    psnrs,
                    filename=opt.checkpoint_model,
                )

                print(
                    f"\n model saved at :{idx} | max_psnr:{max_psnr:.4f}|max_ssim:{max_ssim:.4f}"
                )

        loop.set_postfix(
            iter=idx,
            G_loss=G_loss.item(),
            D_loss=D_loss.item(),
            cycle_H_loss=cycle_haze_loss.item(),
            cycle_C_loss=cycle_clean_loss.item(),
            G_C_loss=loss_G_C.item(),
            G_H_loss=loss_G_H.item(),
            I_H_loss=identity_haze_loss.item(),
            I_loss=identity_clean_loss.item(),
            max_ssim=max_ssim,
            max_psnr=max_psnr,
        )

    if opt.save_model:
        save_checkpoint(
            gen_C,
            gen_H,
            disc_C,
            disc_H,
            opt_gen,
            opt_disc,
            idx,
            max_ssim,
            max_psnr,
            ssims,
            psnrs,
            filename=opt.checkpoint_model,
        )


def test(gen_C, test_loader, max_psnr, max_ssim, idx):
    gen_C.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []
    s = True
    val_loop = tqdm(test_loader, leave=True)
    for i, (inputs, targets) in enumerate(val_loop):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred = gen_C(inputs)
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ts = vutils.make_grid(
            [
                torch.squeeze(unorm(inputs.detach()).cpu()[0]),
                torch.squeeze(unorm(targets.detach()).cpu()[0]),
                torch.squeeze(unorm(pred.detach()).cpu()[0]),
            ]
        )

        vutils.save_image(ts, f"test_images/{i}_testimg.png")
        ssim1 = ssim(unorm(pred.detach()), unorm(targets.detach())).item()
        psnr1 = psnr(unorm(pred.detach()), unorm(targets.detach()))
        ssims.append(ssim1)
        psnrs.append(psnr1)
        if (psnr1 > max_psnr or ssim1 > max_ssim) and s:
            vutils.save_image(
                ts, f"test_images/best_picks/{idx}_ps_{psnr1:.4}_{ssim1:.4}.png"
            )
            s = False
        val_loop.set_postfix(ssim=ssim1, psnr=psnr1)
    return np.mean(ssims), np.mean(psnrs)


def Eval(gen_C, eval_loader):
    gen_C.eval()
    torch.cuda.empty_cache()
    s = True
    val_loop = tqdm(eval_loader, leave=True)
    for i, (inputs) in enumerate(val_loop):
        inputs = inputs.to(opt.device)
        pred = gen_C(inputs)
        unorm = UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        vutils.save_image(
            torch.squeeze(unorm(pred.detach()).cpu()[0]), f"outputs/{i}_testimg.png"
        )


def main():
    disc_C = Discriminator(in_channels=3).to(opt.device)
    disc_H = Discriminator(in_channels=3).to(opt.device)
    gen_H = Generator(img_channels=3).to(opt.device)
    gen_C = Generator(img_channels=3).to(opt.device)
    if opt.device == "cuda":
        print("Running in CUDA")
        disc_C = torch.nn.DataParallel(disc_C)
        disc_H = torch.nn.DataParallel(disc_H)
        gen_H = torch.nn.DataParallel(gen_H)
        gen_C = torch.nn.DataParallel(gen_C)
        cudnn.benchmark = True

    opt_disc = optim.Adam(
        list(disc_C.parameters()) + list(disc_H.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_H.parameters()) + list(gen_C.parameters()),
        lr=opt.lr,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    ITS_train_loader = DataLoader(
        dataset=RESIDE_Dataset("data", train=True, size=opt.crop_size),
        batch_size=opt.bs,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )
    ITS_test_loader = DataLoader(
        dataset=RESIDE_Dataset("data/SOTS/indoor", train=False, size="whole img"),
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    train_fn(
        disc_C,
        disc_H,
        gen_H,
        gen_C,
        ITS_train_loader,
        ITS_test_loader,
        opt_disc,
        opt_gen,
        L1,
        mse,
        just_eval=args.eval,
    )


if __name__ == "__main__":
    main()
