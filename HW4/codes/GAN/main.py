import json

import GAN
from trainer import Trainer
from dataset import Dataset
from tensorboardX import SummaryWriter

import torch
import torch.optim as optim
import os
import argparse
import numpy as np
import random
import time
import torchvision.utils as tvu

def param_count(module: torch.nn.Module):
    p_count = sum(map(torch.numel, module.parameters()))
    print("[{}] Parameter Count {:.2e}".format(type(module).__name__, p_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--latent_dim', default=16, type=int)
    parser.add_argument('--generator_hidden_dim', default=16, type=int)
    parser.add_argument('--discriminator_hidden_dim', default=16, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--num_training_steps', default=5000, type=int)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--saving_steps', type=int, default=200)
    parser.add_argument('--learning_rate', default=0.0002, type=float)
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--data_dir', default='../data', type=str, help='The path of the data directory')
    parser.add_argument('--ckpt_dir', default='results', type=str, help='The path of the checkpoint directory')
    parser.add_argument('--interpolation_batch', default=0, type=int)
    parser.add_argument('--interpolation_K', default=10, type=int)
    parser.add_argument('--interpolation_range', nargs=2, default=[0, 1], type=float)
    parser.add_argument('--log_dir', default='./runs', type=str)
    parser.add_argument('--seed', default=2022, type=int)
    parser.add_argument('--mlp', default=False, action='store_true')
    parser.add_argument('--mlp_g_arch', nargs='+', type=int)
    parser.add_argument('--mlp_d_arch', nargs='+', type=int)
    parser.add_argument('--sampling', default=0, type=int)
    parser.add_argument('--tag', default='', type=str)
    parser.add_argument('--print_param_only', default=False, action='store_true')
    args = parser.parse_args()

    # Global Random Seed
    if args.seed < 0:
        args.seed = int(time.time())
    print("Random Seed", args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    from pytorch_fid import fid_score

    if args.mlp:
        args.latent_dim = args.mlp_g_arch[0]
        config = 'MLP-z-L{}_G{}_D{}_B{}_S{}_SD{}'.format(
            args.latent_dim,
            str(args.mlp_g_arch),
            str(args.mlp_d_arch),
            args.batch_size,
            args.num_training_steps,
            args.seed,
        ).replace(" ", "")
    else:
        config = 'z-L{}_G{}_D{}_B{}_S{}_SD{}'.format(
            args.latent_dim,
            args.generator_hidden_dim,
            args.discriminator_hidden_dim,
            args.batch_size,
            args.num_training_steps,
            args.seed,
        )
    config = args.tag + config

    args.ckpt_dir = os.path.join(args.ckpt_dir, config)
    args.log_dir = os.path.join(args.log_dir, config)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

    dataset = Dataset(args.batch_size, args.data_dir, args.seed)
    if args.mlp:
        netG = GAN.get_mlp_generator(args.mlp_g_arch, device)
        netD = GAN.get_mlp_discriminator(args.mlp_d_arch, device)
    else:
        netG = GAN.get_generator(1, args.latent_dim, args.generator_hidden_dim, device)
        netD = GAN.get_discriminator(1, args.discriminator_hidden_dim, device)
    param_count(netG)
    param_count(netD)
    # print(netG)
    # print(netD)
    if args.print_param_only:
        exit(0)

    tb_writer = SummaryWriter(args.log_dir)

    if args.do_train:
        optimG = optim.Adam(netG.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        optimD = optim.Adam(netD.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
        trainer = Trainer(device, netG, netD, optimG, optimD, dataset, args.ckpt_dir, tb_writer)
        trainer.train(args.num_training_steps, args.logging_steps, args.saving_steps)

    restore_ckpt_path = os.path.join(args.ckpt_dir, str(max(int(step) for step in os.listdir(args.ckpt_dir) if step.isnumeric())))
    netG.restore(restore_ckpt_path)
    netG.eval()

    if args.interpolation_batch > 0:
        print("Interpolating ...")
        with torch.no_grad():
            K, rg, B = args.interpolation_K + 1, args.interpolation_range, args.interpolation_batch
            l_shape = B, netG.latent_dim, 1
            points = torch.randn(2, *l_shape, device=device)
            weights = torch.linspace(*rg, K, device=device)
            # print(weights)
            z_batch = torch.lerp(
                *points,
                weights
            ).permute((0, 2, 1))[..., None, None]  # (batch, K, latent_dim, 1, 1)

            images = netG(torch.cat(tuple(z_batch), dim=0))
            images = (tvu.make_grid(images, nrow=z_batch.size(1), pad_value=0) + 1) / 2
            tvu.save_image(
                images,
                os.path.join(
                    args.ckpt_dir,
                    f"interpolation{B}_{K}_{rg}.png".replace(" ", "")
                )
            )
            exit(0)
    if args.sampling > 0:
        print("Sampling ...")
        with torch.no_grad():
            z_batch = torch.randn(args.sampling, netG.latent_dim, 1, 1, device=device)
            images = netG(z_batch)
            images = (tvu.make_grid(images, nrow=int(np.sqrt(args.sampling)), pad_value=0) + 1) / 2
            tvu.save_image(images, os.path.join(args.ckpt_dir, f"sampling{args.sampling}.png"))
            exit(0)

    num_samples = 3000
    real_imgs = None
    real_dl = iter(dataset.training_loader)
    while real_imgs is None or real_imgs.size(0) < num_samples:
        imgs = next(real_dl)
        if real_imgs is None:
            real_imgs = imgs[0]
        else:
            real_imgs = torch.cat((real_imgs, imgs[0]), 0)
    real_imgs = real_imgs[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5

    with torch.no_grad():
        samples = None
        while samples is None or samples.size(0) < num_samples:
            imgs = netG.forward(torch.randn(args.batch_size, netG.latent_dim, 1, 1, device=device))
            if samples is None:
                samples = imgs
            else:
                samples = torch.cat((samples, imgs), 0)
    samples = samples[:num_samples].expand(-1, 3, -1, -1) * 0.5 + 0.5
    samples = samples.cpu()

    fid = fid_score.calculate_fid_given_images(real_imgs, samples, args.batch_size, device)
    tb_writer.add_scalar('fid', fid)
    print("FID score: {:.2f}\n".format(fid), flush=True)
    with open("log.jsonl", "a") as f:
        json.dump({
            "name": config,
            "fid": fid
        }, f)
        f.write("\n")
    tb_writer.flush()