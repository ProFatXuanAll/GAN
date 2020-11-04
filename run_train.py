import argparse
import json
import os

import torch
import torch.optim
import torch.utils.data
import torch.utils.tensorboard
import torchvision

from tqdm import tqdm

import gan

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--ckpt_step',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dataset',
        choices=gan.dataset.dataset_map.keys(),
        help='',
        required=True,
        type=str
    )
    parser.add_argument(
        '--dis_d_hid',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dis_k',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dis_lr',
        help='',
        required=True,
        type=float
    )
    parser.add_argument(
        '--dis_n_layer',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--dis_p_hid',
        help='',
        required=True,
        type=float
    )
    parser.add_argument(
        '--dis_p_in',
        help='',
        required=True,
        type=float
    )
    parser.add_argument(
        '--dis_ratio',
        help='Discriminator training ratio.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--exp_name',
        help='Name of the experiment.',
        required=True,
        type=str
    )
    parser.add_argument(
        '--is_normal',
        action='store_true',
        help=''
    )
    parser.add_argument(
        '--gen_d_hid',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--gen_d_in',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--gen_lr',
        help='',
        required=True,
        type=float
    )
    parser.add_argument(
        '--gen_n_layer',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--gen_ratio',
        help='Generator training ratio.',
        required=True,
        type=int
    )
    parser.add_argument(
        '--log_step',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--seed',
        help='',
        required=True,
        type=int
    )
    parser.add_argument(
        '--total_step',
        help='',
        required=True,
        type=int
    )

    return parser.parse_args()

def main():
    # Parse arguments.
    args = parse_args()

    # Set seed.
    gan.util.set_seed(seed=args.seed)

    # Get model running device.
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda')

    # Get experiment path and log path.
    exp_path = os.path.join(gan.path.EXP_PATH, args.exp_name)
    log_path = os.path.join(gan.path.LOG_PATH, args.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)

    # Save configuration.
    with open(os.path.join(exp_path, 'cfg.json'), 'w') as output_file:
        json.dump(args.__dict__, output_file, ensure_ascii=False)

    # Get sampler.
    dataset_cstr = gan.dataset.dataset_map[args.dataset]
    dataset = dataset_cstr(
        batch_size=args.batch_size,
        is_train=True,
        shuffle=True
    )
    sampler = iter(dataset)

    dis_model = gan.model.Discriminator(
        d_hid=args.dis_d_hid,
        d_in=dataset_cstr.get_d_in(),
        d_out=1,
        k=args.dis_k,
        n_layer=args.dis_n_layer,
        p_hid=args.dis_p_hid,
        p_in=args.dis_p_in
    ).to(device)

    gen_model = gan.model.Generator(
        d_hid=args.gen_d_hid,
        d_in=args.gen_d_in,
        d_out=dataset_cstr.get_d_in(),
        n_layer=args.gen_n_layer
    ).to(device)

    dis_opt = torch.optim.Adam(
        params=dis_model.parameters(),
        lr=args.dis_lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01
    )

    gen_opt = torch.optim.Adam(
        params=gen_model.parameters(),
        lr=args.gen_lr,
        betas=(0.9, 0.99),
        eps=1e-8,
        weight_decay=0.01
    )

    # Log performance
    writer = torch.utils.tensorboard.SummaryWriter(log_path)
    total_dis_loss = 0.0
    total_gen_loss = 0.0

    total_step = tqdm(
        range(1, args.total_step + 1),
        desc=f'dis loss: {total_dis_loss:.6f}, gen loss: {total_gen_loss:.6f}'
    )
    for step in total_step:
        # Discriminator training loop.
        for _ in range(args.dis_ratio):
            x_real, _ = next(sampler)
            # (B, I)
            z = gan.util.get_noise(
                (x_real.size(0), args.gen_d_in),
                is_normal=args.is_normal
            ).to(device)

            x_fake = gen_model(z)
            x_real = x_real.reshape(-1, dataset_cstr.get_d_in()).to(device)

            y_fake = dis_model(x_fake)
            y_real = dis_model(x_real)

            # Discriminator objective.
            dis_loss = (-y_real.log() - (1.0 - y_fake).log()).mean(dim=0)
            dis_loss.backward()

            # Log average loss.
            total_dis_loss += dis_loss.item() / args.log_step

            # Only update discriminator.
            dis_opt.step()
            dis_opt.zero_grad()
            gen_opt.zero_grad()

        # Generator training loop.
        for _ in range(args.gen_ratio):
            # (B, I)
            z = gan.util.get_noise(
                (args.batch_size, args.gen_d_in),
                is_normal=args.is_normal
            ).to(device)

            x_fake = gen_model(z)
            y_fake = dis_model(x_fake)

            # Generator objective.
            gen_loss = (-y_fake.log()).mean(dim=0)
            gen_loss.backward()

            # Log average loss.
            total_gen_loss += gen_loss.item() / args.log_step

            # Only update generator.
            gen_opt.step()
            dis_opt.zero_grad()
            gen_opt.zero_grad()

        if step % args.ckpt_step == 0:
            pass

        if step % args.log_step == 0:
            total_step.set_description(
                desc=f'dis loss: {total_dis_loss:.6f}, gen loss: {total_gen_loss:.6f}'
            )
            writer.add_scalar('dis/loss', total_dis_loss, step)
            writer.add_scalar('gen/loss', total_gen_loss, step)
            total_dis_loss = 0.0
            total_gen_loss = 0.0

    # Close logger.
    writer.close()

if __name__ == '__main__':
    main()
