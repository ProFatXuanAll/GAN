# GAN

Generative Adversarial Network implementation based on PyTorch.

## Install

```sh

## Install

```sh
# Clone the project.
git clone https://github.com/ProFatXuanAll/GAN.git

# Install dependencies.
pipenv install
```

## Train Model

```sh
python run_train.py \
--batch_size 100 \
--ckpt_step 5000 \
--dataset mnist \
--dis_d_hid 240 \
--dis_k 4 \
--dis_lr 1e-4 \
--dis_momentum 0.5 \
--dis_n_layer 3 \
--dis_p_hid 0.5 \
--dis_p_in 0.2 \
--dis_ratio 1 \
--exp_name gan \
--gen_d_hid 1200 \
--gen_d_in 100 \
--gen_lr 1e-4 \
--gen_momentum 0.5 \
--gen_n_layer 3 \
--gen_ratio 1 \
--log_step 100 \
--seed 42 \
--total_step 100000
```

## Evaluate Model

## Infer Model
