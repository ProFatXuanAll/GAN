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
--batch_size 32 \
--ckpt_step 5000 \
--dataset mnist \
--dis_d_hid 256 \
--dis_k 4 \
--dis_lr 1e-3 \
--dis_n_layer 2 \
--dis_p_hid 0.5 \
--dis_p_in 0.2 \
--dis_ratio 7 \
--exp_name gan \
--gen_d_hid 256 \
--gen_d_in 128 \
--gen_lr 1e-3 \
--gen_n_layer 2 \
--gen_ratio 1 \
--log_step 2500 \
--seed 42 \
--total_step 100000
```

## Evaluate Model

## Infer Model
