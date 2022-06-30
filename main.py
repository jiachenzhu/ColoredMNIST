import random
import argparse

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets

from helper import read_config
from scheduler import Scheduler

parser = argparse.ArgumentParser(description='Colored MNIST')
parser.add_argument('config_paths', nargs='+')
args = parser.parse_args()
config = read_config(args.config_paths)

class Encoder(nn.Module):
    def __init__(self, hidden_dim, expander_factor):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2 * 14 * 14, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.projector = nn.Sequential(  
            nn.Linear(hidden_dim, expander_factor * hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(expander_factor * hidden_dim, expander_factor * hidden_dim)
        )

        nn.init.xavier_uniform_(self.backbone[0].weight)
        nn.init.xavier_uniform_(self.backbone[2].weight)
        nn.init.xavier_uniform_(self.projector[0].weight)
        nn.init.xavier_uniform_(self.projector[2].weight)
        nn.init.zeros_(self.backbone[0].bias)
        nn.init.zeros_(self.backbone[2].bias)
        nn.init.zeros_(self.projector[0].bias)
        nn.init.zeros_(self.projector[2].bias)

    def forward(self, x):
        return self.projector(self.backbone(x))

class MLP(nn.Module):
        def __init__(self, hidden_dim):
            super().__init__()
            self.backbone = nn.Sequential(
                nn.Linear(2 * 14 * 14, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
            )
            
            self.classifier = nn.Linear(hidden_dim, 1)

            nn.init.xavier_uniform_(self.backbone[0].weight)
            nn.init.xavier_uniform_(self.backbone[2].weight)
            nn.init.xavier_uniform_(self.classifier.weight)
            nn.init.zeros_(self.backbone[0].bias)
            nn.init.zeros_(self.backbone[2].bias)
            nn.init.zeros_(self.classifier.bias)
    
        def forward(self, input):
            out = input.view(input.shape[0], 2 * 14 * 14)
            out = self.classifier(self.backbone(out))
            return out

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def vic_loss(x, y, sim_coeff, std_coeff, cov_coeff):
    repr_loss = F.mse_loss(x, y)

    batch_size, num_features = x.shape
    x = x - x.mean(dim=0)
    y = y - y.mean(dim=0)

    std_x = torch.sqrt(x.var(dim=0) + 0.0001)
    std_y = torch.sqrt(y.var(dim=0) + 0.0001)
    std_loss = torch.mean(F.relu(1 - std_x)) / 2 + torch.mean(F.relu(1 - std_y)) / 2

    cov_x = (x.T @ x) / (batch_size - 1)
    cov_y = (y.T @ y) / (batch_size - 1)
    cov_loss = off_diagonal(cov_x).pow_(2).sum().div(num_features) + off_diagonal(cov_y).pow_(2).sum().div(num_features)

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

final_train_accs = []
final_test_accs = []
for restart in range(config.n_restarts):
    print("Restart", restart)

    # Load MNIST, make train/val splits, and shuffle train set examples

    mnist = datasets.MNIST('datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    # Build environments

    def make_environment(images, labels, e):
        def torch_bernoulli(p, size):
            return (torch.rand(size) < p).float()
        def torch_xor(a, b):
            return (a-b).abs() # Assumes both inputs are either 0 or 1
        # 2x subsample for computational convenience
        images = images.reshape((-1, 28, 28))[:, ::2, ::2]
        # Assign a binary label based on the digit; flip label with probability 0.25
        labels = (labels < 5).float()
        labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
        # Assign a color based on the label; flip the color with probability e
        colors = torch_xor(labels, torch_bernoulli(e, len(labels)))
        # Apply the color to the image by zeroing out the other color channel
        images = torch.stack([images, images], dim=1)
        images[torch.tensor(range(len(images))), (1-colors).long(), :, :] *= 0
        return {
            'images': (images.float() / 255.).cuda(),
            'labels': labels[:, None].cuda()
        }

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.2),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.1),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]

    encoder = Encoder(config.hidden_dim, config.expander_factor).cuda()
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=config.encoder_start_lr)
    lr_scheduler = Scheduler(
        "lr",
        config.pretrain_epoch, 50000 // config.pretrain_batch_size,
        config.encoder_start_lr, 0.0,
        config.warmup_epoch,
    )
    
    step = 0
    for epoch in range(config.pretrain_epoch):
        id1 = list(range(25000))
        id2 = list(range(25000))
        random.shuffle(id1)
        random.shuffle(id2)
        current_start = 0
        for i in range(25000 // config.pretrain_batch_size):
            step += 1
            lr = lr_scheduler.get_value(step)
            for g in encoder_optimizer.param_groups:
                g['lr'] = lr
            
            image_env0 = envs[0]['images'][id1[current_start:current_start+config.pretrain_batch_size]]
            image_env1 = envs[0]['images'][id2[current_start:current_start+config.pretrain_batch_size]]
            current_start = current_start + config.pretrain_batch_size

            original_x = torch.cat([image_env0, image_env1], dim=0)
            x1 = F.dropout(original_x.view(-1, 2 * 14 * 14), p=config.dropout_rate_1)
            x2 = F.dropout(original_x.view(-1, 2 * 14 * 14), p=config.dropout_rate_2)

            p1 = encoder(x1)
            p2 = encoder(x2)

            loss = vic_loss(p1, p2, config.sim_coeff, config.std_coeff, config.cov_coeff)

            encoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()

            if step % 100 == 0:
                print(step, loss.item())
    
    mlp = MLP(config.hidden_dim).cuda()
    print(mlp.load_state_dict(encoder.state_dict(), strict=False))
    
    def mean_nll(logits, y):
        return nn.functional.binary_cross_entropy_with_logits(logits, y)

    def mean_accuracy(logits, y):
        preds = (logits > 0.).float()
        return ((preds - y).abs() < 1e-2).float().mean()

    def penalty(logits, y):
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = mean_nll(logits * scale, y)
        grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
        return torch.sum(grad**2)

    # Train loop

    def pretty_print(*values):
        col_width = 13
        def format_val(v):
            if not isinstance(v, str):
                v = np.array2string(v, precision=5, floatmode='fixed')
            return v.ljust(col_width)
        str_values = [format_val(v) for v in values]
        print("   ".join(str_values))

    optimizer = torch.optim.Adam(mlp.parameters(), lr=config.finetune_lr)

    pretty_print('step', 'train nll', 'train acc', 'train penalty', 'test acc')

    for step in range(config.finetune_steps):
        for env in envs:
            logits = mlp(env['images'])
            env['nll'] = mean_nll(logits, env['labels'])
            env['acc'] = mean_accuracy(logits, env['labels'])
            env['penalty'] = penalty(logits, env['labels'])

        train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
        train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
        train_penalty = torch.stack([envs[0]['penalty'], envs[1]['penalty']]).mean()

        weight_norm = torch.tensor(0.).cuda()
        for w in mlp.parameters():
            weight_norm += w.norm().pow(2)

        loss = train_nll.clone()
        loss += config.l2_regularizer_weight * weight_norm
        penalty_weight = (config.penalty_weight if step >= config.penalty_anneal_iters else 1.0)
        loss += penalty_weight * train_penalty
        if penalty_weight > 1.0:
            # Rescale the entire loss to keep gradients in a reasonable range
            loss /= penalty_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        test_acc = envs[2]['acc']
        if step % 100 == 0:
            pretty_print(
                np.int32(step),
                train_nll.detach().cpu().numpy(),
                train_acc.detach().cpu().numpy(),
                train_penalty.detach().cpu().numpy(),
                test_acc.detach().cpu().numpy()
            )

    final_train_accs.append(train_acc.detach().cpu().numpy())
    final_test_accs.append(test_acc.detach().cpu().numpy())
    print('Final train acc (mean/std across restarts so far):')
    print(np.mean(final_train_accs), np.std(final_train_accs))
    print('Final test acc (mean/std across restarts so far):')
    print(np.mean(final_test_accs), np.std(final_test_accs))

