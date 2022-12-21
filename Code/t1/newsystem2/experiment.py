import functools
from typing import List

import torch
import torch.utils.data
import torchvision
from torchvision.utils import save_image as imsave
import torchvision.transforms as tvt
import os
from tqdm.auto import tqdm

from ddpm import DDPM
from unet import UNet

root_path = 'data'
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Experiment:
    def __init__(self, *, exp_name="exp", dataset_choice='CIFAR', image_channels=3, epochs=100, device=dev, image_size=32, n_channels=128, channel_multipiers=[1, 2, 2, 2], is_attention=[False, True, True, True], n_steps=1000, batch_size=128, n_samples=128, learning_rate=2e-5, optimiser=torch.optim.Adam):
        self.exp_name = exp_name
        self.dataset_choice = dataset_choice
        self.image_channels = image_channels
        self.epochs = epochs
        self.device = device
        self.image_size = image_size
        self.n_channels = n_channels
        self.channel_multipliers = channel_multipiers
        self.is_attention = is_attention
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.learning_rate = learning_rate
        self.optimiser = optimiser

        self.create_exppath()
        self.create_dataset()
        self.create_unet()
        self.create_ddpm()
        self.create_dataloader()
        self.create_optimiser()

    def create_exppath(self):
        os.makedirs(root_path, exist_ok=True)
        self.exp_path = os.path.join(root_path, self.exp_name)
        os.makedirs(self.exp_path, exist_ok=True)
        self.sample_path = os.path.join(self.exp_path, 'samples')
        os.makedirs(self.sample_path, exist_ok=True)
        self.model_path = os.path.join(self.exp_path, 'models')
        os.makedirs(self.model_path, exist_ok=True)
        self.data_path = os.path.join(root_path, 'datasets')
        os.makedirs(self.data_path, exist_ok=True)

    def save_image(self, image, count):
        imsave(image, os.path.join(
            self.sample_path, (str(count) + '.png')))

    def save_model(self, count):
        torch.save(self.model.state_dict(), os.path.join(
            self.model_path, (str(count) + '.pt')))

    def create_dataset(self):
        if self.dataset_choice == "CIFAR":
            self.dataset = CIFARDatset(self.data_path, self.image_size)
            self.image_channels = 3
            self.epochs = 100
        elif self.dataset_choice == "MNIST":
            self.dataset = MNISTDataset(self.data_path, self.image_size)
            self.image_channels = 1
            self.epochs = 5
        elif self.dataset_choice == "MNIST_FASHION":
            pass
        else:
            raise Exception("Invalid dataset choice")

    def create_unet(self):
        self.model = UNet(
            image_channels=self.image_channels,
            n_channels=self.n_channels,
            ch_multipliers=self.channel_multipliers,
            is_attn=self.is_attention,
        ).to(self.device)

    def create_ddpm(self):
        self.ddpm = DDPM(
            model=self.model,
            T=self.n_steps,
            device=self.device,
        )

    def create_dataloader(self):
        self.data_loader = torch.utils.data.DataLoader(
            self.dataset, self.batch_size, shuffle=True, pin_memory=True)

    def create_optimiser(self):
        self.optimiser = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def sample(self):
        shape = (self.n_samples, self.image_channels, self.image_size, self.image_size)
        samples = self.ddpm.p_sample_loop(shape)
        for i, sample in enumerate(samples):
            if i % 100 == 0:
                self.save_image(sample, i)
        final = samples[len(samples) - 1]
        self.save_image(final, self.n_steps)
        return final

    def prog_sample(self):
        shape = (self.n_samples, self.image_channels, self.image_size, self.image_size)
        samps, prog_samps = self.ddpm.p_sample_loop_prog(shape)
        return samps, prog_samps

    def train(self):
        pbar = tqdm(self.data_loader)
        for x0 in pbar:
            x0 = x0.to(self.device)
            self.optimiser.zero_grad()
            B, H, W, C = x0.shape
            # TODO: randflip
            t = torch.randint(0, self.ddpm.T, (B,), device=x0.device, dtype=torch.long)
            loss = self.ddpm.loss(x0, t)
            loss.backward()
            self.optimiser.step()
            pbar.set_description(("loss: " + str(round(loss.item(), 4))))

    def fid(self, fake):
        real = next(iter(self.data_loader))
        assert fake.shape == real.shape
        fid = self.ddpm.calculate_frechet(real, fake)
        print(round(fid, 4))
        return fid

    def bpd(self):
        x0 = next(iter(self.data_loader)).to(self.device)
        out = self.ddpm.bpd_loop(x0)
        total_bpd = out["total_bpd"]
        bpd = torch.mean(total_bpd).item()
        print(round(bpd, 4))
        return bpd

    def run(self):
        pbar = tqdm(range(self.epochs))
        for epoch in pbar:
            print("TRAIN")
            self.train()
            print("SAMPLE")
            fake = self.sample()            
            print("FID")
            self.fid(fake)
            print("BPD")
            self.bpd()
            print("SAVING")
            self.save_model(epoch)


class CIFARDatset(torchvision.datasets.CIFAR10):
    def __init__(self, data_path, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(data_path),
                         train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, data_path, image_size):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(data_path),
                         train=True, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]


def main():
    e = Experiment(exp_name="mnist", dataset_choice="MNIST")
    e.run()


if __name__ == '__main__':
    main()
