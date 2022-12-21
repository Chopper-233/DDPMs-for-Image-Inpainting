import torch
import torch.nn as nn
import utils_cifar as utils
from tqdm.auto import tqdm
import torchvision as tv
import numpy as np
from scipy import linalg
import torchvision.transforms.functional as fn

class DDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10**-4, max_beta=0.02, device=None, image_chw=(3, 32, 32), loader=None):
        super(DDPM_3_32, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.loader = loader
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    # add noise to image
    def forward(self, x0, t, eta=None):
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None: eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    # estimate noise added to x at t
    def backward(self, x, t):
        return self.network(x, t)

    def show_forward(self, stages=[0.25, 0.5, 0.75, 1]):
        for batch in self.loader:
            images = batch[0]
            utils.view_images(images=images[:5], title="Noise: 0%", rows=1, cols=5)
            for percent in stages:
                utils.view_images(images=self.forward(images.to(self.device),[int(percent * self.n_steps) - 1 for _ in range(len(images))])[:5],
                                  title=f"Noise: {int(percent * 100)}%",
                                  rows=1, cols=5)
            break

    def generate(self, n_samples=16, c=3, h=32, w=32):
        with torch.no_grad():
            # start from random noise
            x = torch.randn(n_samples, c, h, w).to(self.device)

            for t in list(range(self.n_steps))[::-1]:
                # estimate noise to be removed
                time_tensor = (torch.ones(n_samples, 1) * t).to(self.device).long()
                eta_theta = self.backward(x, time_tensor)

                alpha_t = self.alphas[t]
                alpha_t_bar = self.alpha_bars[t]

                # partially denoise
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    z = torch.randn(n_samples, c, h, w).to(self.device)

                    # sigma_t squared = beta_t
                    beta_t = self.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z
        return x

    def train_loop(self, n_epochs, optim, store_path):
        mse = nn.MSELoss()
        best_loss = float("inf")

        for epoch in tqdm(range(n_epochs), desc=f"Training progress", colour="#00ff00"):
            epoch_loss = 0.0
            for batch in tqdm(self.loader, leave=False, desc=f"Epoch {epoch + 1}/{n_epochs}", colour="#005500"):
                # load data
                x0 = batch[0].to(self.device)
                n = len(x0)

                # random init for noise, timestep
                eta = torch.randn_like(x0).to(self.device)
                t = torch.randint(0, self.n_steps, (n,)).to(self.device)

                # forward process - compute noisy image
                noisy_imgs = self.forward(x0, t, eta)

                # backward process - estimate noise
                eta_theta = self.backward(noisy_imgs, t.reshape(n, -1))

                # optimise mse between noise and predicted noise
                loss = mse(eta_theta, eta)
                optim.zero_grad()
                loss.backward()
                optim.step()

                epoch_loss += loss.item() * len(x0) / len(self.loader.dataset)

            log_string = f"Epoch {epoch + 1}, Loss: {epoch_loss:.3f}"

            # Storing the model
            if best_loss > epoch_loss:
                best_loss = epoch_loss
                torch.save(self.state_dict(), store_path)
                log_string += " --> Stored - Best so far"

            print(log_string)

    # FROM pytorch-fid
    def calculate_activation_statistics(self, images, dims=2048):
        images = fn.resize(images, (299,299))
        self.inception = tv.models.inception_v3(weights='DEFAULT', transform_input=False).to(self.device)
        self.inception.eval()
        act = np.empty((len(images), dims))
        
        batch = images.to(self.device)
        pred = self.inception(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        # if pred.size(2) != 1 or pred.size(3) != 1:
        #     pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        act = pred.cpu().data.numpy().reshape(pred.size(0), -1)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        return mu, sigma

    # FROM pytorch-fid
    def calculate_frechet_distance(self, mu_1, sigma1, mu_2, sigma2, eps=1e-6):
        mu_1 = np.atleast_1d(mu_1)
        mu_2 = np.atleast_1d(mu_2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu_1.shape == mu_2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu_1 - mu_2
        
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1) +
                np.trace(sigma2) - 2 * tr_covmean)

    def calculate_frechet(self, images_real, images_fake):
        mu_1, std_1 = self.calculate_activation_statistics(images_real)
        mu_2, std_2 = self.calculate_activation_statistics(images_fake)

        print("real: ", mu_1, ", ", std_1)
        print("fake: ", mu_2, ", ", std_2)
        
        fid_value = self.calculate_frechet_distance(mu_1, std_1, mu_2, std_2)
        return fid_value
