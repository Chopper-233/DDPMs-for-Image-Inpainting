import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10

import unet
import ddpms
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\t".strip().upper() + ": " + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

batch_size = 16
num_epochs = 20
learning_rate = 0.001
retrain = False
store_path = "ddpm_cifar_10.pt"

dataset = CIFAR10("./datasets", download=True, train=True, transform=utils.transform)
loader = DataLoader(dataset, batch_size, shuffle=True)

# define model
n_steps = 1000
min_beta = 0.0001
max_beta = 0.02
ddpm = ddpms.DDPM_3_32(unet.UNet_3_32(n_steps), n_steps=n_steps, min_beta=min_beta, max_beta=max_beta, device=device, loader=loader, image_chw=(3,32,32))

# train
if retrain: ddpm.train_loop(num_epochs, optim=Adam(ddpm.parameters(), learning_rate), store_path=store_path)

# load best model and generate images
ddpm.load_state_dict(torch.load(store_path, map_location=device))
_ = ddpm.eval()

real = next(iter(loader))[0]
print(real.shape)
fake = ddpm.generate(n_samples=batch_size)
print(fake.shape)

fid = ddpm.calculate_frechet(real, fake)
print(fid)
utils.view_images(images=torch.cat([real.to(device), fake]), rows=8, cols=4)