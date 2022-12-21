import torch
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor, Lambda

def view_images(images=[], rows=None, cols=None, title=""):
    num_images = len(images)
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    if (rows is None) and (cols is None):
        rows = int(num_images**0.5)
        cols = round(num_images / rows)
    fig = plt.figure(figsize=(cols*.5,rows*.5))
    fig.suptitle(title, fontsize=14, y=1.4)
    plt.axis('off')

    # Populate the figure
    i = 0
    for r in range(rows):
        for c in range(cols):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.set_axis_off()
            if i < len(images):
                plt.imshow(images[i][0])
                i += 1

    plt.show()

def view_batch0(loader):
    for batch in loader:
        view_images(images=batch[0])
        break

transform = Compose([
    ToTensor(),
    Lambda(lambda x: (x - 0.5) * 2)])