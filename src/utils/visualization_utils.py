import os
import torch.nn.functional as F

os.environ["MPLCONFIGDIR"] = os.getcwd() + "./tmp/"
import matplotlib.pyplot as plt
import matplotlib
import torch
from PIL import Image
import numpy as np
from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw
from torchvision import transforms
# from moviepy.video.io.bindings import mplfig_to_npimage
import cv2

np.random.seed(2022)
COLORS_SPACE = np.random.randint(0, 255, size=(1000, 3))


def put_image_to_grid(list_imgs, adding_margin=True):
    num_col = len(list_imgs)
    b, c, h, w = list_imgs[0].shape
    device = list_imgs[0].device
    if adding_margin:
        num_all_col = num_col + 1
    else:
        num_all_col = num_col
    grid = torch.zeros((b * num_all_col, 3, h, w), device=device).to(list_imgs[0].dtype)
    idx_grid = torch.arange(0, grid.shape[0], num_all_col, device=device).to(
        torch.int64
    )
    for i in range(num_col):
        grid[idx_grid + i] = list_imgs[i].to(list_imgs[0].dtype)
    return grid, num_col + 1


def convert_cmap(tensor):
    b, h, w = tensor.shape
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    output = torch.zeros((b, 3, h, w), device=tensor.device)
    for i in range(b):
        cmap = matplotlib.cm.get_cmap("magma")
        tmp = cmap(ndarr[i])[..., :3]
        data = transforms.ToTensor()(np.array(tmp)).to(tensor.device)
        output[i] = data
    return output


def resize_tensor(tensor, size):
    return F.interpolate(tensor, size, mode="bilinear", align_corners=True)


