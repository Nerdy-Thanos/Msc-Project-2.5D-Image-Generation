import cv2
import matplotlib.pyplot as plt
import torch
from torchvision.utils import save_image


device = torch.device("mps" if (torch.has_mps) else "cpu")

cpkt = torch.load("DCGan/ckpt/trained_gen.pt")
cpkt.to(device)
print(cpkt.eval())

fixed_noise = torch.randn(128, 256, 1, 1, device=device)
with torch.no_grad():
    fake = cpkt(fixed_noise)
fake.detach().cpu()
save_image(fake[0],"gen.jpg")

