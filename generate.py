import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt

image_size = 64
# Number of channels in training images.
# Color images = 3 (for RGB)
nc = 3
# Size of z latent vector (generator input size)
nz = 128
# Size of feature maps in generator
ngf = 64
# Number of GPUs
ngpu = 1

# set seed if you want, otherwise just let it randomly generate some waifus
seed = 314
print("Seed set as: ", seed)
random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False),
            nn.BatchNorm2d(ngf*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, input):
        return self.main(input)

gen = Generator(ngpu).to(device)
gen.load_state_dict(torch.load('./results/generator.pth'))
img_list = []
with torch.no_grad():
    generated = gen(fixed_noise).detach().cpu()
img_list.append(vutils.make_grid(generated, padding=2, normalize=True))

plt.figure(figsize=(15,15))
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[0],(1,2,0)))
plt.savefig("generated_fakes.png")
plt.show()
