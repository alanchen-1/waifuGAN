import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import time

# custom imports
from models.model_utils import generate_fake_labels, generate_real_labels
from models.models import init_models
from options.options import TrainOptions

def run(opt):
    torch.multiprocessing.freeze_support()
    # set random seed
    seed = 888
    print("Seed set as: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # -------- set parameters ----------#
    # Root directory for dataset
    dataroot = opt.dataroot
    results_path = opt.output_dir
    # Number of workers for the dataloader
    workers = opt.workers
    # Batch size during training
    # DCGAN paper uses 128
    batch_size = opt.batch_size
    # Spatial size of training images (All images resized to this size)
    image_size = opt.image_size
    # Epochs
    num_epochs = opt.epochs

    # default model parameters
    # Color images = 3 (for RGB)
    nc = 3
    # Size of z latent vector (generator input size)
    nz = 128
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Learning rate
    lrg = 0.0002
    lrd = 0.0002
    # Beta 1 hyperparam for adam
    beta1 = 0.5
    # Number of GPUs
    ngpu = 1
    # Specify device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Define custom dataset with ImageFolder
    dataset = dset.ImageFolder(root = dataroot,
                                transform = transforms.Compose([
                                    # specify transforms, including resizing
                                    # could be improved in future (resizing sus)
                                    transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = True, num_workers = workers)


    # Plot training images
    real_batch = next(iter(dataloader))
    """
    plt.figure(figsize=(8,8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.savefig("example_training.png")
    """

    netG, netD = init_models(ngpu=ngpu, device=device, nz=nz, ngf=ngf, ndf=ndf, nc=nc, verbose=True)
    # Use BCE loss
    loss = nn.BCELoss()
    fixed_noise = torch.randn(128, nz, 1, 1, device=device)

    # Real/fake labels - only used if not using soft labels
    real_label = 1.
    fake_label = 0.

    # set up optimizers
    optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    # training loop, nice formatting taken from Pytorch Tutorial
    print("Starting training...")
    start_time = time.time()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Start with discriminator
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            d_output = netD(real_cpu).view(-1)
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            soft_real_labels = generate_real_labels((batch_size,))
            if (opt.no_soft_labels):
                lossD_real = loss(d_output, label)
                lossD_real.backward()
            else:
                lossD_real = loss(d_output, soft_real_labels)
                lossD_real.backward()
            D_x = d_output.mean().item()

            ## Train with all-fake batch
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            fake = netG(noise) # plug in G
            label.fill_(fake_label)
            soft_fake_labels = generate_fake_labels((batch_size,))
            output = netD(fake.detach()).view(-1)
            if (opt.no_soft_labels):
                lossD_fake = loss(output, label)
                lossD_fake.backward()
            else:
                lossD_fake = loss(output, soft_fake_labels)
                lossD_fake.backward()

            D_G_z1 = output.mean().item()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            netG.zero_grad()
            label.fill_(real_label)
            soft_real_labels = generate_real_labels((batch_size, ))  
            # Put fake batch into D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            lossG = None
            if (opt.no_soft_labels):
                lossG = loss(output, label)
                lossG.backward()
            else:
                lossG = loss(output, soft_real_labels)
                lossG.backward()
            # Calculate gradients for G
            D_G_z2 = output.mean().item()
            optimizerG.step()

            # Output training stats, nice formatting by pytorch
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch, num_epochs, i, len(dataloader),
                        lossD.item(), lossG.item(), D_x, D_G_z1, D_G_z2))

            # Save loss function for plotting
            G_losses.append(lossG.item())
            D_losses.append(lossD.item())

            # Check how the generator is doing by saving G output
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    end_time = time.time()
    print("Time elapsed: ", end_time - start_time)

    torch.save(netG.state_dict(), os.path.join(results_path, 'generator.pth'))
    torch.save(netD.state_dict(), os.path.join(results_path, 'discriminator.pth'))

    if (not(opt.no_plots)):
        plt.figure(figsize=(10,5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses,label="G")
        plt.plot(D_losses,label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("loss graph.png")

        # Grab a batch of real images from the dataloader
        real_batch = next(iter(dataloader))

        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.subplot(1,2,1)
        plt.axis("off")
        plt.title("Real Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

        # Plot the fake images from the last epoch
        plt.subplot(1,2,2)
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.savefig("real_vs_fake.png")


        # Plot the real images
        plt.figure(figsize=(15,15))
        plt.axis("off")
        plt.title("Fake Images")
        plt.imshow(np.transpose(img_list[-1],(1,2,0)))
        plt.savefig("fake.png")

parser = TrainOptions()
opt = parser.parse()
if __name__ == '__main__':
    run(opt)
