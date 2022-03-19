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
import matplotlib.animation as animation
import time


UPPER_FAKE = 0.3
def generate_fake_labels(tensor_size):
    return torch.rand(tensor_size) * UPPER_FAKE

UPPER_REAL = 1.2
LOWER_REAL = 0.7
def generate_real_labels(tensor_size):
    return torch.rand(tensor_size) * (UPPER_REAL - LOWER_REAL) + LOWER_REAL

def run():
    torch.multiprocessing.freeze_support()
    # set random seed haha
    seed = 888
    print("Seed set as: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # -------- set parameters ----------#
    # Root directory for dataset
    dataroot = "./cropped_images/"
    # Number of workers for the dataloader
    workers = 4
    # Batch size during training
    # DCGAN paper uses 128
    batch_size = 32
    # Spatial size of training images (All images resized to this size)
    image_size = 64
    # Number of channels in training images.
    # Color images = 3 (for RGB)
    nc = 3
    # Size of z latent vector (generator input size)
    nz = 128
    # Size of feature maps in generator
    ngf = 64
    # Size of feature maps in discriminator
    ndf = 64
    # Epochs
    num_epochs = 100
    # Learning rate
    lrg = 0.0002
    lrd = 0.0002
    # Beta 1 hyperparam for adam
    beta1 = 0.5
    # Number of GPUs
    ngpu = 1

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

    # Specify device
    device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

    # Plot training images
    real_batch = next(iter(dataloader))
    #plt.figure(figsize=(8,8))
    #plt.axis("off")
    #plt.title("Training Images")
    #plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    # plt.savefig("example_training.png")

    # Weights for the generator and discriminator should 
    # be randomly initialized from a normal distr with 0.02 stdev
    def weights_init(model):
        classname = model.__class__.__name__
        # checks for the appropriate layers
        #  and applies the criteria explained above
        if classname.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)

    # Create Generator model

    class Generator(nn.Module):
        def __init__(self, ngpu):
            super(Generator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is Z into a convolution
                nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias = False),
                nn.BatchNorm2d(ngf*8),
                nn.ReLU(True),
                # cur size: (ngf*8) x 4 x 4
                nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 4),
                nn.ReLU(True),
                # cur size: (ngf*4) x 8 x 8
                nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf * 2),
                nn.ReLU(True),
                # state size. (ngf*2) x 16 x 16
                nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.ReLU(True),
                # state size. (ngf) x 32 x 32
                nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
                nn.Tanh()
                # state size. (nc) x 64 x 64
            )
        
        def forward(self, input):
            return self.main(input)

    netG = Generator(ngpu).to(device)

    if(device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    netG.apply(weights_init)

    print(netG)

    class Discriminator(nn.Module):
        def __init__(self, ngpu):
            super(Discriminator, self).__init__()
            self.ngpu = ngpu
            self.main = nn.Sequential(
                # input is (nc) x 64 x 64
                nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),
                # state size. (ndf) x 32 x 32
                nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 2),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Dropout2d(p=0.2, inplace=False),
                # state size. (ndf*2) x 16 x 16
                nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 4),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Dropout2d(p=0.2, inplace=False),
                # state size. (ndf*4) x 8 x 8
                nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
                nn.BatchNorm2d(ndf * 8),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Dropout2d(p=0.2, inplace=False),
                # state size. (ndf*8) x 4 x 4
                nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
                nn.Sigmoid()
            )

        def forward(self, input):
            return self.main(input)


    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)

    # Initialize BCELoss function
    loss = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(128, nz, 1, 1, device=device)

    # Establish convention for real and fake labels during training
    real_label = 1.
    fake_label = 0.

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lrd, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lrg, betas=(beta1, 0.999))

    # Training Loop

    # Lists to keep track of progress
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    print("Starting training...")
    start_time = time.time()
    # For each epoch
    for epoch in range(num_epochs):
        # For each batch in the dataloader
        for i, data in enumerate(dataloader, 0):
            # Start with discriminator
            # Real batch
            netD.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            #label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)

            # Feed into discriminator
            d_output = netD(real_cpu).view(-1)

            # loss for real batch
            soft_real_labels = generate_real_labels((batch_size,))
            lossD_real = loss(d_output, soft_real_labels)
            # Calculate gradients for D
            lossD_real.backward()

            # calculate probability
            D_x = d_output.mean().item()

            ## Train with all-fake batch
            # Generate noise
            noise = torch.randn(batch_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = netG(noise)
            # label using fake 
            # label.fill_(fake_label)
            soft_fake_labels = generate_fake_labels((batch_size,))
            # Classify all fake batch with D
            output = netD(fake.detach()).view(-1)
            # Calc D's loss
            lossD_fake = loss(output, soft_fake_labels)
            # Calc gradients
            lossD_fake.backward()
            D_G_z1 = output.mean().item()
            # Compute error of D 
            lossD = lossD_real + lossD_fake
            optimizerD.step()


            netG.zero_grad()
            # label.fill_(real_label)
            # generate new soft real labels
            soft_real_labels = generate_real_labels((batch_size, ))  
            # Put fake batch into D
            output = netD(fake).view(-1)
            # Calculate G's loss based on this output
            lossG = loss(output, soft_real_labels)
            # Calculate gradients for G
            lossG.backward()
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

    torch.save(netG.state_dict(), 'generator.pth')
    torch.save(netD.state_dict(), 'discriminator.pth')

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

if __name__ == '__main__':
    run()
