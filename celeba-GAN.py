import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
import torch
import time
import os

# --------------- Hyperparameters 
n_epochs = 1
learning_rate = 0.001
batch_size = 64
nz = 100 # size of generator input
ngf = 32#64 # size of feature maps in generator
ndf = 32#64 # size of feature maps in discriminator

# --------------- Data Loading and Preparation
current_dir = os.path.dirname(os.path.realpath(__file__))
data_dir = str(current_dir) + '/celeba'

# set the device to cuda if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    print('No GPU detected. Using CPU instead.')
print('Using device:', device)

dataset = torchvision.datasets.ImageFolder(root=data_dir, transform=transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
]))

dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Plot some training images
# real_batch = next(iter(dataloader))
# plt.figure(figsize=(8,8))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(torchvision.utils.make_grid(real_batch[0].to(device)[:64], 
#                                                     padding=2, normalize=True).cpu(),(1,2,0)))
# plt.show()


# --------------- Generator Model
class DCGAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf*8),
            nn.LeakyReLU(True),
            # size = (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(True),
            # size = (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(True),
            # size = (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(True),
            # size = (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh())
        
        self.discriminator = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # size = (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, inplace=True),
            # size = (ndf*2) x 16 x 16
            nn.Conv2d(ndf*2, ndf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, inplace=True),
            # size = (ndf*4) x 8 x 8
            nn.Conv2d(ndf*4, ndf*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf*8),
            nn.LeakyReLU(0.2, inplace=True),
            # size = (ndf*8) x 4 x 4
            nn.Conv2d(ndf*8, 1, 4, 1, 0, bias=False),
            nn.Flatten())
        
    def forward_generator(self, z):
        return self.generator(z)
            
    def forward_discriminator(self, image):
        return self.discriminator(image)
    
# --------------- Training
gan = DCGAN().to(device)
gen_opt = torch.optim.Adam(gan.generator.parameters(), lr=learning_rate) # use Adam optimizer for generator
dis_opt = torch.optim.Adam(gan.discriminator.parameters(), lr=learning_rate) # use Adam optimizer for discriminator
loss_fn = nn.CrossEntropyLoss() # use cross-entropy loss

# use one-cycle learning rate scheduler
# scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
#                                                 steps_per_epoch=len(dataset), epochs=n_epochs)

# create labels
real_label = 1
fake_label = 0

# logs
discriminator_losses = []
generator_losses = []
discriminator_fake_accuracies = []
discriminator_real_accuracies = []
images_generated = []

noise = torch.randn(64, nz, 1, 1, device=device) # 64 random inputs for generator
start_time = time.time()
print('> Training starting ...')

for epoch in range(n_epochs):
    gan.train()
    for i, data in enumerate(dataloader):
        # real images
        real_images = data[0].to(device)
        batch_size = real_images.size(0)
        real_labels = torch.ones(batch_size, device=device)
        
        # fake images
        noise = torch.randn(batch_size, nz, 1, 1, device=device)
        fake_images = gan.forward_generator(noise)
        fake_labels = torch.zeros(batch_size, device=device)
        flipped_fake_labels = real_labels
        
        # train discriminator
        dis_opt.zero_grad()
        # discriminator loss on real images
        real_output = gan.forward_discriminator(real_images).view(-1)
        real_loss = loss_fn(real_output, real_labels)
        
        # discriminator loss on fake images
        fake_output = gan.forward_discriminator(fake_images.detach()).view(-1)
        fake_loss = loss_fn(fake_output, fake_labels)
        
        disc_total_loss = real_loss + fake_loss
        disc_total_loss.backward()
        dis_opt.step()
        
        # train generator
        gen_opt.zero_grad()
        disc_pred = gan.forward_discriminator(fake_images).view(-1)
        gen_loss = loss_fn(disc_pred, flipped_fake_labels)
        
        gen_loss.backward()
        gen_opt.step()
        
        # logging
        discriminator_losses.append(disc_total_loss.item())
        generator_losses.append(gen_loss.item())
        
        predicted_real = torch.where(real_output.detach() > 0., 1., 0.)
        predicted_fake = torch.where(fake_output.detach() > 0., 1., 0.)
        acc_real = (predicted_real == real_labels).sum().item() / batch_size
        acc_fake = (predicted_fake == fake_labels).sum().item() / batch_size
        discriminator_real_accuracies.append(acc_real)
        discriminator_fake_accuracies.append(acc_fake)
        
        # print progress
        if (i+1) % 1 == 0:
            print(f'Epoch [{epoch+1}/{n_epochs}], Step [{i+1}/{len(dataloader)}], '
                  f'Discriminator Loss: {disc_total_loss.item():.4f}, '
                  f'Generator Loss: {gen_loss.item():.4f}, '
                  f'Discriminator Accuracy on Real: {acc_real:.4f}, '
                  f'Discriminator Accuracy on Fake: {acc_fake:.4f}')
            
    # generate and save images for evaluation
    with torch.no_grad():
        fake_images = gan.forward_generator(noise).detach().cpu()
        img = torchvision.utils.make_grid(fake_images, padding=2, normalize=True)
        images_generated.append(img)

end_time = time.time()
total_time = end_time - start_time
total_time = f'{total_time//60:.0f}m {total_time%60:.0f}s'
print('Total training time:', total_time)

# display the last batch of images generated
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Generated Images")
plt.imshow(np.transpose(images_generated[-1],(1,2,0)))
plt.show()

# plot the losses
plt.plot(discriminator_losses, label='Discriminator loss')
plt.plot(generator_losses, label='Generator loss')
plt.title('Losses')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()
plt.show()

# save the model 
torch.save(gan.state_dict(), 'models/gan.pth')






# --------------- References ---------------
# https://www.youtube.com/watch?v=cTlxZ1FO1mY&list=PLTKMiZHVd_2KJtIXOW0zFhFfBaJJilH51&index=152
# https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
# https://www.youtube.com/watch?v=5fs9PMzrVig
# https://www.youtube.com/watch?v=OljTVUVzPpM