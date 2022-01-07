from typing import Set
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torch.utils import data
import numpy as np

from time import time
import random
import logging
from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.cm
import logging

# This is a dirty workaround for a stupid problem with pytorch and osx that mismanage openMP
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CVAE_2D(nn.Module):
    """
    This is the convolutionnal variationnal autoencoder for the 2D starmen dataset.
    """

    def __init__(self):
        super(CVAE_2D, self).__init__()
        nn.Module.__init__(self)
        self.beta = 5
        self.criterion = nn.MSELoss()
        self.lr = 1e-4                                            # For epochs between MCMC steps
        self.epoch = 0                                            # For tensorboard to keep track of total number of epochs
        self.name = 'CVAE_2D'   

        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)     # 32 x 16 x 16 
        self.conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)    # 64 x 8 x 8 
        self.conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)   # 128 x 4 x 4
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc10 = nn.Linear(2048, 10)
        self.fc11 = nn.Linear(2048, 10)

        #self.fc2 = nn.Linear(8, 64)
        self.fc3 = nn.Linear(10,2048)
        self.upconv1 = nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)    # 128 x 8 x 8 
        self.upconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)   # 64 x 16 x 16 
        self.upconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)    # 3 x 32 x 32
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(32)
        #self.dropout = nn.Dropout(0.4)

    def encoder(self, image):
        h1 = F.relu(self.bn1(self.conv1(image)))
        h2 = F.relu(self.bn2(self.conv2(h1)))
        h3 = F.relu(self.bn3(self.conv3(h2)))
        mu = torch.tanh(self.fc10(h3.flatten(start_dim=1)))
        #mu = (mu + torch.ones(mu.shape[1:])
        logVar = self.fc11(h3.flatten(start_dim=1))
        return mu, logVar

    def decoder(self, encoded):
        #h5 = F.relu(self.fc2(encoded))
        h6 = F.relu(self.fc3(encoded)).reshape([encoded.size()[0], 128, 4, 4])
        h7 = F.relu(self.bn4(self.upconv1(h6)))
        h8 = F.relu(self.bn5(self.upconv2(h7)))
        reconstructed = F.relu(self.upconv3(h8))
        return reconstructed
    
    def reparametrize(self, mu, logVar):
        # Reparameterization takes in the input mu and logVar and sample the mu + std * eps
        std = torch.exp(logVar/2).to(device)
        eps = torch.normal(mean=torch.tensor([0 for i in range(std.shape[1])]).float(), std =1).to(device)
        if self.beta != 0:                   # beta VAE
            return mu + eps*std
        else:                           # regular AE
            return mu
        
    def forward(self, image):
        mu, logVar = self.encoder(image)
        if self.training:
            encoded = self.reparametrize(mu, logVar)
        else:
            encoded = mu
        reconstructed = self.decoder(encoded)
        return mu, logVar, reconstructed

    def plot_images_vae(self, data, n_images, writer=None):
        
        # Plot the reconstruction
        fig, axes = plt.subplots(2, n_images, figsize=(10,2))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(n_images):
            test_image = random.choice(data)
            test_image = Variable(test_image.unsqueeze(0)).to(device)
            mu, logVar, out = self.forward(test_image)
            axes[0][i].matshow(255*test_image[0][0].cpu().detach().numpy())
            axes[1][i].matshow(255*out[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])
        
        if writer is not None:
            writer.add_images('reconstruction', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_reconstruction.png', bbox_inches='tight')
        plt.close()

        # Plot simulated data in all directions of the latent space
        fig, axes = plt.subplots(mu.shape[1], 7, figsize=(14,2*Settings().dimension))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(mu.shape[1]):
            for j in range(-3,4):
                simulated_latent = torch.zeros(mu.shape)
                simulated_latent[0][i] = j/4
                simulated_img = self.decoder(simulated_latent.unsqueeze(0).to(device))
                axes[i][(j+3)%7].matshow(255*simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('latent_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_latent.png', bbox_inches='tight')
        plt.close()        
        self.training = True


    def plot_images_longitudinal(self, encoded_images, writer=None):
        self.training = False
        nrows, ncolumns = encoded_images.shape[0], encoded_images.shape[1]
        fig, axes = plt.subplots(nrows, ncolumns, figsize=(14,14))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(nrows):
            for j in range(ncolumns):
                simulated_img = self.decoder(encoded_images[i][j].unsqueeze(0).to(device))
                axes[i][j].matshow(simulated_img[0][0].cpu().detach().numpy())
        for axe in axes:
            for ax in axe:
                ax.set_xticks([])
                ax.set_yticks([])

        if writer is not None:
            writer.add_images('longitudinal_directions', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_simulation_longitudinal.png', bbox_inches='tight')
        plt.close('all')
        self.training = True

    def plot_images_gradient(self, encoded_gradient, writer=None):
        self.training = False
        
        ncolumns = encoded_gradient.shape[0] 
        fig, axes = plt.subplots(1, ncolumns, figsize=(6,2))
        decoded_p0 = self.decoder(torch.zeros(encoded_gradient[0].shape).unsqueeze(0).to(device))
        plt.subplots_adjust(wspace=0, hspace=0)
        for i in range(0,ncolumns):
            simulated_img = self.decoder(encoded_gradient[i].unsqueeze(0).to(device)) - decoded_p0
            axes[i].matshow(simulated_img[0][0].cpu().detach().numpy(), cmap=matplotlib.cm.get_cmap('bwr'), norm=matplotlib.colors.CenteredNorm())
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        if writer is not None:
            writer.add_images('Gradient', fig2rgb_array(fig), self.epoch, dataformats='HWC')

        plt.savefig('qc_gradient.png', bbox_inches='tight')
        plt.close('all')
        self.training = True



    def evaluate(self, data, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        criterion = self.loss
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,Settings().dimension])

        with torch.no_grad():
            for data in dataloader:

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER)
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)
            writer.add_scalars('Loss/alignment', {'test' : talignment_loss/nb_batches, 'train' : train_losses[2]}, self.epoch)

        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data

    def loss(self, mu, logVar, reconstructed, input_):
        kl_divergence = 0.5 * torch.sum(-1 - logVar + mu.pow(2) + logVar.exp()) / mu.shape[0]
        #recon_error = torch.nn.MSELoss(reduction='mean')(reconstructed, input_)
        recon_error = torch.sum((reconstructed - input_)**2) / input_.shape[0]
        return recon_error, kl_divergence

    def train(self, data_loader, test, optimizer, num_epochs=20, longitudinal=None, individual_RER=None, writer=None):

        self.to(device)
        criterion = self.loss
        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
            tmu, tlogvar = torch.zeros((1,Settings().dimension)).to(device), torch.zeros((1,Settings().dimension)).to(device)
            nb_batches = 0

            for data in data_loader:
                optimizer.zero_grad()

                if longitudinal is not None:
                    input_ = Variable(data[0]).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    alignment_loss = longitudinal(data, mu, reconstructed, individual_RER) 
                    loss = reconstruction_loss + self.beta * kl_loss + self.gamma * alignment_loss
                    trecon_loss += reconstruction_loss 
                    tkl_loss += kl_loss
                    talignment_loss += alignment_loss 
                    tmu = torch.cat((tmu, mu))
                    tlogvar = torch.cat((tlogvar, logVar))
                else:
                    input_ = Variable(data).to(device)
                    mu, logVar, reconstructed = self.forward(input_)
                    reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                    loss = reconstruction_loss + self.beta * kl_loss

                loss.backward()
                optimizer.step()
                tloss += float(loss)
                nb_batches += 1
            epoch_loss = tloss/nb_batches

            if writer is not None:
                self.epoch += 1
                train_losses = (trecon_loss/nb_batches, tkl_loss/nb_batches, talignment_loss/nb_batches)
                test_loss, _ = self.evaluate(test, longitudinal=longitudinal, individual_RER=individual_RER, writer=writer, train_losses=train_losses)
                writer.add_histogram('Mu', tmu, self.epoch)
                writer.add_histogram('Logvar', tlogvar, self.epoch)
            else:
                test_loss, _ = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")

            # Save images to check quality as training goes
            if longitudinal is not None:
                self.plot_images_vae(test.data, 10, writer)
            else:
                self.plot_images_vae(test, 10)

        print('Complete training')
        return
    
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
        nn.Module.__init__(self)
            
        # Discriminator
        self.d_conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)               # 32 x 16 x 16
        self.d_conv2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)              # 64 x 8 x 8
        self.d_conv3 = nn.Conv2d(64, 128, 3, stride=2, padding=1)             # 128 x 4 x 4
        self.d_conv4 = nn.Conv2d(128, 8, 3, stride=1, padding=1)              # 8 x 4 x 4
        self.d_bn1 = nn.BatchNorm2d(32)
        self.d_bn2 = nn.BatchNorm2d(64)
        self.d_bn3 = nn.BatchNorm2d(128)
        self.d_bn4 = nn.BatchNorm2d(1)
        self.relu1 = nn.LeakyReLU(0.02, inplace=True)
        self.relu2 = nn.LeakyReLU(0.02, inplace=True)
        self.relu3 = nn.LeakyReLU(0.02, inplace=True)
        self.relu4 = nn.LeakyReLU(0.02, inplace=True)
        self.d_fc = nn.Linear(128, 1)
        
    def forward(self, image):
        image = image 
        d1 = self.relu1(self.d_conv1(image))
        d2 = self.relu2(self.d_conv2(d1))
        d3 = self.relu3(self.d_conv3(d2))
        d4 = self.relu4(self.d_conv4(d3))
        d5 = torch.sigmoid(self.d_fc(d4.flatten(start_dim=1)))
        return d5
        
class VAE_GAN_2D(nn.Module):
    
    def __init__(self):
        super(VAE_GAN_2D, self).__init__()
        nn.Module.__init__(self)
        
        self.VAE = CVAE_2D()
        self.discriminator = discriminator()
        self.lr = 1e-4                                                      # For epochs between MCMC steps
        self.epoch = 0   
        self.name = 'VAE_GAN'           
        
    def train(self, data_loader, test, vae_optimizer, d_optimizer, num_epochs=20, writer=None):

        self.to(device)
        vae_criterion = self.VAE.loss
        d_criterion = nn.BCELoss(reduction='sum')

        best_loss = 1e10
        es = 0

        for epoch in range(num_epochs):

            start_time = time()
            if es == 100:
                break

            logger.info('Epoch {}/{}'.format(epoch+1, num_epochs ))

            tloss = 0.0
            trecon_loss, tkl_loss, talignment_loss, td_loss = 0.0, 0.0, 0.0, 0.0
            nb_batches = 0


            for data in data_loader:                
                input_ = Variable(data[0]).to(device)
                mu, logVar, reconstructed = self.VAE(input_)
                
                label_real = torch.rand((input_.size(0),), dtype=torch.float) / 10 + 0.05 * torch.ones((input_.size(0),)) # random labels between 0 and .1
                label_fake = torch.rand((input_.size(0),), dtype=torch.float) / 10 + 0.85 * torch.ones((input_.size(0),)) # random labels between .9 and 1
                labels = torch.cat((label_real, label_fake)).to(device)

                # Training the discriminator with input and reconstructed images 
                self.discriminator.zero_grad()
                d_input = torch.cat((input_, reconstructed))      # Pass them both as one single batch
                d_output = self.discriminator(d_input.detach()).view(-1)     # Need to detach the input to avoid weird gradient computation of noise and VAE
                d_loss = d_criterion(d_output, labels)
                d_loss.backward()
                d_optimizer.step()
                print(f"Full d_output : {d_output}")

                # Training the VAE generator (with reparametrization tricks)
                vae_optimizer.zero_grad()
                reconstruction_loss, kl_loss = vae_criterion(mu, logVar, input_, reconstructed)
                #output_generator = self.discriminator(reconstructed).view(-1)    # Another pass is necessary because D has been updated
                #d_loss = d_criterion(output_generator, label_real.to(device))               # Labels are considered true for the generator
                #print(reconstruction_loss, kl_loss, d_loss)
                loss = reconstruction_loss + self.VAE.beta * kl_loss# + 100 * d_loss
                d_loss = 0
                #loss = 100 * d_loss
                
                loss.backward()
                vae_optimizer.step()
                tloss += float(loss)
                td_loss += float(d_loss)
                nb_batches += 1
                
            epoch_loss = tloss/nb_batches
            epoch_d_loss = td_loss/nb_batches

            if writer is not None:
                self.epoch += 1
                train_losses = (trecon_loss/nb_batches, tkl_loss/nb_batches, talignment_loss/nb_batches, td_loss/nb_batches)
                test_loss, _ = self.evaluate(test, longitudinal=longitudinal, individual_RER=individual_RER, writer=writer, train_losses=train_losses)
                writer.add_histogram('Mu', tmu, self.epoch)
                writer.add_histogram('Logvar', tlogvar, self.epoch)
            else:
                test_loss, _ = self.evaluate(test)

            if epoch_loss <= best_loss:
                es = 0
                best_loss = epoch_loss
            else:
                es += 1
            end_time = time()
            logger.info(f"Epoch loss (train/test/discriminator): {epoch_loss:.3e}/{test_loss:.3e}/{epoch_d_loss:.3e} took {end_time-start_time} seconds")
            #logger.info(f"Epoch loss (train/test/discriminator): {epoch_loss:.3e}/{test_loss:.3e} took {end_time-start_time} seconds")
            
            # Save images to check quality as training goes
            self.VAE.plot_images_vae(test, 10, name='qc_reconstruction_test.png')
            self.VAE.plot_images_vae(data, 10, name='qc_reconstruction_train.png')


        print('Complete training')
        return
    
    def evaluate(self, data, longitudinal=None, individual_RER=None, writer=None, train_losses=None):
        """
        This is called on a subset of the dataset and returns the encoded latent variables as well as the evaluation
        loss for this subset.
        """
        self.to(device)
        self.training = False
        criterion = self.VAE.loss
        dataloader = torch.utils.data.DataLoader(data, batch_size=10, num_workers=0, shuffle=False)
        tloss = 0.0
        trecon_loss, tkl_loss, talignment_loss = 0.0, 0.0, 0.0
        nb_batches = 0
        encoded_data = torch.empty([0,Settings().dimension])

        with torch.no_grad():
            for data in dataloader:
                
                input_ = Variable(data).to(device)
                mu, logVar, reconstructed = self.VAE(input_)
                reconstruction_loss, kl_loss = criterion(mu, logVar, input_, reconstructed)
                loss = reconstruction_loss + self.beta * kl_loss

                tloss += float(loss)
                nb_batches += 1
                encoded_data = torch.cat((encoded_data, mu.to('cpu')), 0)

        if writer is not None:
            writer.add_scalars('Loss/recon', {'test' : trecon_loss/nb_batches, 'train' : train_losses[0]} , self.epoch)
            writer.add_scalars('Loss/kl', {'test' : tkl_loss/nb_batches, 'train' : train_losses[1]}, self.epoch)

        loss = tloss/nb_batches
        self.training = True
        return loss, encoded_data
    

class Dataset(data.Dataset):
    def __init__(self, images, labels, timepoints):
        self.data = images
        self.labels = labels
        self.timepoints = timepoints

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        "Generates one sample of data"
        # Select sample
        X = self.data[index]
        y = self.labels[index]
        z = self.timepoints[index]
        return X, y, z

def fig2rgb_array(fig):
    fig.canvas.draw()
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return(data)

def main():
    """
    For debugging purposes only, once the architectures and training routines are efficient,
    this file will not be called as a script anymore.
    """
    logger.info(f"Device is {device}")

    epochs = 250
    batch_size = 4
    lr = 1e-5
    criterion = nn.MSELoss()

    # Load data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 4

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

    autoencoder = VAE_GAN_2D()

    optimizer_fn = optim.Adam
    vae_optimizer = optimizer_fn(autoencoder.VAE.parameters(), lr=lr)
    d_optimizer = optimizer_fn(autoencoder.discriminator.parameters(), lr=lr)
    autoencoder.train(trainloader, test=testloader, vae_optimizer=vae_optimizer, d_optimizer=d_optimizer, num_epochs=epochs)
    torch.save(autoencoder.state_dict(), path_LAE)

    return autoencoder



if __name__ == '__main__':
    main()



