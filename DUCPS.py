import os
import random
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from model import CPS
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
import torchvision
from data_loader import Singledataset,SRdataset,Radondataset
from torchvision.transforms import Compose, Normalize, ToTensor, Grayscale, Resize, Pad
from torchvision.utils import save_image
from skimage.metrics import structural_similarity as ssim
import argparse
from PIL import Image
import wandb


# Pass network's parameter as arguments

def parse_args():
    parser = argparse.ArgumentParser("Network's parameters")

    parser.add_argument(
        "--layers",
        type=int,
        default=1,
        help="Number of layers/iterations",
    )

    parser.add_argument(
        "--learning-rate",
        type = float,
        default=1e-4,
        help = "Learning rate",
    )

    parser.add_argument(
        "--epochs",
        type = int,
        default=20,
        help = "Number of Epochs",
    )

    parser.add_argument("--random", type=int, default=1, help="Random number")
    parser.add_argument("--fold", type=int, default=3, help="Fold number")
    parser.add_argument("--tv", type=float, default=0.0, help="TV regularization weight")
    parser.add_argument("--Cauchy", type=float, default=1e-2, help="Cauchy regularization weight")
    parser.add_argument("--disable_attention", type=int, default=0, help="disable_attention")
    parser.add_argument("--skip", type=int, default=0, help="skip_connection")

    return parser.parse_args()


def print_args(args):
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")

ARGS = parse_args()
print_args(ARGS)

# Model parameters
CPS_iterations = ARGS.layers  # Number of CPS iterations during forward
learning_rate = ARGS.learning_rate  # Adam Learning rate
num_epochs = ARGS.epochs  # Epochs to train
#######################################################################################
Random = ARGS.random
Fold = ARGS.fold
train_dir_spatial = f'F:/LUScode/dataset/DUCV/random{Random}/Fold{Fold}/train'
val_dir_spatial = f'F:/LUScode/dataset/DUCV/ForDUVal/random{Random}/Fold{Fold}/val'
train_dir_radon = f'F:/LUScode/dataset/DUCV/random{Random}/Fold{Fold}/radon_train'
val_dir_radon = f'F:/LUScode/dataset/DUCV/random{Random}/Fold{Fold}/radon_val'

# Create directories for saving images
# train_image_save_dir = f'./train_images_test/random{Random}/Fold{Fold}/iter{CPS_iterations}/'
# os.makedirs(train_image_save_dir, exist_ok=True)
val_image_save_dir = f'./val_images/FULL/random{Random}/Fold{Fold}/TV{ARGS.tv}_Cauchy{ARGS.Cauchy}_iter{CPS_iterations}_att{ARGS.disable_attention}/skip{ARGS.skip}/'
os.makedirs(val_image_save_dir, exist_ok=True)



#######################################################################################
# Data Loading & Utility functions                                                    #
#######################################################################################

def get_data_loaders(train_batch_size, val_batch_size,train_dir,val_dir):
    # data_transform = Compose([ToTensor(),Normalize((0.5, ), (0.5, )),Grayscale(num_output_channels=1)])
    # data_transform_spatial = Compose([ToTensor(),Pad(53),Grayscale(num_output_channels=1)])
    data_transform_radon = ToTensor()
    # data_transform_radon = Compose([Resize([256,256]),ToTensor()])

    # data_transform = Compose([Resize(64),ToTensor()])
    trainset = Radondataset(train_dir,transform=data_transform_radon)
    valset = Singledataset(val_dir,transform=data_transform_radon)

    # trainset = SRdataset(train_dir_spatial,train_dir_radon,spatial_transform=data_transform_spatial,radon_transform=data_transform_radon)
    # valset = SRdataset(val_dir_spatial,val_dir_radon,spatial_transform=data_transform_spatial,radon_transform=data_transform_radon)


    train_loader = DataLoader(trainset, batch_size= train_batch_size, shuffle=True)
    val_loader = DataLoader(valset, batch_size= val_batch_size, shuffle=False)

    return train_loader, val_loader

# Function to visualize attention map (only for the first image in the batch)
def visualize_attention(attention_map, title="Attention Map"):
    attention_map = attention_map[0].squeeze().detach().cpu().numpy()  # Only the first image in the batch
    plt.imshow(attention_map, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.show()

# Calculate relative L2 norm for a batch
def relative_l2_norm_batch(output, input):
    batch_size = output.shape[0]
    norms = []
    for i in range(batch_size):
        norm_diff = torch.norm(output[i] - input[i])
        norm_input = torch.norm(input[i])
        norms.append((norm_diff / norm_input).item())
    return np.mean(norms)

# Calculate Mean Structural Similarity Index (MSSIM) for a batch
def calculate_mssim_batch(output, input):
    batch_size = output.shape[0]
    mssim_vals = []
    for i in range(batch_size):
        output_img = output[i].squeeze().detach().cpu().numpy()
        input_img = input[i].squeeze().detach().cpu().numpy()
        mssim_vals.append(ssim(output_img, input_img,data_range=(output_img.max()-output_img.min())))
    return np.mean(mssim_vals)

# Calculate Signal-to-Noise Ratio (SNR) for a batch
def calculate_snr_batch(output, input):
    batch_size = output.shape[0]
    snr_vals = []
    for i in range(batch_size):
        signal_power = torch.sum(input[i] ** 2)
        noise_power = torch.sum((input[i] - output[i]) ** 2)
        snr_vals.append((10 * torch.log10(signal_power / noise_power)).item())
    return np.mean(snr_vals)

# Total variation regularization
def tv_loss(img):
    batch_size, channels, height, width = img.size()
    tv_h = torch.pow(img[:, :, 1:, :] - img[:, :, :-1, :], 2).sum()
    tv_w = torch.pow(img[:, :, :, 1:] - img[:, :, :, :-1], 2).sum()
    return (tv_h + tv_w) / (batch_size * channels * height * width)

def cauchy_regularization(x, gamma=0.5):
    """
    Cauchy regularization term: log(1 + gamma * |x|^2), averaged by the batch size.
    
    Args:
    x : tensor
        Input tensor for which to compute the regularization.
    gamma : float
        Parameter controlling the scale of the Cauchy regularization.
    
    Returns:
    reg : tensor
        The averaged Cauchy regularization term.
    """
    batch_size, channels, height, width = x.size()
    
    # Calculate Cauchy regularization term
    reg = torch.sum(torch.log(1 + gamma * x**2))
    
    # Return the regularization, normalized by batch size, channels, and pixel count
    return reg / (batch_size * channels * height * width)

# Loss function with MSE and TV regularization
def total_loss(y_pred, x, tv_weight=1e-1, Cauchy_weight=1e-4):
    mse_loss = nn.MSELoss()(y_pred, x)
    tv_reg = tv_loss(y_pred)
    cauchy_reg = cauchy_regularization(y_pred)
    # print("mse:",mse_loss.item(),"tv:", tv_weight * tv_reg.item(), "Cauchy:", Cauchy_weight*cauchy_reg.item())
    return mse_loss + tv_weight * tv_reg + Cauchy_weight*cauchy_reg


#######################################################################################
# Training Functions                                                                  #
#######################################################################################
# Train the model
def train_model(net, train_loader, val_loader, num_iter = 10, num_epochs=30):
    min_val_loss = float('inf')
    optimizer = Adam(net.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
    history = {'train_loss': [], 'val_loss': [], 'outputs': [],'alpha':[],'train_l2':[],'train_ssim':[],'train_snr':[],'val_l2':[],'val_ssim':[],'val_snr':[]}

    for epoch in range(num_epochs):
        net.train()
        train_loss = 0
        for batch_idx, (rbatch1, rbatch2, pth) in enumerate(train_loader):
            rbatch1 = rbatch1.to(device)  # Ensure images are on device
            rbatch2 = rbatch2.to(device)  # Ensure images are on device
            optimizer.zero_grad()
            # outputs = torch.zeros_like(images)+1e-1
            outputs = rbatch1
            iteration_loss = 0

            for iteration in range(num_iter):  # 10 iterations per image
                _,outputs = net(outputs,rbatch1,disable_attention=ARGS.disable_attention,skip_connection=ARGS.skip)
                # save_image(outputs.detach(), train_image_save_dir+str(batch_idx)+"_"+str(iteration)+'.png')
                loss = total_loss(outputs, rbatch2,tv_weight=ARGS.tv, Cauchy_weight=ARGS.Cauchy)
                iteration_loss += loss

                # Visualize the outputs after each iteration
                # visualize_outputs(epoch, iteration, outputs)  # Pass the current outputs
                wandb.log({f"alpha": net.forward_step.alpha.item()})
                # history['alpha'].append(net.forward_step.alpha.item())

            iteration_loss.backward()
            optimizer.step()
            train_loss += iteration_loss.item()

        scheduler.step()
        # Calculate metrics
        train_l2_norm = relative_l2_norm_batch(outputs, rbatch1)
        train_mssim_value = calculate_mssim_batch(outputs, rbatch1)
        train_snr_value = calculate_snr_batch(outputs, rbatch1)

        history['train_loss'].append(train_loss / len(train_loader))
        # history['train_l2'].append(train_l2_norm)
        # history['train_ssim'].append(train_mssim_value)
        # history['train_snr'].append(train_snr_value)
        wandb.log({f"train_loss": train_loss / len(train_loader)})
        wandb.log({f"train_Relative L2 Norm": train_l2_norm})
        wandb.log({f"train_MSSIM": train_mssim_value})
        wandb.log({f"train_SNR": train_snr_value})

        # Validation step
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for images,pth in val_loader:
                images = images.to(device)
                outputs = images
                iteration_val_loss = 0
                for i in range(num_iter):
                    _,outputs = net(outputs,images,disable_attention=ARGS.disable_attention,skip_connection=ARGS.skip)
                    # save_image(outputs.detach(), val_image_save_dir+str(i)+'.png')
                    iteration_val_loss += total_loss(outputs, images,tv_weight=ARGS.tv, Cauchy_weight=ARGS.Cauchy)

                val_loss += iteration_val_loss.item()
                if epoch == num_epochs - 1:  # Save the final outputs of the last epoch
                    # history['outputs'].append(outputs.cpu())
                    os.makedirs(val_image_save_dir+f'epoch{epoch}/', exist_ok=True)
                    save_image(outputs.detach(), val_image_save_dir+f'epoch{epoch}/'+str(pth)[2:-9]+'.png')

        history['val_loss'].append(val_loss / len(val_loader))

        # Save the model with the lowest validation loss
        if history['val_loss'][-1] < min_val_loss:        
            min_val_loss = history['val_loss'][-1]
            os.makedirs(f'./trained_model/TV{ARGS.tv}_Cauchy{ARGS.Cauchy}_att{ARGS.disable_attention}_skip{ARGS.skip}/', exist_ok=True)
            torch.save(net.state_dict(), f'trained_model/TV{ARGS.tv}_Cauchy{ARGS.Cauchy}_att{ARGS.disable_attention}_skip{ARGS.skip}/Iter{CPS_iterations}_Random{Random}Fold{Fold}.pth')

        # Calculate metrics
        val_l2_norm = relative_l2_norm_batch(outputs, images)
        val_mssim_value = calculate_mssim_batch(outputs, images)
        val_snr_value = calculate_snr_batch(outputs, images)

        # history['val_l2'].append(val_l2_norm)
        # history['val_ssim'].append(val_mssim_value)
        # history['val_snr'].append(val_snr_value)
        wandb.log({f"val_loss": val_loss / len(val_loader)})
        wandb.log({f"val_Relative L2 Norm": val_l2_norm})
        wandb.log({f"val_MSSIM": val_mssim_value})
        wandb.log({f"val_SNR": val_snr_value})

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.6f}, Val Loss: {val_loss / len(val_loader):.5f}')

    return history

#  Validation step
def run_time(model, val_loader):
    model.eval()
    execution_time = 0.0
    with torch.no_grad():
        
        for images,pth in val_loader:
            images = images.to(device)
            outputs = images
            start_time = time.time()
            # for i in range(10):
            _,outputs = model(outputs,images)
            end_time = time.time()
            execution_time += (end_time - start_time)
            save_image(outputs.detach(), val_image_save_dir+str(pth)[2:-9]+'.png')
        
    execution_time = execution_time/len(val_loader)
    print(f"average deep unfolding execution time per image: {execution_time:.4f} seconds")
    # wandb.log({f"execution_time": execution_time})
#######################################################################################
# Main                                                                                #
#######################################################################################
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    wandb.init(project='CPSnet_ablation',group = f"{ARGS.tv}_{ARGS.Cauchy}_layer{CPS_iterations}_att{ARGS.disable_attention}_skip{ARGS.skip}", name =f"{ARGS.tv}_{ARGS.Cauchy}_r{Random}f{Fold}_layer{CPS_iterations}_att{ARGS.disable_attention}",\
               id=f"{ARGS.tv}_{ARGS.Cauchy}_r{Random}f{Fold}_layer{CPS_iterations}_att{ARGS.disable_attention}_skip{ARGS.skip}",mode="offline")

    train_loader, val_loader = get_data_loaders(4,1, train_dir_radon,val_dir_radon)

    model = CPS().to(device)

    print(model)

    history = train_model(model, train_loader, val_loader, num_iter = CPS_iterations, num_epochs= num_epochs)

    # model.load_state_dict(torch.load(f'trained_model/Iter{CPS_iterations}_Random{Random}Fold{Fold}.pth'))
    # run_time(model, val_loader)




