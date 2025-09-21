import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        return x

class SmallNetWithCrossAttention(nn.Module):
    def __init__(self,alpha_init=0.5):
        super(SmallNetWithCrossAttention, self).__init__()
        # First block with 1 input channel and 32 output channels
        self.block1 = ConvBlock(1, 32)        
        # Second block with 32 input channels and 32 output channels
        self.block2 = ConvBlock(32, 32)

        # Final layer after the blocks
        self.layer5 = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        # Cross-attention parameters (query, key, value projections)
        self.query_conv = nn.Conv2d(1, 1, kernel_size=1)  # Transform output (Q)
        self.key_conv = nn.Conv2d(1, 1, kernel_size=1)  # Transform residual (K)
        self.value_conv = nn.Conv2d(1, 1, kernel_size=1)  # Transform residual (V)

        self.alpha = nn.Parameter(torch.tensor(alpha_init))
    def forward(self, x, Ry, disable_attention=True, skip_connection=True):
        # Save the residual (input) for attention
        residual = Ry

        # Pass through the first block and apply skip connection
        x1 = self.block1(x)
        if skip_connection:
            x1 = x1 + x  # Skip connection after block1

        # Pass through the second block and apply skip connection
        x2 = self.block2(x1)
        if skip_connection:
            x2 = x2 + x1  # Skip connection after block2

        # Final convolutional layer to reduce back to 1 channel
        output = self.layer5(x2)

        if disable_attention:
            # Directly use the residual if attention is disabled
            final_output = (1 - self.alpha) * output + self.alpha * residual
            attention_scores = None  # No attention scores when attention is disabled
        else:
            # Cross-Attention Mechanism: Attention between residual and output
            # 1. Compute query (from output), key (from residual), and value (from residual)
            query = self.query_conv(output)  # Shape: [batch, 1, height, width]
            key = self.key_conv(residual)  # Shape: [batch, 1, height, width]
            value = self.value_conv(residual)  # Shape: [batch, 1, height, width]

            # 2. Calculate attention scores (dot-product of query and key)
            attention_scores = torch.mul(query, key)  # Element-wise multiplication (Q * K)
            attention_scores = F.softmax(attention_scores.view(attention_scores.shape[0], -1), dim=-1)
            attention_scores = attention_scores.view_as(query)  # Reshape back to original shape

            # 3. Use attention scores to weight the value (residual)
            attention_output = torch.mul(attention_scores, value)  # Shape: [batch, 1, height, width]

            # 4. Add the attention-weighted residual to the output
            final_output = (1 - self.alpha) * attention_output + self.alpha * residual

        return final_output, attention_scores


class CauchyProxNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, kernel_size=3):
        super(CauchyProxNet, self).__init__()

        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv2d(in_channels, num_features, kernel_size, padding=kernel_size//2)
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size, padding=kernel_size//2)
        self.conv3 = nn.Conv2d(num_features, num_features, kernel_size, padding=kernel_size//2)
        self.conv4 = nn.Conv2d(num_features, num_features, kernel_size, padding=kernel_size//2)
        self.conv5 = nn.Conv2d(num_features, out_channels, kernel_size, padding=kernel_size//2)

        # # Soft-thresholding layer to mimic Cauchy proximal operator
        # self.soft_thresh = SoftThresholding()

        # ReLU activation
        self.relu = nn.ReLU()

    def forward(self, x):
        # Save the input (for residual connection)
        x_input = x.clone()

        # Forward through the first conv layer and apply activation
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))

        # Apply soft-thresholding to induce sparsity
        # x = self.soft_thresh(x)

        # Final conv layer to generate output
        x = self.conv5(x)

        # Add residual connection (input to the network + the output)
        return x + x_input


# Combined CPSNet: Combining ForwardStepNet and CauchyProxNet
class CPS(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, num_features=64, kernel_size=3, step_size=0.1):
        super(CPS, self).__init__()

        # Forward step sub-network (Gradient descent-like update)
        # self.forward_step = SmallNet(in_channels, out_channels, num_features,kernel_size)
        self.forward_step = SmallNetWithCrossAttention()

        # Backward step sub-network (Cauchy proximal operator)
        self.backward_step = CauchyProxNet(in_channels, out_channels, num_features, kernel_size)

    def forward(self, x,y,disable_attention=False,skip_connection=True):
        # Apply forward optimization step
        x_forward,att_map = self.forward_step(x, y, disable_attention=disable_attention, skip_connection=skip_connection)

        # Apply backward Cauchy proximal step
        x_backward = self.backward_step(x_forward)

        # Visualize attention map for the first image in the batch
        # visualize_attention(att_map)

        return x_forward, x_backward
    

# # Example usage
# if __name__ == "__main__":
#     # Set up device
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     # Create directories for saving images
#     train_image_save_dir = f'./train_images_test/'
#     os.makedirs(train_image_save_dir, exist_ok=True)
#     val_image_save_dir = f'./val_images_test/'
#     os.makedirs(val_image_save_dir, exist_ok=True)

#     # Transform to convert images to tensor and resize them if needed
#     transform = transforms.Compose([
#         transforms.Resize((256, 360)),  # Resize to a smaller size to avoid memory issues
#         transforms.ToTensor(),  # Convert to tensor
#     ])

#     # Load the dataset from folders using the custom DataLoader
#     train_dataset = CustomImageDataset(image_dir='./data/train/', transform=transform)
#     val_dataset = CustomImageDataset(image_dir='./data/val/', transform=transform)

#     # DataLoader for train and validation sets
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

#     # Initialize network
#     # net = SmallNet().to(device)
#     net = CPS().to(device)
#     # net = SmallNetWithCrossAttention().to(device)


#     # Train the model
#     history = train_model(net, train_loader, val_loader)




# #######################################################################################
# # Model Implementation                                                                #
# #######################################################################################
# import os
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import matplotlib.pyplot as plt
# import numpy as np
# import torchvision
# # from torch.utils.tensorboard import SummaryWriter
# from torchvision.utils import save_image
# # from modified_radon import get_operators
# from subsampler import generate_subimages


# # RADON_H, FBP_H = get_operators(theta_begin=85, theta_end=95, n_angles=40, image_size=364, circle = False,device='cuda')
# # RADON_V, FBP_V = get_operators(theta_begin=-15, theta_end=15, n_angles=120, image_size=364, circle = False,device='cuda')

# # class ShrinkageActivation(nn.Module):
# #     def __init__(self):
# #         super(ShrinkageActivation, self).__init__()

# #     # implements the softh-thresholding function employed in CPS
# #     def forward(self, x, alpha):
# #         return torch.sign(x) * torch.max(torch.zeros_like(x), torch.abs(x) - alpha)


# #######################################################################################
# # CNN Filter
# #######################################################################################
# # class Filter1(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super(Filter1, self).__init__()
        
# #         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
# #         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)
        
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = nn.LeakyReLU()(x)
# #         x = self.conv2(x)
# #         x = nn.LeakyReLU()(x)
# #         x = self.conv3(x)
# #         # x = nn.LeakyReLU()(x)
# #         # x = self.conv4(x)
# #         # x = nn.LeakyReLU()(x)
# #         # x = self.conv5(x)
# #         # x = nn.LeakyReLU()(x)
        
# #         return x
    
# # class Filter2(nn.Module):
# #     def __init__(self, in_channels, out_channels):
# #         super(Filter2, self).__init__()
        
# #         self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
# #         self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
# #         self.conv3 = nn.Conv2d(32, out_channels, kernel_size=3, stride=1, padding=1)

        
# #     def forward(self, x):
# #         x = self.conv1(x)
# #         x = nn.LeakyReLU()(x)
# #         x = self.conv2(x)
# #         x = nn.LeakyReLU()(x)
# #         x = self.conv3(x)
# #         x = nn.LeakyReLU()(x)
# #         # x = self.conv4(x)
# #         # x = nn.LeakyReLU()(x)
# #         # x = self.conv5(x)
# #         # x = nn.LeakyReLU()(x)
        
# #         return x

# # ######################################################################################
# # # MLP Filter
# # ######################################################################################
# # class Filter1(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super(Filter1, self).__init__()
# #         self.fc1 = nn.Linear(input_size, hidden_size)
# #         self.fc2 = nn.Linear(hidden_size, output_size)
        
# #     def forward(self, x):
# #         x = torch.flatten(x, 1)
# #         x = torch.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x

# # class Filter2(nn.Module):
# #     def __init__(self, input_size, hidden_size, output_size):
# #         super(Filter2, self).__init__()
# #         self.fc1 = nn.Linear(input_size, hidden_size)
# #         self.fc2 = nn.Linear(hidden_size, output_size)
        
# #     def forward(self, x):
# #         x = torch.flatten(x, 1)
# #         x = torch.relu(self.fc1(x))
# #         x = self.fc2(x)
# #         return x



# # RADON, FBP = get_operators(n_angles=720, image_size=782, circle = True,device='cuda')
# # file_path = 'F:/LUScode/DUCPS/validation_results/iter7/Random2/Fold2/parameters.txt'
# # directory, filename = os.path.split(file_path)
# # Definition of the decoder

# class Prox(nn.Module):
#     def __init__(self):
#         super(Prox, self).__init__()
        
#         # Encoder
#         self.encoder_conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(64)  # Add batch normalization
#         self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(128)  # Add batch normalization

#         # Decoder (with skip connections)
#         self.decoder_conv1 = nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(64)  # Add batch normalization
#         self.decoder_conv2 = nn.Conv2d(64 + 1, 1, kernel_size=3, padding=1)

#         # Activation
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         # Encoding path
#         x1 = self.relu(self.bn1(self.encoder_conv1(x)))  # Output: (64, H, W)
#         x2 = self.relu(self.bn2(self.encoder_conv2(x1)))  # Output: (128, H, W)

#         # Decoding path with skip connections
#         x3 = self.relu(self.bn3(self.decoder_conv1(torch.cat([x2, x1], dim=1))))  # Concatenate x1 from encoder
#         output = self.sigmoid(self.decoder_conv2(torch.cat([x3, x], dim=1)))  # Concatenate input x

#         return output
    
# class CPS(nn.Module):
#     def __init__(
#         self,        
#         CPS_iterations=10

#     ):
#         super(CPS, self).__init__()
#         self.CPS_iterations = CPS_iterations
#         # self.size1 = size1  # Store size1 as an instance variable
#         # self.size2 = size2  # Store size2 as an instance variable


#         # Initialize lists to hold the parameters for each iteration
#         self.W_list = nn.ParameterList()
#         self.mu_list = nn.ParameterList()
    
#         for i in range(self.CPS_iterations):
#             W_i = nn.Parameter(self._init_W(), requires_grad=True)
#             self.W_list.append(W_i)
#             mu_i = nn.Parameter(self._init_mu(), requires_grad=True)
#             self.mu_list.append(mu_i)

#         # Initialize the denoising network as part of CPS
#         self.denoising_net = Prox()

#     def _init_W(self):
#         # initialization of the matrix W
#         # shape = [batch_size, 1, x.shape[2], x.shape[3]]
#         # init = torch.ones(1, 1,self.size1, self.size2) - 5e-5 
#         init = torch.ones(1, 1)- 5e-5 
#         return init
    
#     def _init_mu(self):
#         # initialization of the step size
#         init = torch.ones(1, 1)*5e-5 
#         return init

#     # def prox(self, x, mu):
#     #     gamma = 1*(torch.sqrt(mu)/2)

#     #     p = torch.square(gamma) + 2*mu - torch.square(x)/3

#     #     q = x*torch.square(gamma) + 2*torch.pow(x, 3)/27 - (x/3)*(torch.square(gamma) + 2*mu)

#     #     DD = torch.pow(p, 3)/27 + torch.square(q)/4

#     #     s = torch.pow(torch.abs(q/2 + torch.sqrt(DD)),1/3)*torch.sign(q/2 + torch.sqrt(DD))

#     #     t = torch.pow(torch.abs(q/2 - torch.sqrt(DD)),1/3)*torch.sign(q/2 - torch.sqrt(DD))

#     #     u = x/3 + s + t

#     #     # self.writer.add_histogram('Tensor Name', self.theta, _)    
#     #     # truncate the reconstructed x_hat, so that it lies in the same values' interval as the original x
#     #     # self.writer.close()
#     #     return u      

#     def forward(self, Ry, mask = None):      

#         x = torch.zeros_like(Ry)
#         delta_x = []
#         old_X = x
#         intermediate_outputs = []
#         p_mu = []
#         p_W =[]
        

#         for i in range(self.CPS_iterations):
#             W_i = self.W_list[i]
#             mu_i = self.mu_list[i]

#             # Ensure mu is positive
#             positive_W = F.softplus(W_i, beta=10000, threshold=0)
#             positive_mu = F.softplus(mu_i, beta=10000, threshold=0)
#             p_W.append(positive_W)
#             p_mu.append(positive_mu.item())
#             # version I : z = x - mu*R*R^T*x + mu*Ry = (I - mu*R*R^T)x + mu*Ry = W*x + mu*Ry
#             # out_x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
#             # save_image(out_x, 'checking/x'+str(i)+'.tif')
#             z = positive_W * x + positive_mu * Ry
#             # z = (z-torch.min(z))/(torch.max(z)-torch.min(z))
#             # print(torch.max(out_z).item())
#             # save_image(out_z, 'checking/z'+str(i)+'.tif') 

#             # version II : z = x - mu*R*R^T*x + mu*Ry = x - mu*(x - Ry)
#             # z = x- positive_mu*(x - Ry) 
#             x = self.denoising_net(z)
#             out_x = (x-torch.min(x))/(torch.max(x)-torch.min(x))
#             intermediate_outputs.append(out_x.detach().clone())
#             # Compute the change in x for convergence monitoring
#             if torch.max(torch.abs(old_X)) > 0:
#                 delta = (torch.max(torch.abs(x - old_X)) / torch.max(torch.abs(old_X))).item()
#             else:
#                 delta = torch.max(torch.abs(x - old_X)).item()
#             delta_x.append(delta)
#             old_X = x


#         return x,delta_x,p_W,p_mu,intermediate_outputs
