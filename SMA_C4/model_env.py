'''
Reference: 2023_npj Computational Materials_A neural network model for high entropy alloy design

This module implements a neural network model for predicting properties of shape memory alloys (SMAs)
based on their elemental composition and processing conditions. The architecture combines
convolutional neural networks (CNNs) with fully connected layers to effectively capture
materials structure-property relationships.
'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Constants for model architecture
# number of elements in the alloy composition
N_ELEM = 11
# number of elemental features (physical, chemical properties of elements)
N_ELEM_FEAT = 30
# number of elemental fetures + 1
N_ELEM_FEAT_P1 = N_ELEM_FEAT + 1
# number of process conditions (temperature, time, etc.)
N_PROC = 3
# learning rate for model optimization
LEARNING_RATE = 5e-4

def hidden_init(layer):
    """
    Initialize the weights for hidden layers using a uniform distribution
    with bounds based on the fan-in of the layer.
    
    This helps with stable training by preventing vanishing/exploding gradients.
    
    Args:
        layer: Neural network layer to initialize
        
    Returns:
        Tuple of lower and upper bounds for uniform initialization
    """
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CnnDnnModel(nn.Module):
    '''
    Combined CNN-DNN model for predicting properties of alloys
    
    Architecture features:
    - CNN layers: Extract features from elemental properties and compositions
    - ELU activation: Provides non-linearity with smoother gradients than ReLU
    - Batch normalization: Stabilizes and accelerates training
    - Dropout: Prevents overfitting
    - Residual connections: Improves gradient flow through deep networks
    
    The model processes:
    1. Composition data
    2. Elemental features (physical/chemical properties); no gradient propagated to this part
    3. Processing conditions
    
    To predict target material properties (e.g., phase transformation properties).
    
    Note: Conv2d default parameters: stride = 1, padding = 0
    '''
    def __init__(self):
        """Initialize the CNN-DNN model with appropriate layers and parameters"""
        super(CnnDnnModel, self).__init__()
        
        # Convolutional layers
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        # First conv block: extracts features from elemental data
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        # Second conv block: further refines features
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        
        # Fully connected layers for regression
        self._num_fc_neuron = N_ELEM * N_ELEM_FEAT_P1 + N_PROC
        self.fc1 = nn.Linear(self._num_fc_neuron, 128)
        self.fc2 = nn.Linear(128, 1)  # Output layer for property prediction
        self.dropout = nn.Dropout(0.5)  # Regularization to prevent overfitting
        self.leaky_relu = nn.ELU(0.2)  # ELU activation with alpha=0.2

        # Initialize weights for stable training
        self.reset_parameters()

        # Define optimizer for training
        self.lr = LEARNING_RATE
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        """
        Initialize network parameters using custom initialization method
        
        This ensures proper weight scaling at the beginning of training.
        """
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # Output layer uses smaller initialization values
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc):
        """
        Forward pass through the network
        
        Args:
            comp: Tensor of shape (batch_size, 1, number_of_elements, 1)
                 Contains the composition fractions for each element
            elem_feature: Tensor of shape (batch_size, 1, number_of_elements, number_of_elemental_features)
                         Contains the elemental features (physical/chemical properties)
            proc: Tensor of shape (batch_size, 1, N_PROC, 1)
                 Contains the processing conditions (temperature, time, etc.)
                 
        Returns:
            Tensor of predicted material properties (shape: batch_size, 1)
        
        Note: 
            For this model, elem_feature is fixed for each element but varies across elements
        """
        # Concatenate composition and elemental features along the last dimension
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x  # Store for residual connection
        
        # First convolutional block with batch normalization and activation
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        # Second convolutional block
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)

        
        # Apply residual connection to improve gradient flow
        x += residual
        
        # Flatten the convolutional feature maps
        x = x.view(-1, N_ELEM * N_ELEM_FEAT_P1)

        # Concatenate processing conditions
        x = torch.cat([x, proc.reshape(-1, N_PROC)], dim=-1)
        
        # Fully connected layers for final prediction
        x = self.leaky_relu(self.fc1(x))  # Hidden layer with activation
        x = self.dropout(x)               # Apply dropout for regularization
        x = self.fc2(x)                   # Output layer (property prediction)
        
        return x

if __name__ == '__main__':
    # Test code to verify model dimensions
    _batch_size = 8
    # Create dummy input tensors with appropriate shapes
    test_input = (torch.ones((_batch_size, 1, N_ELEM, 1)), \
                 torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)), \
                 torch.ones((_batch_size, 1, N_PROC, 1)))
    # Initialize model
    model = CnnDnnModel()
    # Verify output shape
    print(model(*test_input).size())