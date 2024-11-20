'''Ref_2023_npj Comp Mat_A neural network model for high entropy alloy design'''
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# number of elements
N_ELEM = 10
# number of elemental fetures
N_ELEM_FEAT = 30
# number of elemental fetures + 1
N_ELEM_FEAT_P1 = N_ELEM_FEAT + 1
# number of process conditions
N_PROC = 4
# learning rate
LEARNING_RATE = 5e-4

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class CnnDnnModel(nn.Module):
    '''
        CNN, ELU, batch normalization
        DNN, ELU, drop out
        Attention mech.
        Residual.

        # nn.Conv2d default: stride = 1, padding = 0
    '''
    def __init__(self):
        super(CnnDnnModel, self).__init__()
        
        # 卷积层
        self._kernel_size = (1, N_ELEM_FEAT_P1)
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn1 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        self.conv2 = nn.Conv2d(in_channels = 1, out_channels = N_ELEM_FEAT_P1, kernel_size = self._kernel_size)
        self.bn2 = nn.BatchNorm2d(N_ELEM_FEAT_P1)
        # self.attention = SelfAttention(1)
        
        # 全连接层
        self._num_fc_neuron = N_ELEM * N_ELEM_FEAT_P1 + N_PROC
        self.fc1 = nn.Linear(self._num_fc_neuron, 128)
        self.fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)
        self.leaky_relu = nn.ELU(0.2)

        self.reset_parameters()

        self.lr = LEARNING_RATE  # learning rate
        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
    
    def reset_parameters(self):
        self.conv1.weight.data.uniform_(*hidden_init(self.conv1))
        self.conv2.weight.data.uniform_(*hidden_init(self.conv2))
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, comp, elem_feature, proc):
        '''
            comp: (batch_size, 1, number_of_elements, 1)
            elem_feature: (batch_size, 1, number_of_elements, number_of_elemental_features), NOTE alway fixed
            prop: (batch_size, 1, N_PROC, 1)

            NOTE: 
                For the model, elem_feature seems to be different for each sample, 
                but ...
        '''
        x = torch.cat([comp, elem_feature], dim=-1)
        residual = x
        
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = x.reshape(-1, 1, N_ELEM, N_ELEM_FEAT_P1)

        # # Attention机制 TODO
        # x = self.attention(x)
        
        # 残差连接
        x += residual
        
        # 展平
        x = x.view(-1, N_ELEM * N_ELEM_FEAT_P1)

        # 链接process condition
        x = torch.cat([x, proc.reshape(-1, N_PROC)], dim=-1)
        
        # 全连接层
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

if __name__ == '__main__':
    _batch_size = 8
    test_input = (torch.ones((_batch_size, 1, N_ELEM, 1)), \
                 torch.ones((_batch_size, 1, N_ELEM, N_ELEM_FEAT)), \
                 torch.ones((_batch_size, 1, N_PROC, 1)))
    model = CnnDnnModel()
    print(model(*test_input).size())