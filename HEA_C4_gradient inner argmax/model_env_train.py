import random
from typing import Callable, Tuple
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset

from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from model_env import N_ELEM, N_ELEM_FEAT, N_ELEM_FEAT_P1, N_PROC
from model_env import CnnDnnModel

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 设置随机数种子
set_seed(0) # default 0

seeds = np.random.randint(0, 9999, (9999, ))

def load_data():
    # Load the default dataset
    data = pd.read_excel('data\DSC-DATA-0613.xlsx', skiprows=[1,])
    
    # composition labels 
    comp_labels = ['Ti', 'Ni', 'Cu', 'Hf', 'Co', 'Zr', 'Fe', 'Pd', 'Ta', 'Nb', 'V']
                        
    # processing condition labels 
    proc_labels = ['冷轧变形量', '退火处理温度', '退火处理时间']

    # property labels, one by one for ['enthalpy（Heating）', 'Ms', 'Mp', 'Mf', 'As', 'Ap', 'Af']
    prop_labels = ['Mp']
    print(f'loading SMA data ...')

    ''' delete proc condition all 0 items '''
    _mask = (data[proc_labels[1:]] == 0).all(axis = 1)
    data = data[~_mask]
    print(f'deleted {sum(_mask)} items')

    comp_data = data[comp_labels].to_numpy()
    proc_data = data[proc_labels].to_numpy()
    prop_data = data[prop_labels].to_numpy()

    elem_feature = pd.read_excel('data\\sma_element_features.xlsx')
    elem_feature = elem_feature[comp_labels].to_numpy()  # transpose: column for each elemental feature, row for each element 

    # (num_samples, num_elements), (num_samples, num_proc), (num_samples, num_prop), (num_elements, num_elem_features,)
    return comp_data, proc_data, prop_data, elem_feature

def fit_transform(data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]):
    '''fit and transform the data'''
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler = \
        [StandardScaler() for _ in range(len(data_tuple))]
    
    comp_data = comp_data_scaler.fit_transform(comp_data)
    proc_data = proc_data_scaler.fit_transform(proc_data)
    prop_data = prop_data_scaler.fit_transform(prop_data)
    ''' 
        input elem_feature:     (num_elem_features, num_elements, ), as defined in the EXCEL file
        output elem_feature:    (num_elements, num_elem_features, )
        however sklearn scaler works colum-wise,
        should calculate the mu and sigma of element features (say, VEC) for diff elements,
        so transpose the elem_feature
    '''
    elem_feature = elem_feature_scaler.fit_transform(elem_feature.T)

    # return the data and the scalers
    return (
        (comp_data, proc_data, prop_data, elem_feature,),
        (comp_data_scaler, proc_data_scaler, prop_data_scaler, elem_feature_scaler,),
    )

class CustomDataset(Dataset):
    ''' store comp, proc, prop data '''
    def __init__(self, 
                 data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray,],
                 scaler: TransformerMixin): # TODO deprecate scaler
        self.data_tuple = data_tuple
        self.scaler = scaler

        self.comp = self.data_tuple[0]
        self.proc = self.data_tuple[1]
        self.prop = self.data_tuple[2]
        
    def __len__(self):
        return len(self.comp)
    
    def __getitem__(self, idx):
        _comp = self.comp[idx]
        _proc = self.proc[idx]
        _prop = self.prop[idx]
        
        return _comp, _proc, _prop

def get_dataloader(data_tuple, batch_size = 16) -> DataLoader:
    ''' 
        get the dataloader

        input:
            (comp_data, proc_data, prop_data, elem_feature)
            elem_feature: (num_elem_features, num_elements)
    '''
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    dataset = CustomDataset((comp_data, proc_data, prop_data,), None)

    # target elem_feature: (batch_size, 1, number_of_elements, number_of_elemental_features)
    _elem_feature_tensor = torch.tensor(elem_feature, dtype=torch.float32).reshape(1, 1, *(elem_feature.shape))

    def _collate_fn(batch):
        comp, proc, prop = zip(*batch)
        comp = torch.tensor(np.vstack(comp), dtype=torch.float32).reshape(-1, 1, comp_data.shape[-1], 1)
        proc = torch.tensor(np.vstack(proc), dtype=torch.float32).reshape(-1, 1, proc_data.shape[-1], 1)
        prop = torch.tensor(np.vstack(prop), dtype=torch.float32).reshape(-1, 1, prop_data.shape[-1], 1)

        _elem_feature_tensor_clone = _elem_feature_tensor.expand(len(comp), 1, *(elem_feature.shape)).clone().detach()
        _elem_feature_tensor_clone.requires_grad_(False)

        return comp, proc, prop, _elem_feature_tensor_clone

    return DataLoader(dataset, batch_size = batch_size, collate_fn = _collate_fn, shuffle = True)

def train_validate_split(data_tuple, ratio_tuple = (0.8, 0.1, 0.1)):
    ''' 
        split the data into train, validate_1 and validate_2 set
    '''
    _random_seed = next(iter(seeds))
    comp_data, proc_data, prop_data, elem_feature = data_tuple
    _ratio_1 = sum(ratio_tuple[1:]) / sum(ratio_tuple)
    comp_train, comp_tmp, proc_train, proc_tmp, prop_train, prop_tmp = \
        train_test_split(comp_data, proc_data, prop_data, test_size = _ratio_1, random_state = _random_seed)
    _ratio_2 = ratio_tuple[2] / sum(ratio_tuple[1:])
    comp_val_1, comp_val_2, proc_val_1, proc_val_2, prop_val_1, prop_val_2 = \
        train_test_split(comp_tmp, proc_tmp, prop_tmp, test_size = _ratio_2, random_state = _random_seed)
    
    return (comp_train, proc_train, prop_train, elem_feature,), \
            (comp_val_1, proc_val_1, prop_val_1, elem_feature,), \
            (comp_val_2, proc_val_2, prop_val_2, elem_feature,)

def validate(model: CnnDnnModel, data_tuple: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,]) -> float:
    ''' calculate the R2 score of the model on the validate set '''
    model.eval()
    dl = get_dataloader(data_tuple, len(data_tuple[0]))
    comp, proc, prop, elem_t = next(iter(dl))
    out = model(comp, elem_t, proc).detach().numpy()
    prop = prop.reshape(*(out.shape)).detach().numpy()
    return r2_score(prop, out)

def validate_a_model(num_training_epochs = 2000,
                     batch_size = 16,
                     save_path = None):
    ''' util func for training_epoch_num validation '''
    model = CnnDnnModel()
    d = load_data()
    d, scalers = fit_transform(d)

    train_d, val_d_1, val_d_2 = train_validate_split(d, (0.8, 0.1, 0.1))
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    # train one epoch
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for comp, proc, prop, elem_t in dl:
            # forward pass
            out = model(comp, elem_t, proc)
            prop = prop.reshape(*(out.shape))
            l = loss_fn(out, prop)

            # backward pass
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
        # model.eval()
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        val_1_r2 = validate(model, val_d_1)
        val_2_r2 = validate(model, val_d_2)
        epoch_log_buffer.append((epoch, _batch_mean_loss, val_1_r2, val_2_r2))
        if epoch % 5 == 0:
            print(epoch, _batch_mean_loss, val_1_r2, val_2_r2)
    
    if save_path:
        np.savetxt(
            save_path,
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def train_a_model(num_training_epochs = 500,
                     batch_size = 16,
                     save_log = True):
    ''' train a model '''
    model = CnnDnnModel()
    d = load_data()
    d, scalers = fit_transform(d)

    train_d = d
    loss_fn = torch.nn.MSELoss()
    dl = get_dataloader(train_d, batch_size)
    # train one epoch
    epoch_log_buffer = []
    for epoch in range(num_training_epochs):
        model.train()
        _batch_loss_buffer = []
        for comp, proc, prop, elem_t in dl:
            # forward pass
            out = model(comp, elem_t, proc)
            prop = prop.reshape(*(out.shape))
            l = loss_fn(out, prop)

            # backward pass
            model.optimizer.zero_grad()
            l.backward()
            model.optimizer.step()
            
            _batch_loss_buffer.append(l.item())
        
        # model.eval()
        _batch_mean_loss = np.mean(_batch_loss_buffer)
        epoch_log_buffer.append((epoch, _batch_mean_loss))
        if epoch % 25 == 0: 
            print(epoch, _batch_mean_loss)
    
    if save_log:
        np.savetxt(
            'train_err_log.txt',
            np.array(epoch_log_buffer),
            fmt = '%.6f',
            delimiter = '\t',
        )
    
    return model, d, scalers

def get_model(default_model_pth = 'model.pth',
              default_data_pth = 'data.pth',
              resume = False):

    if resume:
        model = CnnDnnModel()
        model.load_state_dict(torch.load(default_model_pth))
        d, scalers = joblib.load(default_data_pth)
    else:
        model, d, scalers = train_a_model()
        torch.save(model.state_dict(), default_model_pth)
        joblib.dump((d, scalers), default_data_pth)
    
    return model, d, scalers

if __name__ == '__main__':
    # get_model('en_model.pth', 'en_data.pth')

    # validate_a_model()
    validate_a_model(num_training_epochs = 1000, save_path='mp_validate_log.txt')