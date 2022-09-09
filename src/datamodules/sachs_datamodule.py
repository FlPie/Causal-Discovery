import os
from tqdm import tqdm

import pandas as pd
import numpy as np


import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl

import torch_geometric as pyg

class SachsDataset(pyg.data.InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['sachs.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        raise Exception("We are not downloading sachs.csv")

    def process(self):
        # Read data into huge `Data` list.
        df = pd.read_csv(self.raw_paths[0], index_col=1).reset_index()
        data_list = []
        for index, feature in tqdm(df.iterrows(), total=df.shape[0]):
            x = torch.tensor(feature.values, dtype=torch.float)
            #adj = torch.zeros([len(feature), len(feature)], dtype=torch.float)
            #data = DenseData(x = x, adj = adj)
            data = pyg.data.Data(x=x)
            #data = Data(x = x, edge_index=torch.zeros([2, 1]), adj=adj)
            data_list.append(data)
        

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class SachsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        **kwargs
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.data_train = None

    
    def load_data(self, root, transform=None, pre_transform=None, pre_filter=None):
        return SachsDataset(root)

    
    def setup(self, stage=None):
        if not self.data_train:
            self.data_train = self.load_data(root=self.hparams.data)


    def train_dataloader(self):
        return pyg.loader.DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            drop_last=True)


    def val_dataloader(self):
        return pyg.loader.DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            shuffle=self.hparams.shuffle,
            drop_last=True)