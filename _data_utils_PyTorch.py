#This assumes the data (song play count linked to their respective features) are all in a .csv file
import pandas as pd
import numpy as np
import torch

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from df_generation import *

#Update so that data is taken from generated DataFrame rather than csv
data = feed_gen()
y = data['play count']
X = data.drop('play count', axis = 1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

#define transforms, change if intending to transform data
transform = None

#custom dataset for song data: len() annd getitem() functions must be overwritten
class SongDataset(Dataset):
    def __init__(self, features, labels = None, transform = None):
        """
        Args:
            features: song feature columns
            labels: play count column
            transform: optional for calling a transformation of data
        """
        self.X = features
        self.y = labels
        self.transform = transform

    def __getitem__(self, idx):
        """
        Load song data as some datatype
        """
        features = self.X.iloc[idx, :] ###
        features = np.asarray(features).astype(np.float32, casting = 'unsafe')

        if self.transform is not None:
            features = self.transform(features)

        if self.y is not None:
            return (features, self.y[idx])

    def __len__(self):
        return len(self.X)


train_data = SongDataset(x_train, x_test, transform)
test_data = SongDataset(y_train, y_test, transform)

#Dataloaders
trainloader = Dataloader(train_data, batch_size = 10, shuffle = True)
testloader = DataLoader(test_data, batch_size = 10, shuffle = True)
