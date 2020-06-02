#This assumes the data (song play count linked to their respective features) are all in a .csv file
import pandas as pd
import numpy as np
import torch

from torchvision.transforms import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#.csv files must be present in the curernt working directory
train, test = train_test_split("complete_data.csv", test_size = 0.2)
train.to_csv("train.csv"), test.to_csv("test.csv")

train_dataset = Song_dataset(csv_file = "train.csv")
test_dataset = Song_dataset(csv_file = "test.csv")

df_train = pd.read_csv(train_dataset)
df_test = pd.read_csv(test_dataset)

#Assuming the play count is stored on the first column of the csv and its respective features following, change as necessary
train_labels = df_train.iloc[:, 0]
train_features = df_train.iloc[:, 1:]
test_labels = df_test.iloc[:, 0]
test_features = df_test.iloc[:, 1:]

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


train_data = SongDataset(train_features, train_labels, transform)
test_data = SongDataset(test_features, test_labels, transform)

#Dataloaders
trainloader = Dataloader(train_data, batch_size = 10, shuffle = True)
testloader = DataLoader(test_data, batch_size = 10, shuffle = True)
