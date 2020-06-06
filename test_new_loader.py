import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchvision
data = pd.read_csv('drive/My Drive/predictify/final_data_cleaned_csv.csv', index_col = 0)
#put everything into a csv
#TO-DO format the data so that the labels are on the first column and the features on the columns following that
#test that the data is formatted correctly and print that out
#print(data)

features = data.iloc[:, 1:]
labels = data.iloc[:, 0]

print('labels: \n{}'.format(labels))
print('features:  \n{}'.format(features))
print(type(features))
features = torch.tensor(features.values)
print(features)
labels = torch.tensor(labels.values)
print(labels, type(labels))

#dataset class
class SongDatasetCSV(Dataset):
    """Song Dataset with extraction from csv"""
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file, index_col = 0)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        features = self.data.iloc[index, 1:] #might have to do to_numpy() here depending if it works initially
        print(features)
        labels = self.data.iloc[index, 0]
        features = torch.tensor(features)
        labels = torch.tensor(labels)
        

        if self.transform is not None:
            features = self.transform(features)

        return features, labels

#test __getitem__, should print (num_rows, 5)
testdataset = SongDatasetCSV('drive/My Drive/predictify/final_data_cleaned_csv.csv', transform = None)
feat, lab = testdataset.__getitem__(0)
print(type(feat))
print(type(lab))
print('feature shape at the first row: {}'.format(feat.size()))
#should print 'feature shape at the first row: torch.Size([5])'

data_loader = DataLoader(testdataset, batch_size = 10, shuffle = True)
test_iter = iter(data_loader)
print(type(test_iter))
#should return that it is a DataLoaderIter

features, labels = test_iter.next()
print('features shape on batch size = {}'.format(features.size()))
print('labels shape on batch size = {}'.format(labels.size()))
#should print torch.Size([10, 5])
#should print torch.Size([10])
