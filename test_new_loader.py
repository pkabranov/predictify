import torch
import pandas
import torchvision
from torch.utils import DataLoader, Dataset

data = pd.read_csv('')
#put everything into a csv
#TO-DO format the data so that the labels are on the first column and the features on the columns following that
#test that the data is formatted correctly and print that out
features = data.iloc[:, 1:]
labels = data.iloc[:, 0]

for i in range(5):
    print('features: {}'.format(features))
    print('labels: {}'.format(labels))

#dataset class
class SongDatasetCSV(Dataset):
    """Song Dataset with extraction from csv"""
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv('file path')
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        features = self.data.iloc[index, 1:] #might have to do to_numpy() here depending if it works initially
        labels = self.data.iloc[index, 0]

        if self.transform is not None:
            features = self.transform(features)

        return features, labels

#test __getitem__, should print (num_rows, 5)
testdataset = SongDatasetCSV('filepath', transform = torchvision.transforms.ToTensor()) #main diff here is that we're chaning to tensor uniformly thorugh ths dataform
feat, lab = testdataset.__getitem__(0)
print('feature shape at the first row: {}'.format(feat.size()))
#should print 'image shape at the first row: torch.Size([1, 5])'

data_loader = DataLoader(testdataset, batch_size = 10, shuffle = True)
test_iter = iter(data_loader)
print(type(test_iter))
#should return that it is a DataLoaderIter

features, labels = test_iter.next()
print('features shape on batch size = {}'.format(featurs.size()))
print('labels shape on batch size = {}'.format(labels.size()))
#should print torch.Size([10, 1, 5])
#should print torch.Size([10])
