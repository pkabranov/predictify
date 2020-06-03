import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from tqdm import tqdm_notebook as tqdm
from torch import utils

EPOCHS = 1000
BATCH_SIZE = 50
LEARNING_RATE = 1e-5
NUM_FEATURES = len(x.columns) #length of any feature df

class Predictify_Net(nn.Module):
    def __init__(self, num_features):
        super(Predictify_Net, self).__init__()
        self.hidden1 = nn.Linear(num_features, 16)
        self.hidden2 = nn.Linear(16, 32)
        self.hidden3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 1)
        self.activation = nn.LeakyReLU()

    def forward(self, inputs):
        x = self.activation(self.hidden1(inputs))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.layer_out(x)
        return x

    def predict(self, test_inputs):
        x = self.activation(self.hidden1(inputs))
        x = self.activation(self.hidden2(x))
        x = self.activation(self.hidden3(x))
        x = self.layer_out(x)
        return x

model = Predictify_Net(NUM_FEATURES)

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))

loss_function = RMSLELoss()
optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

def run():
  for epoch in tqdm(range(1, EPOCHS + 1)):
      #Training
      train_epoch_loss = 0
      model.train()
      for x_train, y_train in trainloader:
        #Clear gradients
        model.zero_grad()

        #Make vector and wrap target in a Tensor
        y_train_pred = model(x_train.float()).double()
        train_loss = loss_function(y_train_pred, y_train.unsqueeze(1))
        train_loss.backward()
        optimizer.step()
        train_epoch_loss += train_loss.item()

      #Validation
      with torch.no_grad():
        val_epoch_loss = 0
        model.eval()
        for x_val, y_val in valloader:
            y_val_pred = model(x_val.float())
            val_loss = loss_function(y_val_pred, y_val.unsqueeze(1))
            val_epoch_loss += val_loss.item()

      #loss_stats['train'].append(train_epoch_loss/len(trainloader))
      #loss_stats['val'].append(val_epoch_loss/len(valloader))

      if (epoch % 100 == 0):
        print(f'Epoch {epoch}: | Train Loss: {train_epoch_loss/len(trainloader):.5f} | Val Loss: {val_epoch_loss/len(valloader):.5f}')
