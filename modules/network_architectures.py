import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_Net(nn.Module):

  def __init__(self):
      super(Linear_Net, self).__init__()
      
      self.fc1 = nn.Linear(150, 50)
      self.fc2 = nn.Linear(50, 10)
      self.fc3 = nn.Linear(10, 1)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x