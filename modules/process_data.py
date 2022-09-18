import numpy as np
from tqdm.auto import tqdm

# sklearn libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# to create dataset and dataloaders
from torch.utils.data import DataLoader, Dataset


def process_hr(hr_data, end_time):
  hr_data = hr_data.to_dict()
  hr_ts = hr_data.keys()
  if len(hr_ts) > 1:
    raise Exception('encountered more than 1 timestamps')
  else:
    start_time = 1000*int(float(list(hr_ts)[0]))
    hr_values = list(hr_data.values())[0]
    hr_values = list(hr_values.values())

    hr_ts = np.linspace(start_time, end_time, len(hr_values))
  return {'epoch': hr_ts, 'values': hr_values}


def interpolate(inputs, new_size, inputs_shape=None):
  if inputs_shape is not None:
    n_samples, n_channels, sample_lens = inputs_shape
  else:
    n_samples, n_channels, sample_lens = input.shape
  new_inputs = np.zeros((n_samples,n_channels,new_size))
  for sample in tqdm(range(n_samples), leave=False, desc="resizing"):
    for channel in range(n_channels):
      vals = inputs[sample][channel]
      new_inputs[sample,channel] = np.interp(np.linspace(0,len(vals)-1,new_size),range(len(vals)), vals)
  return new_inputs

def standard_normal(inputs, scalers=None, standardize=True, normalize=True):
  n_samples, n_channels, sample_lens = inputs.shape
  if scalers is None:
    scalers = {"standard" : [], "minmax" : []}
  for c in range(n_channels):

    if standardize:
      if c == len(scalers["standard"]):
        scaler = StandardScaler().fit(inputs[:,c,:])
        scalers["standard"].append(scaler)
      inputs[:,c,:] = scalers["standard"][c].transform(inputs[:,c,:])   # apply standardization before normalization
    
    if normalize:
      if c == len(scalers["minmax"]):
        scaler = MinMaxScaler(feature_range=(-1,1)).fit(inputs[:,c,:])
        scalers["minmax"].append(scaler)    
      inputs[:,c,:] = scalers["minmax"][c].transform(inputs[:,c,:])

  return inputs, scalers



  # create dataset and dataloaders
class mydataset(Dataset):
  def __init__(self, inputs, labels):
    self.inputs = inputs
    self.labels = labels

  def __len__(self):
    return len(self.inputs)

  def __getitem__(self, index):
    input = self.inputs[index]
    label = self.labels[index]
    return input,label

# function to create train, val and test loaders
def createLoaders(train_inputs, train_labels, test_inputs=None, test_labels=None, batch_size=32, val_percent=.25):
  train_inputs, val_inputs, train_labels, val_labels = train_test_split(train_inputs, train_labels, test_size=val_percent, random_state=0)

  train_dataset = mydataset(train_inputs, train_labels)
  train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

  val_dataset = mydataset(val_inputs, val_labels)
  val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

  if test_inputs is not None:
    test_dataset = mydataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
  else:
    test_loader = None

  dataloaders = {"train":train_loader, "val":val_loader, "test":test_loader}
  dataloaders = {k: v for k, v in dataloaders.items() if v is not None}
  return dataloaders