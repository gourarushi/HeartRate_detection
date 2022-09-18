import numpy as np
import matplotlib.pyplot as plt

# for neural network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy

# for evaluation
from sklearn.metrics import classification_report

# to track progress
from tqdm.auto import tqdm

torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def setupModel(model):
  model = model.to(device)
  print("params:",sum(p.numel() for p in model.parameters() if p.requires_grad))

  #Define a Loss function and optimizer
  criterion = torch.nn.MSELoss()
  optimizer = optim.Adam(model.parameters(), lr=1e-3)
  return model, criterion, optimizer


def train_model(model,criterion,optimizer,dataloaders,epochs,check_every=None,earlyStopping=False):

  print("training model")
  optimizer.zero_grad()

  if not check_every:
    check_every = int(epochs / 10) if epochs > 10 else 1

  phases = dataloaders.keys()
  valExists = True if "val" in phases else False
  avg_loss = {phase:0 for phase in phases}
  avg_losses = {phase:[] for phase in phases}

  for epoch in tqdm(range(epochs), desc="epoch"):  # loop over the dataset multiple times

    batchLoss = {phase:[] for phase in phases}

     # Each epoch has a training and validation phase
    for phase in phases:
      if phase == "train":  model.train()  # Set model to training mode
      else: model.eval()   # Set model to evaluate mode

      for i, (inputBatch,outTrueBatch) in enumerate(tqdm(dataloaders[phase], leave=False)):

        inputBatch = inputBatch.to(device).float()
        outTrueBatch = outTrueBatch.to(device).float()

        # forward
        with torch.set_grad_enabled(not phase=="val"):
          outPredBatch = model(inputBatch)
        loss = criterion(outPredBatch, outTrueBatch)
        batchLoss[phase].append(loss.item())

        # backward + optimize only if in training phase
        if phase == "train":
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()


    for phase in phases : avg_loss[phase] = np.mean(batchLoss[phase])

    phase = "val" if valExists else "train"
    if epoch > 0:
      if avg_loss[phase] < min(avg_losses[phase]):
        best_params = deepcopy(model.state_dict())
        best_epoch, best_loss = epoch, avg_loss[phase]
    else:
      best_params = deepcopy(model.state_dict())
      best_epoch, best_loss = epoch, avg_loss[phase]
      movAvg_old = avg_loss[phase]

    for phase in phases : avg_losses[phase].append(avg_loss[phase])

    # print statistics
    if epoch % check_every == check_every - 1:
      print("epoch: %d" % (epoch + 1), end="  | ")
      for phase in phases:
        print("%s loss: %.4f" % (phase, avg_loss[phase]), end=", ")
      if check_every > 1:   # else avg loss would be same as loss so no need to print
        print(" | ", end='')
        for phase in phases:
          print("avg %s loss: %.4f" % (phase, np.mean(avg_losses[phase][epoch+1-check_every:epoch+1])), end=", ")
      if valExists:
        movAvg_new = np.mean(avg_losses["val"][epoch+1-check_every:epoch+1])

      if (valExists) and earlyStopping:
        if movAvg_old < movAvg_new:
          print("\nStopping Early");  break
        else:   movAvg_old = movAvg_new



  last_params = deepcopy(model.state_dict())
  print('Finished Training')
  for phase in phases:  plt.plot(avg_losses[phase], label=phase+" loss")
  #plt.plot([best_loss]*epoch, linestyle='dashed')
  plt.plot(best_epoch, best_loss, 'o')
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.legend()
  plt.show()

  return best_params, last_params