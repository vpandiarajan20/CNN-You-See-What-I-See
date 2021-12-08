import os
import gym
import numpy as np
import torch


class LinearClassifier(torch.nn.Module):

    def __init__(self, input_dimension):
        super().__init__()
        self.linear_classifier = torch.nn.Linear(input_dimension, 1)
        #self.sigmoid_layer = torch.nn.Sigmoid()

    def forward(self, batch_data):
      logits = self.linear_classifier(batch_data)
      #probs = self.sigmoid_layer(logits)
      return logits

def train_model(model, criterion, optimizer, X_train, y_train, n_epochs=100):
  train_losses = np.zeros(n_epochs)
  test_losses = np.zeros(n_epochs)

  for epoch in range(n_epochs): 
    outputs = model(X_train)
    y_train = y_train.type_as(outputs)


    loss = criterion(torch.squeeze(outputs), y_train)
    loss.backward()
    
    optimizer.step()
    optimizer.zero_grad()
    train_losses[epoch] = loss.item()
    if (epoch + 1) % 20 == 0:
      print(f'In epoch {epoch+1}/{n_epochs}, Training loss: {loss.item():.4f}')

  return train_losses
