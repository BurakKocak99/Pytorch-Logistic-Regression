import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
from six.moves import urllib
import torch
import torch.nn as nn


# Load dataset.
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

dftrain['sex'].replace('female',0,inplace = True)#Female 0
dftrain['sex'].replace('male',1,inplace = True)#Male 1
dfeval['sex'].replace('female',0,inplace = True)#Female 0
dfeval['sex'].replace('male',1,inplace = True)#Male 1

dftrain['class'].replace(['First','Second','Third'],[1,2,3],inplace = True)#First Second Third => 1 2 3
dfeval['class'].replace(['First','Second','Third'],[1,2,3],inplace = True)

dftrain['alone'].replace(['n','y'],[-1,1],inplace = True)
dfeval['alone'].replace(['n','y'],[-1,1],inplace = True)
dftrain.pop('deck')
dftrain.pop('embark_town')
dfeval.pop('deck')
dfeval.pop('embark_town')

t_train = torch.from_numpy(dftrain.values.astype(np.float32))
t_y_train = torch.from_numpy(y_train.values.astype(np.float32))
t_eval = torch.from_numpy(dfeval.values.astype(np.float32))
t_y_eval = torch.from_numpy(y_eval.values.astype(np.float32))

class LogisticRegression(nn.Module):
   def __init__(self,num_of_features):
      super(LogisticRegression,self).__init__()
      self.linear = nn.Linear(num_of_features,1)
   def forward(self,x):
      y_predicted = torch.sigmoid(self.linear(x))
      return y_predicted




model = LogisticRegression(7)

criterion = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=0.002)

t_y_eval = torch.reshape(t_y_eval,(t_y_eval.shape[0],1))
t_y_train = torch.reshape(t_y_train,(627,1))

num_of_epochs = 100000


for epoch in range(num_of_epochs):
   y_predicted = model(t_train)
   loss = criterion(y_predicted,t_y_train)

   loss.backward()
   optimizer.step()

   optimizer.zero_grad()
   if (epoch+1) %10000 == 0:
      print("epoch is: ",epoch+1," loss is: ",loss.item())


with torch.no_grad():
   y_predicted = model(t_eval)
   y_predicted_cls = y_predicted.round()
   acc = y_predicted_cls.eq(t_y_eval).sum() / float(t_y_eval.shape[0])
   print("Accuarcy is: ",acc)
