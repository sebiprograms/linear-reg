import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes as ax
from matplotlib.animation import FuncAnimation


url = './data_for_lr.csv' 
data = pd.read_csv(url)

data = data.dropna()

train_input = np.array(data.x[0:500]).reshape(500,1)
train_output = np.array(data.x[0:500]).reshape(500,1)

test_input = np.array(data.x[500:700]).reshape(199,1)
test_output = np.array(data.y[500:700]).reshape(199,1)

class LinearRegression:
  def __init__(self):
    self.parameters = {}
  
  def forward_propagation(self, train_input):
    m = self.parameters['m']
    c = self.parameters['c']
    predications = np.multiply(m, train_input) + c
    return predications
  
  def cost_function(self, predictions, train_output):
    cost = np.mean((train_output-predictions) ** 2)
    return cost
  
  