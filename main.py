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
  
  def backward_propagation(self, train_input, train_output, predictions):
    derivatives = {}
    df = (predictions-train_output)
    dm = 2 * np.mean(np.multiply(train_input, df))
    dc = 2 * np.mean(df)
    derivatives['dm'] = dm
    derivatives['dc'] = dc 
    return derivatives
  
  def update_parameters(self, derivatives, learning_rate):
    self.parameters['m'] = self.parameters['m'] - learning_rate * derivatives['dm']
    self.parameters['c'] = self.parameters['c'] - learning_rate * derivatives['dc']

  def train(self, train_input, train_output, learning_rate, iters):
    self.parameters['m'] = np.random.uniform(0,1) * -1
    self.parameters['c'] = np.random.uniform(0,1) * -1

    self.loss = []

    fig, ax = plt.subplots()
    x_vals = np.linspace(min(train_input), max(train_input), 100)
    line, = ax.plot(x_vals, self.parameters['m'] * x_vals + self.parameters['c'], color='r')
    