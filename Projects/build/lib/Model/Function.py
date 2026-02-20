import numpy as np

class Sigmoid:
  def __call__(self, x):
    x = np.clip(x, -250, 250)
    return 1/(1+np.exp(-x))
  
  def derivative(self, x):
    s = self(x)
    return s*(1-s)
  
class TanH:
  def __call__(self, x):
    return np.tanh(x)

  def derivative(self, x):
    t = self(x)
    return 1-t**2
  
class Softmax:
  def __call__(self, x):
    exp_x = np.exp(x-np.max(x))
    return exp_x/np.sum(exp_x, axis=0)

  def derivative(self, x):
    pass