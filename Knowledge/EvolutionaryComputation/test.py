import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from Projects.Model.EC import *
# from Projects.Utils.EC_tools import *
from Projects.Utils.tools import *

device = utils_getDevice()

class Problem(EC_Problem):
  def fitness(self, x):
    return x * np.sin(10 * np.pi * x) + 2
  
  def decode(self, chorm, x_min=-1, x_max=2, n_bits=16):
    return super().decode(chorm, x_min, x_max, n_bits)
  
problem = Problem()

ga = GA(problem=problem)
ga(50)
