from functions_final import DQN
import numpy as np
import torch

import torch.nn as nn
import torch.nn.functional as F

model = DQN(4,2)
criterion = torch.nn.MSELoss()
opt = torch.optim.Adam(params = model.parameters(), lr=0.01)
a = np.array([1,2,3,4])
a = torch.tensor(a, dtype=torch.float32)
b = []
b.append(a)
b.append(a)
print(torch.tensor(b))