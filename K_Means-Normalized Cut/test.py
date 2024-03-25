import numpy as np
import torch

data= torch.tensor([
    [1.0,2.0,3.0],
    [4.0,5.0,6.0]
])

print(torch.exp(-0.1*torch.cdist(data,data)**2))

graph = torch.zeros((len(data), len(data)))
for i in range(len(data)):
    for j in range(len(data)):
        graph[i][j] = torch.exp(-0.1 * torch.norm(data[i] - data[j]) ** 2)

print(graph)