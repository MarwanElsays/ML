import numpy as np
import torch

# data= torch.tensor([
#     [1.0,2.0,3.0],
#     [4.0,5.0,6.0]
# ])

# print(torch.exp(-0.1*torch.cdist(data,data)**2))

# graph = torch.zeros((len(data), len(data)))
# for i in range(len(data)):
#     for j in range(len(data)):
#         graph[i][j] = torch.exp(-0.1 * torch.norm(data[i] - data[j]) ** 2)

# print(graph)

# distances = torch.tensor([
#     [1,2,3,4],
#     [5,6,7,8],
#     [2,3,4,5],
#     [1,2,3,4],
# ])

# cluster1 = torch.tensor([[1]]).flatten()
# cluster2 = torch.tensor([[0],[2]]).flatten()
# print(cluster1,cluster2)
# grouped_cluster = torch.flatten(torch.cat((cluster1, cluster2)))
# print(grouped_cluster)
# broadcasted_rows_indices = grouped_cluster.unsqueeze(1).expand(-1, grouped_cluster.size(0))
# print(broadcasted_rows_indices)

# distances[broadcasted_rows_indices, grouped_cluster] = 0
# print(distances)

cluster2 = torch.tensor([0,1,2,3,4,5,6])
 
val = cluster2[0].item()
cluster2[torch.tensor([1,2,3])] = val

print(cluster2)