import torch
import numpy as np
from collections import Counter

def knn(data,test,labels,k):
    
    n_test = len(test)
    n_data = len(data)
    k_smallest = torch.Tensor(n_test)
    for i,test_point in enumerate(test):
        distances = torch.Tensor(n_data)
        for j,data_point in enumerate(data):
            distances[j] = torch.linalg.norm(test_point - data_point)
        
        idx = torch.argsort(distances)[:k]
        req_labels = labels[idx]
        k_smallest[i] = torch.tensor(Counter(np.array(req_labels)).most_common(1)[0][0])
        
    return k_smallest  


data =  [
            [ 2.0, 4.0],[ 3.0, 3.0],[ 5.0, 4.0],[ 5.0, 6.0],[ 5.0, 8.0],[ 6.0, 4.0],
            [ 6.0, 7.0],[ 7.0, 3.0],[ 7.0, 4.0],[ 8.0, 2.0],[ 9.0, 4.0],[10.0, 6.0],
            [10.0, 7.0],[10.0, 8.0],[11.0, 5.0],[11.0, 8.0],[12.0, 7.0],[13.0, 6.0],[13.0, 7.0],
            [14.0, 6.0],[15.0, 4.0]
        ]         
    
test = [
    [ 3.0, 4.0],[ 6.0, 5.0],[15.0, 5.0]
]

labels = [1., 1., 2., 2., 2., 2., 2., 2., 2., 2., 2., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]

    
print(knn(torch.tensor(data),torch.tensor(test),torch.tensor(labels),5))
            
    
    
    
    