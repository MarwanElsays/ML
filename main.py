import torch


if __name__ == "__main__":

    # Assuming tensor1 is a tensor of size (10, 1)
    tensor1 = torch.randn(5, 1)
    print("Tensor 1 (size 5x1):")
    print(tensor1)

    # Assuming tensor2 is the transpose of tensor1
    tensor2 = tensor1.T
    print("\nTensor 2 (size 1x5), which is the transpose of Tensor 1:")
    print(tensor2)

    # Perform matrix multiplication
    result = torch.matmul(tensor1, tensor2)
    print("\nResult of matrix multiplication (size 5x5):")
    print(result)
