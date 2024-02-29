from PIL import Image
import torch
from torch import Tensor
import numpy as np


def power_method(a: Tensor, tol: float, max_iterations: int) -> tuple[float, Tensor]:
    x = torch.ones((len(a), 1))
    k = 1.0

    error = 100.0
    iterations = 0
    while error > tol and iterations < max_iterations:
        iterations += 1

        w = a @ x
        k = ((w.T @ x) / (x.T @ x))[0]

        error = torch.norm(w / torch.norm(w) - x)

        x = w / torch.norm(w)

    return (k.numpy()[0], x)


def largest_covariance_eigenvalues(covariance: Tensor, variance: float, alpha: float):
    eigen_values = []
    eigen_vectors = Tensor(torch.Size((covariance.shape[1], 0)))
    sum = 0
    while sum < variance * alpha:
        eigenvalue, eigenvector = power_method(covariance, 1e-6, 1000)
        eigen_values.append(eigenvalue)
        eigen_vectors = torch.hstack((eigen_vectors, eigenvector))
        covariance -= eigenvalue * eigenvector * eigenvector.T
        sum += eigenvalue

    return eigen_values, eigen_vectors


def pca(data: Tensor, alpha: float) -> Tensor:
    centered_data: Tensor = data - data.mean(0)
    covariance: Tensor = 1 / len(data) * centered_data.T @ centered_data
    variance: float = covariance.trace().numpy()
    eigen_values, eigen_vectors = largest_covariance_eigenvalues(covariance, variance, alpha)
    return data @ eigen_vectors


def main():

    pca(a, 0.8)


if __name__ == "__main__":
    main()
