import numpy as np

def LDA(D , y):
    n_features = len(D[0])
    n_classes = len(y)

    overall_mean = np.mean(D , axis=0)

    # between-class scatter
    Sb = np.zeros((n_features , n_features))

    # within-class scatter
    S = np.zeros((n_features , n_features))

    for i in range(1 , n_classes + 1):
        Kth_class = D[y == i]
        cur_mean = np.mean(Kth_class , axis=0)
        Sb += np.outer((cur_mean - overall_mean) , (cur_mean - overall_mean)) * (Kth_class.shape[0])
        S += np.dot((Kth_class - cur_mean).T , (Kth_class - cur_mean))

    A = np.linalg.inv(S).dot(Sb)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    eigenvectors = eigenvectors.T
    idxs = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[idxs]

    return eigenvectors[0:39]
