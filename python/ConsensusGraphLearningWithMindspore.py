import numpy as np
from mindspore.ops import operations as ops
from mindspore import Tensor
from mindspore import dtype as mstype
from scipy.linalg import eigh as largest_eigh
from scipy import sparse
import scipy.io as scio
from scipy import sparse
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import accuracy_score
from sklearn import metrics



def acc(y_true, y_pred):
    # Calculate clustering accuracy
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):    
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def prox_wtnn(Y, C):
    # weighted tensor nuclear norm
    # min_X ||X||_w* + 0.5||X - Y||_F^2
    n1, n2, n3 = np.shape(Y)
    X = np.zeros((n1, n2, n3), dtype=complex)
    #Y = np.fft.fft(Y, n3)
    Y = np.fft.fftn(Y)
    #Y = np.fft.fftn(Y, s=[n1, n2, n3])
    eps = 1e-6
    for i in range(n3):
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        temp = np.power(S - eps, 2) - 4*(C - eps*S)
        ind = np.where(temp > 0)
        ind = np.array(ind)
        r = np.max(ind.shape)
        if np.min(ind.shape) == 0:
            r = 0
        if r >= 1:
            temp2 = S[ind] - eps + np.sqrt(temp[ind])
            S = temp2.reshape(temp2.size, )
            X[:, :, i] = np.dot(np.dot(U[:, 0:r], np.diag(S)), V[:, 0:r].T)
    newX = np.fft.ifftn(X)
    #newX = np.fft.ifftn(X, s=[n1, n2, n3])
    #newX = np.fft.ifft(X, n3)
    return np.real(newX)


def constructW(Dist, n_neighbors):
    # construct a knn graph
    neighbors_graph = kneighbors_graph(
        Dist, n_neighbors, mode='connectivity', include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)
    return W


def cgl(A, n_cluster, lambda_1, rho, n_iter):
    # consensus graph learning
    # min_H, Z 0.5||A - H'H||_F^2 + 0.5||Z - hatHhatH'||_F^2 + ||Z||_w*
    # s.t. H'H = I_k
    n_sample, n_sample, n_view = A.shape
    matmul = ops.MatMul()
    diag = ops.Diag()
    sqrt = ops.Sqrt()
    transpose = ops.Transpose()
    # matrix initial
    H = np.zeros((n_sample, n_cluster, n_view))
    HH = np.zeros((n_sample, n_sample, n_view))
    hatH = np.zeros((n_sample, n_cluster, n_view))  
    hatHH = np.zeros((n_sample, n_sample, n_view))
    Q = np.zeros((n_sample, n_sample, n_view))
    Z = np.zeros((n_sample, n_sample, n_view))
    obj = np.zeros((n_iter, 1))
    # repeat
    for iter in range(n_iter):
        # 
        H = Tensor(H, mstype.float32)
        HH = Tensor(HH, mstype.float32)
        hatH = Tensor(hatH, mstype.float32)
        hatHH = Tensor(hatHH, mstype.float32)
        Q = Tensor(Q, mstype.float32)
        Z = Tensor(Z, mstype.float32)
        # update H
        temp = np.zeros((n_sample, n_sample, n_view))
        temp = Tensor(temp, mstype.float32)
        G = np.zeros((n_sample, n_sample, n_view))
        G = Tensor(G, mstype.float32)
        for view in range(n_view):
            temp[:, :, view] = matmul(matmul(
                Q[:, :, view], 0.5*(Z[:, :, view] + transpose(Z[:, :, view])) - 0.5*hatHH[:, :, view]), Q[:, :, view])
            temp[:, :, view] = temp[:, :, view].asnumpy()
            G[:, :, view] = lambda_1*A[:, :, view] + temp[:, :, view]
            _, H[:, :, view] = largest_eigh(
                G[:, :, view], eigvals=[n_sample-n_cluster, n_sample-1])
            H[:, :, view] = Tensor(H[:, :, view])
            HH[:, :, view] = matmul(H[:, :, view], transpose(H[:, :, view]))
            Q[:, :, view] = diag(1/np.sqrt(diag(HH[:, :, view])))
            hatH[:, :, view] = matmul(Q[:, :, view], H[:, :, view])
            hatHH[:, :, view] = matmul(hatH[:, :, view], transpose(hatH[:, :, view]))
        # 
        H = H.asnumpy()
        HH = HH.asnumpy()
        hatH = hatH.asnumpy()
        hatHH = hatHH.asnumpy()
        Q = Q.asnumpy()
        Z = Z.asnumpy()
        # update Z
        hatHH2 = hatHH.transpose((0, 2, 1))
        Z2 = prox_wtnn(hatHH2, rho)
        Z = Z2.transpose((0, 2, 1))
        # update obj
        f = np.zeros((n_view, 1))
        for view in range(n_view):
            f[view] = 0.5*lambda_1 * np.linalg.norm(A[:, :, view] - HH[:, :, view], ord='fro') + \
                0.5 * \
                np.linalg.norm(
                Z[:, :, view] - hatHH[:, :, view],  ord='fro')
        obj[iter] = np.sum(f)

    # compute similarity matrix
    Dist = np.zeros((n_sample, n_sample))
    for view in range(n_view):
        Dist += hatHH[:, :, view]
    W = constructW(1 - Dist, 15)
    # apply spectral clustering
    laplacian = sparse.csgraph.laplacian(W, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(
        laplacian.shape[0]) - laplacian, n_cluster, sigma=None, which='LA')
    embedding = normalize(vec)
    est = cluster.KMeans(n_clusters=n_cluster).fit(embedding)
    # reture results
    return W, est.labels_


if __name__ == '__main__':
    data = scio.loadmat('MSRCV1.mat')
    GT = data['Y']
    GT = GT.reshape(np.max(GT.shape), )
    n_cluster = 7
    Plambda = [1, 5, 10, 50, 100, 500, 1000, 5000]
    Prho = [1, 5, 10, 50, 100, 500, 1000, 5000]
    ACC = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            W, label = cgl(data['A'], n_cluster, Plambda[i], Prho[j], 100)
            label = label + 1
            ACC[i, j] = acc(GT, label)
            print('clustering accuracy: {}, lambda: {}, rho: {}'.format(
                ACC[i, j], Plambda[i], Prho[j]))
    print('max clustering accuracy: {}'.format(ACC.max()))
