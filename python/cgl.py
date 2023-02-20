import numpy as np
from scipy.linalg import eigh as largest_eigh
import scipy.io as scio
from scipy import sparse
from sklearn import cluster
from sklearn.preprocessing import normalize
from sklearn.neighbors import kneighbors_graph
from metrics import cal_clustering_acc


def prox_weight_tensor_nuclear_norm(Y, C):
    # calculate the weighted tensor nuclear norm
    # min_X ||X||_w* + 0.5||X - Y||_F^2
    n1, n2, n3 = np.shape(Y)
    X = np.zeros((n1, n2, n3), dtype=complex)
    # Y = np.fft.fft(Y, n3)
    Y = np.fft.fftn(Y)
    # Y = np.fft.fftn(Y, s=[n1, n2, n3])
    eps = 1e-6
    for i in range(n3):
        U, S, V = np.linalg.svd(Y[:, :, i], full_matrices=False)
        temp = np.power(S - eps, 2) - 4 * (C - eps * S)
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
    # newX = np.fft.ifftn(X, s=[n1, n2, n3])
    # newX = np.fft.ifft(X, n3)

    return np.real(newX)


def cal_knn_graph(distance, neighbor_num):
    # construct a knn graph
    neighbors_graph = kneighbors_graph(
        distance, neighbor_num, mode='connectivity', include_self=False)
    W = 0.5 * (neighbors_graph + neighbors_graph.T)
    return W


def consensus_graph_learning(A, cluster_num, lambda_1, rho, iteration_num):
    # optimize the consensus graph learning problem
    # min_H, Z 0.5||A - H'H||_F^2 + 0.5||Z - hatHhatH'||_F^2 + ||Z||_w*
    # s.t. H'H = I_k
    sample_num, sample_num, view_num = np.shape(A)
    # initial variables
    H = np.zeros((sample_num, cluster_num, view_num))
    HH = np.zeros((sample_num, sample_num, view_num))
    hatH = np.zeros((sample_num, cluster_num, view_num))
    hatHH = np.zeros((sample_num, sample_num, view_num))
    Q = np.zeros((sample_num, sample_num, view_num))
    Z = np.zeros((sample_num, sample_num, view_num))
    obj = np.zeros((iteration_num, 1))
    # loop
    for iter in range(iteration_num):
        # update H
        temp = np.zeros((sample_num, sample_num, view_num))
        G = np.zeros((sample_num, sample_num, view_num))
        for view in range(view_num):
            temp[:, :, view] = np.dot(
                np.dot(Q[:, :, view], 0.5 * (Z[:, :, view] + Z[:, :, view].T) - 0.5 * hatHH[:, :, view])
                , Q[:, :, view]
            )
            G[:, :, view] = lambda_1 * A[:, :, view] + temp[:, :, view]
            _, H[:, :, view] = largest_eigh(
                G[:, :, view], subset_by_index=[sample_num - cluster_num, sample_num - 1]
            )
            HH[:, :, view] = np.dot(H[:, :, view], H[:, :, view].T)
            Q[:, :, view] = np.diag(1 / np.sqrt(np.diag(HH[:, :, view])))
            hatH[:, :, view] = np.dot(Q[:, :, view], H[:, :, view])
            hatHH[:, :, view] = np.dot(hatH[:, :, view], hatH[:, :, view].T)
        # update Z
        hatHH2 = hatHH.transpose((0, 2, 1))
        Z2 = prox_weight_tensor_nuclear_norm(hatHH2, rho)
        Z = Z2.transpose((0, 2, 1))
        # update obj
        f = np.zeros((view_num, 1))
        for view in range(view_num):
            f[view] = 0.5 * lambda_1 * np.linalg.norm(A[:, :, view] - HH[:, :, view], ord='fro') + \
                      0.5 * np.linalg.norm(Z[:, :, view] - hatHH[:, :, view], ord='fro')
        obj[iter] = np.sum(f)

    # construct knn graph
    distance = np.zeros((sample_num, sample_num))
    for view in range(view_num):
        distance += hatHH[:, :, view]
    W = cal_knn_graph(1 - distance, 15)
    # perform spectral clustering
    laplacian = sparse.csgraph.laplacian(W, normed=True)
    _, vec = sparse.linalg.eigsh(sparse.identity(
        laplacian.shape[0]) - laplacian, cluster_num, sigma=None, which='LA')
    embedding = normalize(vec)
    est = cluster.KMeans(n_clusters=cluster_num, n_init="auto").fit(embedding)
    # reture results
    return W, est.labels_


if __name__ == '__main__':
    precomputed = 1
    if precomputed != 1:
        # precomputed graph with the neighbor graph learning method
        data_matrix = scio.loadmat('MSRCV1_A.mat')
        ground_truth = data_matrix['Y']
        A = data_matrix['A']
    else:
        # knn graph via sklearn.neighbors.kneighbors_graph
        data_matrix = scio.loadmat('MSRCV1_X.mat')
        ground_truth = data_matrix['Y']
        features = data_matrix['X']
        # construct view_specific knn graph
        view_num = features.shape[1]
        sample_num, _ = features[0][0].shape
        A = np.zeros((sample_num, sample_num, view_num))
        for view in range(view_num):
            knn_graph = cal_knn_graph(features[0][view], neighbor_num=15)
            S = sparse.identity(knn_graph.shape[0]) - sparse.csgraph.laplacian(knn_graph, normed=True).toarray()
            A[:, :, view] = S

    ground_truth = ground_truth.reshape(np.max(ground_truth.shape), )
    cluster_num = 7
    parameter_lambda = [1, 5, 10, 50, 100, 500, 1000, 5000]
    parameter_rho = [1, 5, 10, 50, 100, 500, 1000, 5000]
    ACC = np.zeros((8, 8))
    for i in range(8):
        for j in range(8):
            W, predict_label = consensus_graph_learning(A, cluster_num, parameter_lambda[i], parameter_rho[j], 100)
            predict_label = predict_label + 1
            ACC[i, j] = cal_clustering_acc(ground_truth, predict_label)
            print('clustering accuracy: {}, lambda: {}, rho: {}'.format(
                ACC[i, j], parameter_lambda[i], parameter_rho[j]))
    print('max clustering accuracy: {}'.format(ACC.max()))
