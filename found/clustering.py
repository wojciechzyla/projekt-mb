import numpy as np
from typing import Any, Tuple
import enum
from clusteval_fix import clusteval
import sklearn.cluster as skl_cluster

@enum.unique
class ClusteringAlgorithm(enum.Enum):
    KMEANS = enum.auto()
    AGGLOMERATIVE = enum.auto()
    SPECTRAL = enum.auto()

cluster_algo_name_to_algo_enum: dict[str, ClusteringAlgorithm] = {
    'kmeans': ClusteringAlgorithm.KMEANS,
    'agglomerative': ClusteringAlgorithm.AGGLOMERATIVE,
    'spectral': ClusteringAlgorithm.SPECTRAL
}

CLUSTER_ALGORITHMS = list(cluster_algo_name_to_algo_enum.keys())

cluster_algo_enum_to_algo_instance: dict[ClusteringAlgorithm, Any] = {
    ClusteringAlgorithm.KMEANS: skl_cluster.KMeans(),
    ClusteringAlgorithm.AGGLOMERATIVE: skl_cluster.AgglomerativeClustering(),
    ClusteringAlgorithm.SPECTRAL: skl_cluster.SpectralClustering()
}

def find_clusters(matrix : np.matrix, algorithm: ClusteringAlgorithm, evaluate : str, k : int, auto_k : bool) -> Tuple[np.ndarray, clusteval]:

    # Verification
    if not matrix.shape[0] == matrix.shape[1]:
        raise Exception('matrix is not quadratic')
    if not check_symmetric(matrix):
        raise Exception('matrix is not symmetric')
    
    # Cluster Evaluation
    ce = clusteval(cluster=algorithm, evaluate=evaluate, min_clust=2 if auto_k else k, max_clust=k + 1) # k values missing
    results = ce.fit(matrix)

    # ce.plot(savefig={'fname': '.\data\clustering\plot.png', 'format': 'png'})
    # ce.scatter(matrix, savefig={'fname': '.\data\clustering\scatter.png', 'format': 'png'})
    # ce.dendrogram()

    return results['labx'], ce

# assuming x has entries greater or equal to zero! 
def distance_matrix_to_affinity_matrix(x: np.matrix) -> np.matrix:
    max_val = x.max()
    res = max_val - x
    return res

cluster_algo_prepare_functions: dict[ClusteringAlgorithm, Any] = {
    ClusteringAlgorithm.SPECTRAL: distance_matrix_to_affinity_matrix
} 

def check_symmetric(m: np.matrix, accuracy: float=1e-8):
    return np.all(np.abs(m-m.T) < accuracy)
