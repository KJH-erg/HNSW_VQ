import faiss
import os
import sys
import time
from utils.io import read_fbin, to_fvecs, to_ivecs


K = 4096
meta_path = "../data/"

if __name__ == "__main__":

    SOURCE = sys.argv[1]
    DIMENSION = sys.argv[2]
    DATASET = sys.argv[3]
    faiss.omp_set_num_threads(60)

    print(f"Clustering - {DATASET} ({DIMENSION}D)")

    # Paths
    faiss.omp_set_num_threads(60)
    path = os.path.join(SOURCE, DIMENSION)
    data_path = os.path.join(path, f"{DATASET}_base.fbin")
    X = read_fbin(data_path)

    D = X.shape[1]
    os.makedirs(meta_path + str(DIMENSION), exist_ok=True)
    centroids_path = os.path.join(meta_path + DIMENSION, f"{DATASET}_centroid_{K}.fvecs")
    dist_to_centroid_path = os.path.join(meta_path + DIMENSION, f"{DATASET}_dist_to_centroid_{K}.fvecs")
    cluster_id_path = os.path.join(meta_path + DIMENSION, f"{DATASET}_cluster_id_{K}.ivecs")

    # === Start timing
    t0 = time.time()

    # Clustering
    print("FAISS threads:", faiss.omp_get_max_threads())
    index = faiss.index_factory(D, f"IVF{K},Flat")
    index.verbose = True
    index.train(X)

    build_time = time.time() - t0

    # Post-process
    centroids = index.quantizer.reconstruct_n(0, index.nlist)
    dist_to_centroid, cluster_id = index.quantizer.search(X, 1)
    dist_to_centroid = dist_to_centroid ** 0.5

    to_fvecs(dist_to_centroid_path, dist_to_centroid)
    to_ivecs(cluster_id_path, cluster_id)
    to_fvecs(centroids_path, centroids)

    print(f"✅ Done in {build_time:.2f} sec")

    # === Log build time
    log_file = "./logs/cluster_build_time.csv"
    os.makedirs("./logs", exist_ok=True)

    header = "dataset,dimension,num_clusters,build_time_sec\n"
    log_line = f"{DATASET},{DIMENSION},{K},{build_time:.4f}\n"

    if not os.path.exists(log_file) or os.path.getsize(log_file) == 0:
        with open(log_file, "w") as f:
            f.write(header)
    with open(log_file, "a") as f:
        f.write(log_line)