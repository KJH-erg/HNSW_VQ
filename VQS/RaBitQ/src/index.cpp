#define EIGEN_DONT_PARALLELIZE
#define USE_AVX2
#include <iostream>
#include <cstdio>
#include <fstream>
#include <queue>
#include <getopt.h>
#include <unordered_set>

#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"

using namespace std;
// g++ -o ./bin/index_${data} ./src/index.cpp -I ./src/ -O3 -march=core-avx2 -D BB=${B} -D DIM=${D} -D numC=${C} -D B_QUERY=4 -D SCAN
int main(int argc, char * argv[]) {
    
    
    assert(argc == 4);
    char* DATASET_PATH = argv[1];
    char* DATASET = argv[2];
    int thread_num = atoi(argv[3]);

    //load files
    char data_file[500];
    char centroids_file[500];
    char cids_file[500];
    char ivf_file[500];
    char x0_path[256] = "";
    char dist_to_centroid_path[256] = "";
    char cluster_id_path[256] = "";
    char binary_path[256] = "";


    sprintf(data_file, "%s/%d/%s_base.fbin", DATASET_PATH, DIM, DATASET);
    Matrix<float> X(data_file,true);
    sprintf(centroids_file, "../data/%d/%s_RandCentroid_C%d_B%d.fvecs", DIM, DATASET, numC,BB);
    Matrix<float> C(centroids_file);
    sprintf(x0_path, "../data/%d/%s_x0_C%d_B%d.fvecs", DIM, DATASET, numC, BB);
    Matrix<float> x0(x0_path);
    sprintf(dist_to_centroid_path, "../data/%d/%s_dist_to_centroid_%d.fvecs", DIM, DATASET, numC);
    Matrix<float> dist_to_centroid(dist_to_centroid_path);
    sprintf(cluster_id_path, "../data/%d/%s_cluster_id_%d.ivecs", DIM, DATASET, numC);
    Matrix<uint32_t> cluster_id(cluster_id_path);
    sprintf(binary_path, "../data/%d/%s_RandNet_C%d_B%d.Ivecs", DIM, DATASET, numC,BB);
    Matrix<uint64_t> binary(binary_path);

    sprintf(ivf_file, "../indices/%d/%s.index", DIM, DATASET);


    std::cout << "Data loaded\n";

    IVFRN<DIM, BB> ivf(X, C, dist_to_centroid, x0, cluster_id, binary);

    ivf.save(ivf_file);

    return 0;
}
