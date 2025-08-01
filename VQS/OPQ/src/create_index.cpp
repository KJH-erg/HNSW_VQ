#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include "../../../third/utils/IO.hpp"
#include "../../../third/hnswlib/hnswlib.h"
#include "../../../third/Eigen/Dense"
#include "../../../third/defines.hpp"
#include <map>
#include <sys/resource.h>
#include <unistd.h>
#include <filesystem>  // C++17
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/index_io.h>
#include <faiss/VectorTransform.h>
namespace fs = std::filesystem;
#include <random>
#include <algorithm>

// float* sample_random_rows(float* data_ptr, size_t N, int DIM, float ratio, int& out_n_sample) {
//     size_t N_sample = std::max(size_t(1), size_t(N * ratio));
//     out_n_sample = static_cast<int>(N_sample);

//     std::vector<size_t> indices(N);
//     std::iota(indices.begin(), indices.end(), 0);

//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::shuffle(indices.begin(), indices.end(), gen);

//     float* sample_data = new float[N_sample * DIM];
//     for (size_t i = 0; i < N_sample; ++i) {
//         size_t idx = indices[i];
//         std::memcpy(&sample_data[i * DIM], &data_ptr[idx * DIM], sizeof(float) * DIM);
//     }

//     return sample_data;
// }



inline size_t getProcessPeakRSS() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return (size_t) rusage.ru_maxrss / 1024L;  // Return in MB
}

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}
std::vector<int> get_proper_divisors(int dims) {
    std::vector<int> divisors;
    for (int i = 2; i <= dims; ++i) {
        if (dims % i == 0 && i >= (dims / 8)) {
            divisors.push_back(i);
        }
    }
    return divisors;
}

int main(int argc, char* argv[]) {
    assert(argc == 7);
    char* DATASET_PATH = argv[1];
    char* DATASET = argv[2];
    int DIM = atoi(argv[3]);
    int M = atoi(argv[4]);
    int ef_construction = atoi(argv[5]);
    int thread_num = atoi(argv[6]);
    char data_file[500];
    sprintf(data_file, "%s/%d/%s_base.fbin", DATASET_PATH, DIM, DATASET);
    FloatRowMat data;
    load_bin<float, FloatRowMat>(data_file, data);
    std::cout << "Data loaded\n";
    std::string filename = "../../logs/build_log.csv";
    bool exists = file_exists(filename);
    std::ofstream summary_file(filename,std::ios::app);
    if (!exists && summary_file.is_open()) {
        std::string header = "dimension,dataset,index_build_time,peak_memory_mb";
        summary_file << header << "\n" << std::flush;
    }
    
    
    float* data_ptr = data.data();
    size_t N = data.rows();
    // int n_train;
    // float* train_data = sample_random_rows(data_ptr, N, DIM, 0.1f, n_train);
    hnswlib::L2Space space(DIM);
    hnswlib::HierarchicalNSW<float> index(&space);

    std::string index_path = "../../indices/float/" + std::to_string(DIM) + "/" + std::string(DATASET) + "_hnsw.bin";
    index.loadIndex(index_path, &space);
    std::vector<int> divisors = get_proper_divisors(DIM);
    
    omp_set_num_threads(thread_num);
    // for (int d : divisors) {
    int d = 64;

        std::string out_index_path = "../../indices/OPQ/" + std::to_string(DIM) + "/" +std::to_string(d)+"_"+ std::string(DATASET) + "_hnsw.bin";
        std::string out_opq_path = "../../indices/OPQ/" + std::to_string(DIM) + "/" +std::to_string(d)+"_"+ std::string(DATASET) + "_opq.bin";
        // std::string out_pq_path = "../../indices/OPQ/" + std::to_string(DIM) + "/" +std::to_string(d)+"_"+ std::string(DATASET) + "_pq.bin";
        fs::path index_dir = fs::path(out_index_path).parent_path();
        if (!fs::exists(index_dir)) {
            fs::create_directories(index_dir);
        }  // creates all missing parent directories
        faiss::ProductQuantizer pq(DIM, d, 8);
        faiss::OPQMatrix opq(DIM, d);
        opq.verbose = true;
        opq.niter = 5; 
        opq.train(N, data_ptr);
        float* rotated = new float[N * DIM];
        rotated = opq.apply(N,data_ptr);
        uint8_t* codes = new uint8_t[N * d];

        pq.compute_codes(rotated, codes, N);

        float* decoded = new float[N * DIM];
        pq.decode(codes, decoded, N);
        faiss::write_VectorTransform(&opq, out_opq_path.c_str());

        #pragma omp parallel for schedule(static)
        for (size_t i=0; i<N; i++) {
            int inner_id = index.label_lookup_[i];
            std::memcpy(index.getDataByInternalId(inner_id), &rotated[i*DIM], sizeof(float) * DIM);
        }
        index.saveIndex(out_index_path);
        std::cout << "✅ Index saved to " << out_index_path << std::endl;
    // }

    return 0;
}
