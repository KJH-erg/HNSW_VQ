#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include "../../third/utils/IO.hpp"
#include "../../third/hnswlib/hnswlib.h"
#include "../../third/Eigen/Dense"
#include "../../third/defines.hpp"
#include <map>
#include <sys/resource.h>
#include <unistd.h>
#include <filesystem>  // C++17
namespace fs = std::filesystem;


inline size_t getProcessPeakRSS() {
  struct rusage rusage;
  getrusage(RUSAGE_SELF, &rusage);
  return (size_t) rusage.ru_maxrss / 1024L;  // Return in MB
}

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
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
    std::string filename = "../logs/build_log.csv";
    bool exists = file_exists(filename);
    std::ofstream summary_file(filename,std::ios::app);
    if (!exists && summary_file.is_open()) {
        std::string header = "dimension,dataset,index_build_time,peak_memory_mb";
        summary_file << header << "\n" << std::flush;
    }
    
    float* data_ptr = data.data();
    size_t N = data.rows();
    
    
    omp_set_num_threads(thread_num);

    std::string index_path = "../indices/float/" + std::to_string(DIM) + "/" + std::string(DATASET) + "_hnsw.bin";
    fs::path index_dir = fs::path(index_path).parent_path();
    if (!fs::exists(index_dir)) {
        fs::create_directories(index_dir);  // creates all missing parent directories
    }

    hnswlib::L2Space space(DIM);
    hnswlib::HierarchicalNSW<float> index(&space, N, M, ef_construction);

    double t_start = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        index.addPoint(data_ptr + i * DIM, i);
    }
    double t_end = omp_get_wtime();
    double build_time = t_end - t_start;

    index.saveIndex(index_path);
    std::cout << "✅ Index saved to " << index_path << std::endl;


    if (summary_file.is_open()) {
        summary_file << DIM << "," << DATASET << "," << build_time << "," << getProcessPeakRSS() << "\n";
    }
    summary_file.close();
    return 0;
}
