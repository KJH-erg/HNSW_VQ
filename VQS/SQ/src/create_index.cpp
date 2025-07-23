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
#include <faiss/impl/ScalarQuantizer.h>
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
    float* data_ptr = data.data();
    size_t N = data.rows();
    std::string filename = "../../logs/build_log.csv";
    bool exists = file_exists(filename);
    std::ofstream summary_file(filename,std::ios::app);
    if (!exists && summary_file.is_open()) {
        std::string header = "dimension,dataset,index_build_time,peak_memory_mb";
        summary_file << header << "\n" << std::flush;
    }

    
   

    
    std::map<faiss::ScalarQuantizer::QuantizerType, std::string> quantizer_names = {
        {faiss::ScalarQuantizer::QuantizerType::QT_4bit, "4"},
        {faiss::ScalarQuantizer::QuantizerType::QT_6bit, "6"},
        {faiss::ScalarQuantizer::QuantizerType::QT_8bit, "8"},
        {faiss::ScalarQuantizer::QuantizerType::QT_fp16, "16"}
    };
    std::vector<faiss::ScalarQuantizer::QuantizerType> quantizer_types = {
        faiss::ScalarQuantizer::QuantizerType::QT_4bit,
        faiss::ScalarQuantizer::QuantizerType::QT_6bit,
        faiss::ScalarQuantizer::QuantizerType::QT_8bit,
        faiss::ScalarQuantizer::QuantizerType::QT_fp16
    };
    
    omp_set_num_threads(thread_num);
    for (faiss::ScalarQuantizer::QuantizerType qtype : quantizer_types) {
        hnswlib::L2Space space(DIM);
        hnswlib::HierarchicalNSW<float> index(&space);
        std::string index_path = "../../indices/float/" + std::to_string(DIM) + "/" + std::string(DATASET) + "_hnsw.bin";
        index.loadIndex(index_path, &space);
        std::cout << "Original HNSW loaded" << std::endl;
        std::string label = quantizer_names[qtype];

        std::string out_index_path = "../../indices/SQ/" + std::to_string(DIM) + "/" +label+"_"+ std::string(DATASET) + "_hnsw.bin";
        fs::path index_dir = fs::path(out_index_path).parent_path();

        if (!fs::exists(index_dir)) {
            fs::create_directories(index_dir);
        }  // creates all missing parent directories
        faiss::ScalarQuantizer sq(DIM, qtype);

        
        sq.train(N, data_ptr);
        std::vector<uint8_t> codes(N * sq.code_size);
        sq.compute_codes(data_ptr, codes.data(), N);
        std::vector<float> decoded(N * DIM);
        sq.decode(codes.data(), decoded.data(), N);
    
        #pragma omp parallel for schedule(static)
        for (size_t i=0; i<N; i++) {
            int inner_id = index.label_lookup_[i];
            std::memcpy(index.getDataByInternalId(inner_id), &decoded[i*DIM], sizeof(float) * DIM);
        }
        index.saveIndex(out_index_path);
        std::cout << "✅ Index saved to " << out_index_path << std::endl;
    }

    return 0;
}
