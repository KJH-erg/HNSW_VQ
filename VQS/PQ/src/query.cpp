#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include "../../../third/utils/IO.hpp"
#include "../../../third/hnswlib/hnswlib.h"
#include "../../../third/Eigen/Dense"
#include "../../../third/defines.hpp"
#include <faiss/impl/ScalarQuantizer.h>

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

std::vector<int> get_proper_divisors(int dims) {
    std::vector<int> divisors;
    for (int i = 2; i <= dims; ++i) {
        if (dims % i == 0 && i >= (dims / 8)) {
            divisors.push_back(i);
        }
    }
    return divisors;
}

bool file_exists(const std::string& filename) {
    struct stat buffer;
    return (stat(filename.c_str(), &buffer) == 0);
}


float calculate_recall_at_k(
    const std::vector<std::vector<int>>& gt_vec,                     // ground truth top-k
    const std::vector<std::vector<int>>& all_labels,  // top-k ANN results
    int k
) {
    int NQ = gt_vec.size();
    float total_recall = 0.0f;

    for (int i = 0; i < NQ; ++i) {
        const auto& ground_truth = gt_vec[i];
        const auto& retrieved = all_labels[i];
        int gt_k = std::min(k, static_cast<int>(ground_truth.size()));
        
        int res_k = std::min(k, static_cast<int>(retrieved.size()));

        std::unordered_set<int> gt_top_k(
            ground_truth.begin(), ground_truth.begin() + gt_k);

        int hit = 0;
        for (int j = 0; j < res_k; ++j) {
            if (gt_top_k.count(retrieved[j])) ++hit;
        }
        total_recall += static_cast<float>(hit) / gt_k;
    }
    return total_recall / NQ;
}


int main(int argc, char* argv[]) {
    assert(argc == 5);
    char* DATASET_PATH = argv[1];
    char* DATASET = argv[2];
    int DIM = atoi(argv[3]);
    int thread_num = atoi(argv[4]);


    std::string filename = "../../logs/search_log.csv";
    bool exists = file_exists(filename);
    std::ofstream summary_file(filename,std::ios::app);
    if (!summary_file.is_open()) {
        std::cerr << "Failed to open log file: " << filename << std::endl;
        return 1;  // or exit(1);
    }
    if (!exists && summary_file.is_open()) {
        std::string header = "method,dataset,dim,peak_memory_mb,vec_size,ef_search,qps,n_cmp,recall@1,recall@10,recall@100,recall@1000";
        summary_file << header << "\n" << std::flush;
    }
    char query_file[500];
    char gt_file[500];
    FloatRowMat query;
    UintRowMat gt;
    sprintf(query_file, "%s/%d/%s_query.fbin", DATASET_PATH, DIM, DATASET);
    sprintf(gt_file, "%s/%d/%s_gt.fbin", DATASET_PATH, DIM, DATASET);
    load_bin<float, FloatRowMat>(query_file, query);
    load_bin<u_int32_t, UintRowMat>(gt_file, gt);
    std::vector<std::vector<int>> gt_vec;

    for (int i = 0; i < gt.rows(); ++i) {
        std::vector<int> row;
        for (int j = 0; j < gt.cols(); ++j) {
            row.push_back(gt(i, j));
        }
        gt_vec.push_back(row);
    }
    std::cout << "gt Data loaded" << std::endl;
    std::vector<int> ef_search_values;
    for (int i = 10; i <= 200; i += 10) {
        ef_search_values.push_back(i);
    }
    size_t NQ = query.rows();


    std::vector<int> divisors = get_proper_divisors(DIM);
    omp_set_num_threads(thread_num);
    for (int d : divisors) {
        std::string index_path = "../../indices/PQ/" + std::to_string(DIM) + "/" +std::to_string(d)+"_"+ std::string(DATASET) + "_hnsw.bin";
        std::cout << "index_path" << ":" << index_path << std::endl;
        hnswlib::L2Space space(DIM);
        hnswlib::HierarchicalNSW<float> index(&space, index_path);
        std::cout << "HNSW loaded" << std::endl; 
        for (int ef : ef_search_values) {
            index.setEf(ef);
            std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> all_results(NQ);
            double t_start = omp_get_wtime();
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < NQ; i++) {
                std::priority_queue<std::pair<float, hnswlib::labeltype>> result = index.searchKnn(&query(i,0),1000);
                all_results[i] = std::move(result);
            }
            double t_end = omp_get_wtime();
            double search_time = t_end - t_start;
            
            std::vector<std::vector<int>> all_labels(NQ);
            // std::vector<std::vector<float>> all_dists(NQ);
            for (int i = 0; i < NQ; ++i) {
                auto result = all_results[i];
                while (!result.empty()) {
                    // all_dists[i].insert(all_dists[i].begin(), float(result.top().first));
                    all_labels[i].insert(all_labels[i].begin(), int(result.top().second));
                    result.pop();
                }
            }

            double qps = NQ / search_time;
            float recall_at_1000 = calculate_recall_at_k(gt_vec, all_labels, 1000);
            float recall_at_100 = calculate_recall_at_k(gt_vec, all_labels, 100);
            float recall_at_10 = calculate_recall_at_k(gt_vec, all_labels, 10);
            float recall_at_1 = calculate_recall_at_k(gt_vec, all_labels, 1);
            double n_cmp = (index.metric_distance_computations+index.upper_metric_distance_computations)/NQ;
            index.metric_distance_computations = 0;
            index.upper_metric_distance_computations = 0;
            summary_file << "PQ," << std::string(DATASET) << "," << DIM << "," << getProcessPeakRSS() << "," << d << "," << ef <<
            "," << qps << "," << n_cmp << ","  << recall_at_1 << "," << recall_at_10 << "," << recall_at_100 << "," << recall_at_1000 << "\n" <<std::flush;
        }    
    
    }
    summary_file.close();
    return 0;
}
