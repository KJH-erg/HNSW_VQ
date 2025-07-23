#define HIGH_ACC_FAST_SCAN
#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include "../../../third/utils/IO.hpp"
#include "../../../third/hnswlib/hnswlib.h"
#include "../../../third/Eigen/Dense"
#include "../../../third/defines.hpp"
#include "../../../third/hnswlib/space_rabitq.h"
#include "../../../third/EXrabitq/Item.hpp"
#include <map>
#include <sys/resource.h>
#include <unistd.h>
#include <filesystem>  // C++17
#include "../../../third/EXrabitq/IVF.hpp"

namespace fs = std::filesystem;


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
    //input configs
    assert(argc == 7);
    char* DATASET_PATH = argv[1];
    char* DATASET = argv[2];
    int DIM = atoi(argv[3]);
    int thread_num = atoi(argv[4]);
    int B = atoi(argv[5]);
    int K = atoi(argv[6]);


    //load files
    char data_file[500];
    char centroids_file[500];
    char cids_file[500];
    char ivf_file[500];

    sprintf(data_file, "%s/%d/%s_base.fbin", DATASET_PATH, DIM, DATASET);
    sprintf(centroids_file, "data/%d/%s_centroid_%d.fvecs", DIM, DATASET, K);
    sprintf(cids_file, "data/%d/%s_cluster_id_%d.ivecs", DIM, DATASET, K);
    

    FloatRowMat data;
    FloatRowMat centroids;
    UintRowMat cids;

    load_bin<float, FloatRowMat>(data_file, data);
    load_vecs<float, FloatRowMat>(centroids_file, centroids);
    load_vecs<PID, UintRowMat>(cids_file, cids);
    std::cout << "Data loaded\n";
    
    //logging part
    std::string filename = "../../logs/search_log.csv";
    bool exists = file_exists(filename);
    std::ofstream summary_file(filename,std::ios::app);
    if (!exists && summary_file.is_open()) {
        std::string header = "dimension,dataset,index_build_time,peak_memory_mb";
        summary_file << header << "\n" << std::flush;
    }

    size_t N = data.rows();
    DIM = data.cols();
    IVF ivf(N, DIM, K, B);

    ivf.construct(data.data(), centroids.data(), cids.data());
    hnswlib::RabitqSpace space(DIM, sizeof(Item));
    hnswlib::L2Space hnsw_space(DIM);

    hnswlib::HierarchicalNSW<float> index(&hnsw_space);

    std::string index_path = "../../indices/float/" + std::to_string(DIM) + "/" + std::string(DATASET) + "_hnsw.bin";
    index.loadIndex(index_path, &hnsw_space);
    index.updateIndex(&space);
    
    omp_set_num_threads(thread_num);
    std::string out_index_path = "../../indices/EXRabitQ/" + std::to_string(DIM) + "/" +std::to_string(B)+"_"+ std::string(DATASET) + "_hnsw.bin";
    fs::path index_dir = fs::path(out_index_path).parent_path();
    if (!fs::exists(index_dir)) {
        fs::create_directories(index_dir);
    }  // creates all missing parent directories
    #pragma omp parallel for schedule(static)
    for (size_t id=0; id<N; id++) {
        int inner_id = index.label_lookup_[id];
        Item tmp_item(id,
            ivf.DQ.id_val_map[id],
            ivf.DQ.nth_compnent_map[id],
            ivf.DQ.total_long_code[id],
            ivf.DQ.total_Exfactor[id],
            ivf.DQ.id_factor_x2_map[id],
            DIM);
        
        std::memcpy(index.getDataByInternalId(inner_id), &tmp_item, sizeof(Item));
    }


    //query part

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



    //query process
    size_t NQ = query.rows();
    FloatRowMat padded_query(NQ, ivf.padded_dim());
    padded_query.setZero();
    FloatRowMat rotated_query(NQ, ivf.padded_dim());
    for (size_t i = 0; i < NQ; ++i) {
        std::memcpy(&padded_query(i, 0), &query(i, 0), sizeof(float) * DIM);
    }
    Rotator& rp = ivf.rotator();
    rp.rotate(padded_query, rotated_query);

    

    std::vector<int> ef_search_values;
    for (int i = 10; i <= 200; i += 10) {
        ef_search_values.push_back(i);
    }

    std::vector<Query*> queries;
    queries.resize(NQ);  // Important!
    int D = static_cast<int>(ivf.padded_dim());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < NQ; ++i) {
        auto* centroid_dist = new std::vector<Candidate>(K);
        ivf.initer->centroids_distances(&rotated_query(i, 0), K, *centroid_dist);
        Query* cur_query = new Query(&rotated_query(i, 0), &cids, D, &ivf, K, *centroid_dist);
        queries[i] = cur_query;
    }
    std::cout << "query loaded" <<std::endl;

    omp_set_num_threads(thread_num);
    for (int ef_search : ef_search_values) {
        index.setEf(ef_search);
        std::vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> all_results(NQ);
        double t_start = omp_get_wtime();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < NQ; i++) {

            std::priority_queue<std::pair<float, hnswlib::labeltype>> result = index.searchKnn(queries[i],1000);
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
        summary_file << "RabitQ," << std::string(DATASET) << "," << DIM << "," << getProcessPeakRSS() << "," << B*DIM/8 << "," << ef_search <<
        "," << qps << "," << n_cmp << ","  << recall_at_1 << "," << recall_at_10 << "," << recall_at_100 << "," << recall_at_1000 << "\n" <<std::flush;
    }
    summary_file.close();
    return 0;
}
