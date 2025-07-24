#define HIGH_ACC_FAST_SCAN
#include <omp.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <iostream>
#include "../../../third/utils/IO.hpp"
#include "../../../third/single_rabitq_hnswlib/hnswlib.h"
#include "../../../third/Eigen/Dense"
#include "../../../third/defines.hpp"
// #include "../../../third/hnswlib/space_rabitq.h"
// #include "../../../third/EXrabitq/Item.hpp"
#include <map>
#include <sys/resource.h>
#include <unistd.h>
#include <filesystem>  // C++17
#include "matrix.h"
#include "utils.h"
#include "ivf_rabitq.h"
#include "Item.hpp"

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
    assert(argc == 4);
    char* DATASET_PATH = argv[1];
    char* DATASET = argv[2];
    int thread_num = atoi(argv[3]);


    //load files


    std::cout << "Data loaded\n";
    
    //logging part
    std::string filename = "../../../logs/search_log.csv";
    bool exists = file_exists(filename);
    std::ofstream summary_file(filename,std::ios::app);
    if (!exists && summary_file.is_open()) {
        std::string header = "dimension,dataset,index_build_time,peak_memory_mb";
        summary_file << header << "\n" << std::flush;
    }

    char cids_file[500];
    sprintf(cids_file, "../data/%d/%s_cluster_id_%d.ivecs", DIMENSION, DATASET, numC);
    UintRowMat cids;
    load_vecs<PID, UintRowMat>(cids_file, cids);

    char ivf_path[256] = "";
    sprintf(ivf_path, "../indices/%d/%s.index", DIMENSION, DATASET);
    IVFRN<DIMENSION, BB> ivf;
    ivf.load(ivf_path);

    
    hnswlib::L2Space hnsw_space(DIMENSION);

    hnswlib::HierarchicalNSW<float> index(&hnsw_space);
    std::string index_path = "../../../indices/float/" + std::to_string(DIMENSION) + "/" + std::string(DATASET) + "_hnsw.bin";
    
    index.loadIndex(index_path, &hnsw_space);

    hnswlib::RabitqSpace space(DIMENSION, sizeof(Item));
    index.updateIndex(&space);


    for(int i=0; i<ivf.N; i++){
        int x = ivf.id[i];
    
        int inner_id = index.label_lookup_[x];
        Item tmp_item(x,
            ivf.data+1ull*x*DIMENSION,
            ivf.binary_code+1ull* i * (BB / 64),
            ivf.dist_to_c[i],
	        ivf.x0[i],
	        ivf.fac[i].sqr_x,
            ivf.fac[i].error,
            ivf.fac[i].factor_ppc,
            ivf.fac[i].factor_ip,
            cids.data()[x]
        );

        
    std::memcpy(index.getDataByInternalId(inner_id), &tmp_item, sizeof(Item));
    }
    // exit(0);
    


    //query part

    char query_path[500];
    char gt_file[500];
    char transformation_path[500];
    UintRowMat gt;
    sprintf(query_path, "%s/%d/%s_query.fbin", DATASET_PATH, DIMENSION, DATASET);
    Matrix<float> Q(query_path,true);

    sprintf(transformation_path, "../data/%d/%s_P_C%d_B%d.fvecs", DIMENSION, DATASET, numC, BB);
    Matrix<float> P(transformation_path);
    
    sprintf(gt_file, "%s/%d/%s_gt.fbin", DATASET_PATH, DIMENSION, DATASET);

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
    Matrix<float> RandQ(Q.n, BB, Q);
    RandQ = mul(RandQ, P);
    int NQ = RandQ.n;
    
    

    

    std::vector<int> ef_search_values;
    for (int i = 10; i <= 10; i += 10) {
        ef_search_values.push_back(i);
    }

    std::vector<Query*> queries;
    queries.resize(RandQ.n);  // Important!
    // #pragma omp parallel for schedule(static)
    for (int i = 0; i < NQ; ++i) {
        float * ptr_c = ivf.centroid;
        Result* centroid_dist = new Result[numC];
        for(int j=0;j<numC;j++){
            centroid_dist[j].first = sqr_dist<BB>(RandQ.data + i * RandQ.d, ptr_c);
            centroid_dist[j].second = j;
            ptr_c += BB;
        }
        // ivf.initer->centroid
        Query* cur_query = new Query(RandQ.data + i * RandQ.d, centroid_dist ,&ivf);
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
        summary_file << "RabitQ," << std::string(DATASET) << "," << DIMENSION << "," << getProcessPeakRSS() << "," << DIMENSION/8 << "," << ef_search <<
        "," << qps << "," << n_cmp << ","  << recall_at_1 << "," << recall_at_10 << "," << recall_at_100 << "," << recall_at_1000 << "\n" <<std::flush;
    }
    summary_file.close();
    return 0;
}
