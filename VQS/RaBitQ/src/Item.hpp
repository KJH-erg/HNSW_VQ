#pragma once
#include "ivf_rabitq.h"
#include "space.h"

struct Item {
	uint32_t id;
	float* data;
	uint64_t* binary_code;
	float dist_to_c;
	float x0;
	float sqr_x;
    float error;
    float factor_ppc;
    float factor_ip;
	int cluster_id;
	
    Item(uint32_t id , float* data, uint64_t* binary_code, float dist_to_c, float x0,float sqr_x,float error,
		float factor_ppc, float factor_ip, int cluster_id)
        : id(id), data(data),binary_code(binary_code), dist_to_c(dist_to_c), x0(x0), sqr_x(sqr_x), error(error),factor_ppc(factor_ppc),factor_ip(factor_ip), cluster_id(cluster_id)
		  {
		  }
		~Item() {
	}
};



struct Query {
	float* query;
	Result* centroid_dist;
	float* centroid;
	Result *ptr_centroid_dist ;
	PORTABLE_ALIGN64 uint8_t byte_query[numC][BB];
	PORTABLE_ALIGN32 uint64_t quant_query[numC][B_QUERY * BB / 64];
	std::vector<uint32_t> sum_q;
	IVFRN<DIMENSION, BB>* ivf;
	std::vector<float> vls; 
	std::vector<float> vrs; 

    Query(float* query, Result* centroid_dist, IVFRN<DIMENSION, BB>* ivf)
        : query(query), centroid_dist(centroid_dist), ivf(ivf)
		  {
			// std::partial_sort(centroid_dist, centroid_dist + numC, centroid_dist + numC);
			// ptr_centroid_dist = (&centroid_dist[0]);
			// vls.reserve(numC); 
			// vrs.reserve(numC); 
			// sum_q.reserve(numC);
			// for(int pb=0;pb<numC;pb++){ 
			// 	uint32_t c = ptr_centroid_dist -> second;
			// 	// std::cout << "cluster id" << c << std::endl;
			// 	float sqr_y = ptr_centroid_dist -> first;
			// 	ptr_centroid_dist ++;
			// 	uint32_t tmp_sumq;
			// 	float vl, vr;
				
				
				
			// }
			
		}
		
		
		~Query() {}
};