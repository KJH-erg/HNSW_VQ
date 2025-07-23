#pragma once
#include "IVF.hpp"


struct Item {
	float* values;
	PID id;
    uint8_t* short_code;  // First byte array
    uint8_t* long_code;   // Second byte array
    ExFactor* exfactor;   // Custom struct pointer
    float* factor_x2;     // Float array
    int DIM;
	PID nth;
	
    Item(int id , uint8_t* short_code, PID nth, uint8_t* long_code, ExFactor* exfactor, float* factor_x2,
                int DIM)
        : id(id), short_code(short_code),nth(nth), factor_x2(factor_x2), DIM(DIM)
		  {
			this->long_code = long_code;
			this->exfactor = exfactor;
		  }
		~Item() {
	}
};



struct Query {
	float* query;
	PID cid;
	float sqr_y;
	PID centroid_id;
	UintRowMat* cids;
	int DIM;
	std::vector<Candidate> centroid_dist;
	IVF *ivf;
	int K;
	HASearcher* searcher;
    Query(float* query,  UintRowMat* cids, int DIM , IVF *ivf, int K, std::vector<Candidate>& centroid_dist)
        : query(query), cids(cids), DIM(DIM), K(K)
		  {
			this->searcher = new HASearcher(query, DIM, ivf->EX_BITS, ivf->DQ);
			this->ivf = ivf;
			this->centroid_dist = centroid_dist;
		}
		
		
		
		~Query() {}
};