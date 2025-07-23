#pragma once

#include "../EXrabitq/Item.hpp"

namespace hnswlib {
static float
RL2Sqr(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
    const Item* item1 = static_cast<const Item*>(pVect1v);
    const Item* item2 = static_cast<const Item*>(pVect2v);
    float *pVect1 = (float *) item1->values;
    float *pVect2 = (float *) item2->values;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *pVect1 - *pVect2;
        pVect1++;
        pVect2++;
        res += t * t;
    }
    return (res);
}
    
    #if defined(USE_AVX512)
    
    // Favor using AVX512 if available.
    static float
    RL2SqrSIMD16ExtAVX512(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const Item* item1 = static_cast<const Item*>(pVect1v);
        const Item* item2 = static_cast<const Item*>(pVect2v);
        float *pVect1 = (float *) item1->values;
        float *pVect2 = (float *) item2->values;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN64 TmpRes[16];
        size_t qty16 = qty >> 4;
    
        const float *pEnd1 = pVect1 + (qty16 << 4);
    
        __m512 diff, v1, v2;
        __m512 sum = _mm512_set1_ps(0);
    
        while (pVect1 < pEnd1) {
            v1 = _mm512_loadu_ps(pVect1);
            pVect1 += 16;
            v2 = _mm512_loadu_ps(pVect2);
            pVect2 += 16;
            diff = _mm512_sub_ps(v1, v2);
            // sum = _mm512_fmadd_ps(diff, diff, sum);
            sum = _mm512_add_ps(sum, _mm512_mul_ps(diff, diff));
        }
    
        _mm512_store_ps(TmpRes, sum);
        float res = TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] +
                TmpRes[7] + TmpRes[8] + TmpRes[9] + TmpRes[10] + TmpRes[11] + TmpRes[12] +
                TmpRes[13] + TmpRes[14] + TmpRes[15];
    
        return (res);
    }
    #endif
    
    #if defined(USE_AVX)
    
    // Favor using AVX if available.
    static float
    RL2SqrSIMD16ExtAVX(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const Item* item1 = static_cast<const Item*>(pVect1v);
        const Item* item2 = static_cast<const Item*>(pVect2v);
        float *pVect1 = (float *) item1->values;
        float *pVect2 = (float *) item2->values;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;
    
        const float *pEnd1 = pVect1 + (qty16 << 4);
    
        __m256 diff, v1, v2;
        __m256 sum = _mm256_set1_ps(0);
    
        while (pVect1 < pEnd1) {
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    
            v1 = _mm256_loadu_ps(pVect1);
            pVect1 += 8;
            v2 = _mm256_loadu_ps(pVect2);
            pVect2 += 8;
            diff = _mm256_sub_ps(v1, v2);
            sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
        }
    
        _mm256_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3] + TmpRes[4] + TmpRes[5] + TmpRes[6] + TmpRes[7];
    }
    
    #endif
    
    #if defined(USE_SSE)
    
    static float
    RL2SqrSIMD16ExtSSE(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const Item* item1 = static_cast<const Item*>(pVect1v);
        const Item* item2 = static_cast<const Item*>(pVect2v);
        float *pVect1 = (float *) item1->values;
        float *pVect2 = (float *) item2->values;
        size_t qty = *((size_t *) qty_ptr);
        float PORTABLE_ALIGN32 TmpRes[8];
        size_t qty16 = qty >> 4;
    
        const float *pEnd1 = pVect1 + (qty16 << 4);
    
        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);
    
        while (pVect1 < pEnd1) {
            //_mm_prefetch((char*)(pVect2 + 16), _MM_HINT_T0);
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
    
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
    
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
    #endif
    
    #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
    static DISTFUNC<float> RL2SqrSIMD16Ext = RL2SqrSIMD16ExtSSE;
    
    static float
    RL2SqrSIMD16ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const Item* item1 = static_cast<const Item*>(pVect1v);
        const Item* item2 = static_cast<const Item*>(pVect2v);
        float *pVect1 = (float *) item1->values;
        float *pVect2 = (float *) item2->values;
        size_t qty = *((size_t *) qty_ptr);
        size_t qty16 = qty >> 4 << 4;
        float res = L2SqrSIMD16Ext(pVect1, pVect2, &qty16);
        float *p1 = (float *) pVect1 + qty16;
        float *p2 = (float *) pVect2 + qty16;
    
        size_t qty_left = qty - qty16;
        float res_tail = L2Sqr(p1, p2, &qty_left);
        return (res + res_tail);
    }
    #endif
    
    
    #if defined(USE_SSE)
    static float
    RL2SqrSIMD4Ext(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        float PORTABLE_ALIGN32 TmpRes[8];
        const Item* item1 = static_cast<const Item*>(pVect1v);
        const Item* item2 = static_cast<const Item*>(pVect2v);
        float *pVect1 = (float *) item1->values;
        float *pVect2 = (float *) item2->values;
        size_t qty = *((size_t *) qty_ptr);
    
    
        size_t qty4 = qty >> 2;
    
        const float *pEnd1 = pVect1 + (qty4 << 2);
    
        __m128 diff, v1, v2;
        __m128 sum = _mm_set1_ps(0);
    
        while (pVect1 < pEnd1) {
            v1 = _mm_loadu_ps(pVect1);
            pVect1 += 4;
            v2 = _mm_loadu_ps(pVect2);
            pVect2 += 4;
            diff = _mm_sub_ps(v1, v2);
            sum = _mm_add_ps(sum, _mm_mul_ps(diff, diff));
        }
        _mm_store_ps(TmpRes, sum);
        return TmpRes[0] + TmpRes[1] + TmpRes[2] + TmpRes[3];
    }
    
    static float
    RL2SqrSIMD4ExtResiduals(const void *pVect1v, const void *pVect2v, const void *qty_ptr) {
        const Item* item1 = static_cast<const Item*>(pVect1v);
        const Item* item2 = static_cast<const Item*>(pVect2v);
        float *pV1 = (float *) item1->values;
        float *pV2 = (float *) item2->values;
        size_t qty = *((size_t *) qty_ptr);
        size_t qty4 = qty >> 2 << 2;
    
        float res = RL2SqrSIMD4Ext(pV1, pV2, &qty4);
        size_t qty_left = qty - qty4;
    
        float *pVect1 = (float *) pV1 + qty4;
        float *pVect2 = (float *) pV2 + qty4;
        float res_tail = RL2Sqr(pVect1, pVect2, &qty_left);
    
        return (res + res_tail);
    }
    #endif

static float
Calculate_dist( const void *pVect1, const void *pVect2,  const void *qty_ptr) {
    const Item* item1 = static_cast<const Item*>(pVect1);
    const Item* item2 = static_cast<const Item*>(pVect2);


    float *vec1 = (float *) item1->values;
    float *vec2 = (float *) item2->values;
    size_t qty = *((size_t *) qty_ptr);

    float res = 0;
    for (size_t i = 0; i < qty; i++) {
        float t = *vec1 - *vec2;
        vec1++;
        vec2++;
        res += t * t;
    }
    return (res);
}
static float
Calculate_Qdist( const void *pVect1, const void *pVect2,  const void *qty_ptr) {
    
    Query* q = const_cast<Query*>(static_cast<const Query*>(pVect1));
    
    const Item* vb = static_cast<const Item*>(pVect2);
    

    double dist =0;
    
    PID id = vb->id;//
    PID cid = q->cids->data()[id];
    // float sqr_y =  (q->centroid_dist)[cid].distance;
    float sqr_y = 0.0;
    for (const auto& cand : q->centroid_dist) {
        if (cand.id == cid) {
            sqr_y = cand.distance;
            break;
        }
    }
    float y = std::sqrt(sqr_y);
    
    float* cur_centroid = q->ivf->initer->centroid(cid);
    q->searcher->search_prepare(cur_centroid,y,sqr_y);
    dist = q->searcher->search_point(vb->id, vb->short_code, vb->nth, vb->long_code,
        vb->exfactor, vb->factor_x2);


    return dist;
}



class RabitqSpace : public SpaceInterface<float> {
 private:
    size_t dim_;
    size_t data_size_;
    DISTFUNC<float> fstdistfunc_;
    DISTFUNC<float> querystdistfunc_;

 public:
    explicit RabitqSpace(size_t dim, size_t size) {
        #if defined(USE_SSE) || defined(USE_AVX) || defined(USE_AVX512)
        #if defined(USE_AVX512)
            if (AVX512Capable())
                RL2SqrSIMD16Ext = RL2SqrSIMD16ExtAVX512;
            else if (AVXCapable())
                RL2SqrSIMD16Ext = L2SqrSIMD16ExtAVX;
        #elif defined(USE_AVX)
            if (AVXCapable())
                RL2SqrSIMD16Ext = RL2SqrSIMD16ExtAVX;
        #endif
    
            if (dim % 16 == 0)
                fstdistfunc_ = RL2SqrSIMD16Ext;
            else if (dim % 4 == 0)
                fstdistfunc_ = RL2SqrSIMD4Ext;
            else if (dim > 16)
                fstdistfunc_ = RL2SqrSIMD16ExtResiduals;
            else if (dim > 4)
                fstdistfunc_ = RL2SqrSIMD4ExtResiduals;
    #endif
        querystdistfunc_ = Calculate_Qdist;
        dim_ = dim;
        data_size_ = size;
    }
    

    size_t get_data_size() override {
        return data_size_;
    }

    DISTFUNC<float> get_dist_func() override {
        return fstdistfunc_;
    }
    DISTFUNC<float> get_qdist_func() override {
        return querystdistfunc_;
    }

    void *get_dist_func_param() override {
        return &dim_;
    }

    ~RabitqSpace() override = default;
};

} // namespace hnswlib