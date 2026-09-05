__device__ void warpOnlineReduce(float& m, float& l){
    for(int offset = 16; offset > 0; offset >>= 1){
        float other_m = __shfl_down_sync(0xffffffff, m, offset);

        float other_l = __shfl_down_sync(0xffffffff, l, offset);

        float new_m = fmaxf(other_m, m);

        l = l * expf(m - new_m) + other_l * expf(other_m - new_m);

        m = new_m;
    }
}

__global__ 