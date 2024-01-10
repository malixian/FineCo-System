
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
#define __shfl_sync(mask, var, lane, width) \
        __shfl((var), (lane), (width))

#define __shfl_down_sync(mask, var, offset, width) \
        __shfl_down((var), (offset), (width))

#define __shfl_up_sync(mask, var, offset, width) \
        __shfl_up((var), (offset), (width))
#endif


#ifdef _WIN32
  using uint = unsigned int;
  using uchar = unsigned char;
  using ushort = unsigned short;
  using int64_t = long long;
  using uint64_t = unsigned long long;
#else
  #define uint unsigned int
  #define uchar unsigned char
  #define ushort unsigned short
  #define int64_t long long
  #define uint64_t unsigned long long
#endif
extern "C" __global__ void __launch_bounds__(32) candidate0(float* __restrict__ data, float* __restrict__ tensor) {
  float normal_reduce_temp0[1];
  float red_buf0[1];
  __shared__ float tensor1[1];
  normal_reduce_temp0[0] = 0.000000e+00f;
  normal_reduce_temp0[0] = (normal_reduce_temp0[0] + data[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))]);
  if (((int)threadIdx.x) < 17) {
    normal_reduce_temp0[0] = (normal_reduce_temp0[0] + data[(((((int)blockIdx.x) * 49) + ((int)threadIdx.x)) + 32)]);
  }
  uint mask[1];
  float t0[1];
  red_buf0[0] = normal_reduce_temp0[0];
  mask[0] = __activemask();
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 16, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 8, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 4, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 2, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  t0[0] = __shfl_down_sync(mask[0], red_buf0[0], 1, 32);
  red_buf0[0] = (red_buf0[0] + t0[0]);
  red_buf0[0] = __shfl_sync(mask[0], red_buf0[0], 0, 32);
  tensor1[0] = red_buf0[0];
  __syncthreads();
  if (((int)threadIdx.x) < 1) {
    tensor[(((int)blockIdx.x) + ((int)threadIdx.x))] = (tensor1[((int)threadIdx.x)] * 2.040816e-02f);
  }
}


