#include "hip/hip_runtime.h"

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
extern "C" __global__ void __launch_bounds__(64) candidate0(float* __restrict__ tensor, float* __restrict__ data) {
  tensor[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 2; ++rv0) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      tensor[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = max(tensor[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))], data[(((((((((int)blockIdx.x) * 8) + (((int)threadIdx.x) >> 3)) / 7) * 224) + (rv0 * 112)) + ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 56) * 2)) + rv1)]);
    }
  }
}


