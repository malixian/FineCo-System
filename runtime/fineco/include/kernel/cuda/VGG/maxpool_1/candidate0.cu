
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
extern "C" __global__ void __launch_bounds__(49) candidate0(float* __restrict__ tensor, float* __restrict__ data) {
  tensor[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 2; ++rv0) {
    for (int rv1 = 0; rv1 < 2; ++rv1) {
      tensor[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))] = max(tensor[((((int)blockIdx.x) * 49) + ((int)threadIdx.x))], data[(((((((((int)blockIdx.x) * 7) + (((int)threadIdx.x) / 7)) >> 4) * 448) + (rv0 * 224)) + ((((((int)blockIdx.x) * 49) + ((int)threadIdx.x)) % 112) * 2)) + rv1)]);
    }
  }
}


