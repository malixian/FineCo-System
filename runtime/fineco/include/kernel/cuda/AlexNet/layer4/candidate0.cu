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
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))] = max(tensor[((((int)blockIdx.x) * 64) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) / 169) * 729) + (((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 169) / 13) * 54)) + (rv0 * 27)) + ((((((int)blockIdx.x) * 64) + ((int)threadIdx.x)) % 13) * 2)) + rv1)]);
    }
  }
}


