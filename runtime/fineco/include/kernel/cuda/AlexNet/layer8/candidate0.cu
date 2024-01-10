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
extern "C" __global__ void __launch_bounds__(8) candidate0(float* __restrict__ tensor, float* __restrict__ data) {
  tensor[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))] = max(tensor[((((int)blockIdx.x) * 8) + ((int)threadIdx.x))], data[((((((((((int)blockIdx.x) * 2) + (((int)threadIdx.x) >> 2)) / 9) * 169) + (((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 1)) % 18) / 3) * 26)) + (rv0 * 13)) + ((((((int)blockIdx.x) * 8) + ((int)threadIdx.x)) % 6) * 2)) + rv1)]);
    }
  }
}


