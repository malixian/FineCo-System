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
extern "C" __global__ void __launch_bounds__(32) candidate0(float* __restrict__ tensor, float* __restrict__ data) {
  tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = -3.402823e+38f;
  for (int rv0 = 0; rv0 < 3; ++rv0) {
    for (int rv1 = 0; rv1 < 3; ++rv1) {
      tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))] = max(tensor[((((int)blockIdx.x) * 32) + ((int)threadIdx.x))], (((1 <= ((((((((int)blockIdx.x) % 98) * 4) + (((int)threadIdx.x) >> 3)) / 7) * 2) + rv0)) && (1 <= (((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 56) * 2) + rv1))) ? data[((((((((((int)blockIdx.x) * 4) + (((int)threadIdx.x) >> 3)) / 7) * 224) + (rv0 * 112)) + ((((((int)blockIdx.x) * 32) + ((int)threadIdx.x)) % 56) * 2)) + rv1) - 113)] : -3.402823e+38f));
    }
  }
}