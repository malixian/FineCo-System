
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
extern "C" __global__ void __launch_bounds__(224) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[56];
  __shared__ float pad_temp_shared[1568];
  __shared__ float kernel_shared[2048];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[28] = 0.000000e+00f;
  conv2d_nchw[42] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[29] = 0.000000e+00f;
  conv2d_nchw[43] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[30] = 0.000000e+00f;
  conv2d_nchw[44] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[31] = 0.000000e+00f;
  conv2d_nchw[45] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[32] = 0.000000e+00f;
  conv2d_nchw[46] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[33] = 0.000000e+00f;
  conv2d_nchw[47] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[34] = 0.000000e+00f;
  conv2d_nchw[48] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[35] = 0.000000e+00f;
  conv2d_nchw[49] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[36] = 0.000000e+00f;
  conv2d_nchw[50] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[37] = 0.000000e+00f;
  conv2d_nchw[51] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[38] = 0.000000e+00f;
  conv2d_nchw[52] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[39] = 0.000000e+00f;
  conv2d_nchw[53] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[40] = 0.000000e+00f;
  conv2d_nchw[54] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  conv2d_nchw[41] = 0.000000e+00f;
  conv2d_nchw[55] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + (((((int)threadIdx.x) % 98) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = Input[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 224) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + ((((((int)threadIdx.x) / 14) + 2) % 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = Input[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 448) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + ((((((int)threadIdx.x) / 14) + 4) % 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = Input[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 672) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + ((((((int)threadIdx.x) / 14) + 6) % 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = Input[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 896) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + ((((((int)threadIdx.x) / 14) + 1) % 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = Input[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1120) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + ((((((int)threadIdx.x) / 14) + 3) % 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = Input[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1344) / 98) * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + ((((((int)threadIdx.x) / 14) + 5) % 7) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 17920)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 25088)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 28672)];
    if (((int)threadIdx.x) < 32) {
      kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) >> 3) * 32768) + ((((int)threadIdx.x) >> 4) * 256)) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 32256)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[42] = (conv2d_nchw[42] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[43] = (conv2d_nchw[43] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[44] = (conv2d_nchw[44] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[45] = (conv2d_nchw[45] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[32] = (conv2d_nchw[32] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[46] = (conv2d_nchw[46] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[33] = (conv2d_nchw[33] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[47] = (conv2d_nchw[47] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[(((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2))]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 512)]));
      conv2d_nchw[34] = (conv2d_nchw[34] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1024)]));
      conv2d_nchw[48] = (conv2d_nchw[48] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1536)]));
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[42] = (conv2d_nchw[42] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[43] = (conv2d_nchw[43] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[44] = (conv2d_nchw[44] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[45] = (conv2d_nchw[45] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[32] = (conv2d_nchw[32] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[46] = (conv2d_nchw[46] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[33] = (conv2d_nchw[33] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[47] = (conv2d_nchw[47] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 513)]));
      conv2d_nchw[34] = (conv2d_nchw[34] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1025)]));
      conv2d_nchw[48] = (conv2d_nchw[48] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1537)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[35] = (conv2d_nchw[35] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[49] = (conv2d_nchw[49] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[36] = (conv2d_nchw[36] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[50] = (conv2d_nchw[50] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[37] = (conv2d_nchw[37] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[51] = (conv2d_nchw[51] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[38] = (conv2d_nchw[38] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[52] = (conv2d_nchw[52] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[39] = (conv2d_nchw[39] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[53] = (conv2d_nchw[53] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[40] = (conv2d_nchw[40] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[54] = (conv2d_nchw[54] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 16)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 528)]));
      conv2d_nchw[41] = (conv2d_nchw[41] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1040)]));
      conv2d_nchw[55] = (conv2d_nchw[55] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1552)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[35] = (conv2d_nchw[35] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[49] = (conv2d_nchw[49] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[36] = (conv2d_nchw[36] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[50] = (conv2d_nchw[50] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[37] = (conv2d_nchw[37] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[51] = (conv2d_nchw[51] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[38] = (conv2d_nchw[38] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[52] = (conv2d_nchw[52] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[39] = (conv2d_nchw[39] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[53] = (conv2d_nchw[53] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[40] = (conv2d_nchw[40] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[54] = (conv2d_nchw[54] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 17)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 529)]));
      conv2d_nchw[41] = (conv2d_nchw[41] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1041)]));
      conv2d_nchw[55] = (conv2d_nchw[55] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 14)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 14) * 32) + (rc_outer_inner * 2)) + 1553)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 7; ++i2_inner) {
      compute[((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 1568)) + (i1_inner * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))] = max(conv2d_nchw[((i1_inner * 7) + i2_inner)], 0.000000e+00f);
      compute[(((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 1568)) + (i1_inner * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 25088)] = max(conv2d_nchw[(((i1_inner * 7) + i2_inner) + 14)], 0.000000e+00f);
      compute[(((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 1568)) + (i1_inner * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 50176)] = max(conv2d_nchw[(((i1_inner * 7) + i2_inner) + 28)], 0.000000e+00f);
      compute[(((((((((((int)blockIdx.x) >> 3) * 100352) + ((((int)threadIdx.x) / 14) * 1568)) + (i1_inner * 784)) + (((((int)blockIdx.x) & 7) >> 1) * 196)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 75264)] = max(conv2d_nchw[(((i1_inner * 7) + i2_inner) + 42)], 0.000000e+00f);
    }
  }
}


