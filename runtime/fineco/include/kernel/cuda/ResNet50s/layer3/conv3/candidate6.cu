
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
extern "C" __global__ void __launch_bounds__(224) candidate6(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[28];
  __shared__ float pad_temp_shared[1568];
  __shared__ float kernel_shared[4096];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 25088) + ((((int)threadIdx.x) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + (((((int)threadIdx.x) % 49) / 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 224) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) / 7) + 4) % 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 448) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) / 7) + 1) % 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 672) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) / 7) + 5) % 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 896)] = data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 896) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) / 7) + 2) % 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1120) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) / 7) + 6) % 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = data[((((((rc_outer_outer * 25088) + (((((int)threadIdx.x) + 1344) / 49) * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + ((((((int)threadIdx.x) / 7) + 3) % 7) * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 896)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 2688)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 4480)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5376)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 6272)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8064)];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8960)];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 9856)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 11648)];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 13440)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 15232)];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((((int)blockIdx.x) >> 4) * 16384) + ((((int)threadIdx.x) >> 5) * 128)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16128)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 7))] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[(((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 32)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 64)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[((rc_outer_inner * 196) + (((int)threadIdx.x) % 7))] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 7)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 21)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 35)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 96)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 77)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 91)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 77)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 91)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 33)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 77)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 91)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 65)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 49)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 63)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 77)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 91)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 97)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 119)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 133)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 119)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 133)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 34)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 119)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 133)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 66)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 105)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 119)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 126)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 133)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 140)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 98)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 147)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 161)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 175)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 189)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 147)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 161)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 175)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 189)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 35)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 147)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 161)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 175)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 189)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 67)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 147)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 154)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 161)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 168)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 175)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 182)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 196) + (((int)threadIdx.x) % 7)) + 189)] * kernel_shared[((((((int)threadIdx.x) / 7) * 128) + (rc_outer_inner * 4)) + 99)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) >> 4) * 100352) + ((((int)threadIdx.x) / 7) * 3136)) + (ff_inner * 784)) + (((((int)blockIdx.x) & 15) >> 2) * 196)) + (yy_inner * 28)) + ((((int)blockIdx.x) & 3) * 7)) + (((int)threadIdx.x) % 7))] = conv2d_nchw_local[((ff_inner * 7) + yy_inner)];
    }
  }
}


