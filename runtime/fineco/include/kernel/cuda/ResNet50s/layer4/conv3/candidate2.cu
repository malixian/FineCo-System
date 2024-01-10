
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
extern "C" __global__ void __launch_bounds__(224) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[32];
  __shared__ float pad_temp_shared[896];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = data[((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28))];
    pad_temp_shared[(((int)threadIdx.x) + 224)] = data[(((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 448)] = data[(((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 672)] = data[(((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 28) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28)) + 4704)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 1792)];
    kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 672)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 5376)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    kernel_shared[(((int)threadIdx.x) + 1120)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 8960)];
    kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 10752)];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 14336)];
    kernel_shared[(((int)threadIdx.x) + 2016)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 16128)];
    kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 17920)];
    kernel_shared[(((int)threadIdx.x) + 2464)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 19712)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 21504)];
    kernel_shared[(((int)threadIdx.x) + 2912)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 23296)];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 25088)];
    kernel_shared[(((int)threadIdx.x) + 3360)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 26880)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 28672)];
    kernel_shared[(((int)threadIdx.x) + 3808)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 30464)];
    kernel_shared[(((int)threadIdx.x) + 4032)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 32256)];
    kernel_shared[(((int)threadIdx.x) + 4256)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 34048)];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 35840)];
    kernel_shared[(((int)threadIdx.x) + 4704)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 37632)];
    kernel_shared[(((int)threadIdx.x) + 4928)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 39424)];
    kernel_shared[(((int)threadIdx.x) + 5152)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 41216)];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 43008)];
    kernel_shared[(((int)threadIdx.x) + 5600)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 44800)];
    kernel_shared[(((int)threadIdx.x) + 5824)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 46592)];
    kernel_shared[(((int)threadIdx.x) + 6048)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 48384)];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 50176)];
    kernel_shared[(((int)threadIdx.x) + 6496)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 51968)];
    kernel_shared[(((int)threadIdx.x) + 6720)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 53760)];
    kernel_shared[(((int)threadIdx.x) + 6944)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 55552)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 7392)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 59136)];
    kernel_shared[(((int)threadIdx.x) + 7616)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 60928)];
    kernel_shared[(((int)threadIdx.x) + 7840)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 62720)];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[((((((((int)blockIdx.x) / 7) * 65536) + ((((int)threadIdx.x) >> 5) * 256)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 64512)];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 32; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 14) * 64) + rc_outer_inner)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 1024)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 2048)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 3072)]));
      conv2d_nchw_local[16] = (conv2d_nchw_local[16] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 4096)]));
      conv2d_nchw_local[20] = (conv2d_nchw_local[20] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 5120)]));
      conv2d_nchw_local[24] = (conv2d_nchw_local[24] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 6144)]));
      conv2d_nchw_local[28] = (conv2d_nchw_local[28] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 7168)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 14) * 64) + rc_outer_inner)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 1024)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 2048)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 3072)]));
      conv2d_nchw_local[17] = (conv2d_nchw_local[17] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 4096)]));
      conv2d_nchw_local[21] = (conv2d_nchw_local[21] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 5120)]));
      conv2d_nchw_local[25] = (conv2d_nchw_local[25] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 6144)]));
      conv2d_nchw_local[29] = (conv2d_nchw_local[29] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 7168)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 32)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 1056)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 2080)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 3104)]));
      conv2d_nchw_local[18] = (conv2d_nchw_local[18] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 4128)]));
      conv2d_nchw_local[22] = (conv2d_nchw_local[22] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 5152)]));
      conv2d_nchw_local[26] = (conv2d_nchw_local[26] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 6176)]));
      conv2d_nchw_local[30] = (conv2d_nchw_local[30] + (pad_temp_shared[((rc_outer_inner * 28) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 7200)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 32)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 1056)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 2080)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 3104)]));
      conv2d_nchw_local[19] = (conv2d_nchw_local[19] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 4128)]));
      conv2d_nchw_local[23] = (conv2d_nchw_local[23] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 5152)]));
      conv2d_nchw_local[27] = (conv2d_nchw_local[27] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 6176)]));
      conv2d_nchw_local[31] = (conv2d_nchw_local[31] + (pad_temp_shared[(((rc_outer_inner * 28) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 14) * 64) + rc_outer_inner) + 7200)]));
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14))] = conv2d_nchw_local[((ff_inner * 2) + yy_inner)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 6272)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 4)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 12544)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 8)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 18816)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 12)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 25088)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 16)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 31360)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 20)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 37632)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 24)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 14) * 392)) + (ff_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14)) + 43904)] = conv2d_nchw_local[(((ff_inner * 2) + yy_inner) + 28)];
    }
  }
}


