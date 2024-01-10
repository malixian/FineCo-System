
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
extern "C" __global__ void __launch_bounds__(256) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[4];
  __shared__ float pad_temp_shared[50];
  __shared__ float kernel_shared[4608];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 50) {
      pad_temp_shared[((int)threadIdx.x)] = (((1 <= (((((int)blockIdx.x) / 7) * 4) + ((((int)threadIdx.x) % 25) / 5))) && (1 <= (((((int)blockIdx.x) % 7) * 4) + (((int)threadIdx.x) % 5)))) ? data[(((((((rc_outer_outer * 1568) + ((((int)threadIdx.x) / 25) * 784)) + ((((int)blockIdx.x) / 7) * 112)) + (((((int)threadIdx.x) % 25) / 5) * 28)) + ((((int)blockIdx.x) % 7) * 4)) + (((int)threadIdx.x) % 5)) - 29)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) / 18) * 1152) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) + 256) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) + 512) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) + 768) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) + 1024) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) + 1280) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) + 1536) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) + 1792) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) + 2048) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) / 18) * 1152) + (rc_outer_outer * 18)) + (((int)threadIdx.x) % 18)) + 147456)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) + 2560) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 4) % 18))];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) + 2816) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 8) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) + 3072) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 12) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) + 3328) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 16) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) + 3584) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 2) % 18))];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) + 3840) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 6) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) + 4096) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 10) % 18))];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) + 4352) / 18) * 1152) + (rc_outer_outer * 18)) + ((((int)threadIdx.x) + 14) % 18))];
    __syncthreads();
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((int)threadIdx.x) & 1) * 2)] * kernel_shared[((((int)threadIdx.x) >> 1) * 18)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((int)threadIdx.x) & 1) * 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2304)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 1)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 1)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2305)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 2)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2306)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 3)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 5)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2307)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 4)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 6)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2308)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 5)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 7)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2309)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 6)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2310)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 7)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2311)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 8)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2312)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 9)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 25)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2313)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 10)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 26)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2314)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 27)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 11)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 27)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2315)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 12)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 30)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2316)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 13)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 31)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2317)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 14)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2318)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 15)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2319)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 16)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2320)]));
    conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 17)]));
    conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2321)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[((((int)threadIdx.x) >> 1) * 18)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 10)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2304)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 1)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 11)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2305)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 12)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2306)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 3)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 15)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2307)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 4)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2308)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 5)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 17)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2309)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 6)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 20)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2310)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 21)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 7)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 21)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2311)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 22)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 8)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 22)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2312)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 9)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 35)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2313)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 10)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 36)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2314)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 11)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 37)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2315)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 12)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 40)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2316)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 41)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 13)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 41)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2317)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 14)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 42)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2318)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 45)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 15)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 45)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2319)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 46)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 16)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 46)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2320)]));
    conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 47)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 17)]));
    conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((int)threadIdx.x) & 1) * 2) + 47)] * kernel_shared[(((((int)threadIdx.x) >> 1) * 18) + 2321)]));
  }
  for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
    conv2d_nchw[((((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + (yy_inner * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))] = conv2d_nchw_local[yy_inner];
    conv2d_nchw[(((((((((int)threadIdx.x) >> 1) * 196) + ((((int)blockIdx.x) / 7) * 28)) + (yy_inner * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1)) + 25088)] = conv2d_nchw_local[(yy_inner + 2)];
  }
}


