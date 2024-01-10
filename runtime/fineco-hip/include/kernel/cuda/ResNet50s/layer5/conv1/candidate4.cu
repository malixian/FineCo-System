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
extern "C" __global__ void __launch_bounds__(128) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[64];
  __shared__ float kernel_shared[8192];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    if (((int)threadIdx.x) < 64) {
      pad_temp_shared[((int)threadIdx.x)] = data[((((((rc_outer_outer * 3136) + ((((int)threadIdx.x) >> 2) * 196)) + ((((int)blockIdx.x) / 7) * 28)) + (((((int)threadIdx.x) & 3) >> 1) * 14)) + ((((int)blockIdx.x) % 7) * 2)) + (((int)threadIdx.x) & 1))];
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 128)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 8192)];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 16384)];
    kernel_shared[(((int)threadIdx.x) + 384)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 24576)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 32768)];
    kernel_shared[(((int)threadIdx.x) + 640)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 40960)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 49152)];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 57344)];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 65536)];
    kernel_shared[(((int)threadIdx.x) + 1152)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 73728)];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 81920)];
    kernel_shared[(((int)threadIdx.x) + 1408)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 90112)];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 98304)];
    kernel_shared[(((int)threadIdx.x) + 1664)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 106496)];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 114688)];
    kernel_shared[(((int)threadIdx.x) + 1920)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 122880)];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 131072)];
    kernel_shared[(((int)threadIdx.x) + 2176)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 139264)];
    kernel_shared[(((int)threadIdx.x) + 2304)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 147456)];
    kernel_shared[(((int)threadIdx.x) + 2432)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 155648)];
    kernel_shared[(((int)threadIdx.x) + 2560)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 163840)];
    kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 172032)];
    kernel_shared[(((int)threadIdx.x) + 2816)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 180224)];
    kernel_shared[(((int)threadIdx.x) + 2944)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 188416)];
    kernel_shared[(((int)threadIdx.x) + 3072)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 196608)];
    kernel_shared[(((int)threadIdx.x) + 3200)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 204800)];
    kernel_shared[(((int)threadIdx.x) + 3328)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 212992)];
    kernel_shared[(((int)threadIdx.x) + 3456)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 221184)];
    kernel_shared[(((int)threadIdx.x) + 3584)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 229376)];
    kernel_shared[(((int)threadIdx.x) + 3712)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 237568)];
    kernel_shared[(((int)threadIdx.x) + 3840)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 245760)];
    kernel_shared[(((int)threadIdx.x) + 3968)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 253952)];
    kernel_shared[(((int)threadIdx.x) + 4096)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 262144)];
    kernel_shared[(((int)threadIdx.x) + 4224)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 270336)];
    kernel_shared[(((int)threadIdx.x) + 4352)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 278528)];
    kernel_shared[(((int)threadIdx.x) + 4480)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 286720)];
    kernel_shared[(((int)threadIdx.x) + 4608)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 294912)];
    kernel_shared[(((int)threadIdx.x) + 4736)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 303104)];
    kernel_shared[(((int)threadIdx.x) + 4864)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 311296)];
    kernel_shared[(((int)threadIdx.x) + 4992)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 319488)];
    kernel_shared[(((int)threadIdx.x) + 5120)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 327680)];
    kernel_shared[(((int)threadIdx.x) + 5248)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 335872)];
    kernel_shared[(((int)threadIdx.x) + 5376)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 344064)];
    kernel_shared[(((int)threadIdx.x) + 5504)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 352256)];
    kernel_shared[(((int)threadIdx.x) + 5632)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 360448)];
    kernel_shared[(((int)threadIdx.x) + 5760)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 368640)];
    kernel_shared[(((int)threadIdx.x) + 5888)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 376832)];
    kernel_shared[(((int)threadIdx.x) + 6016)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 385024)];
    kernel_shared[(((int)threadIdx.x) + 6144)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 393216)];
    kernel_shared[(((int)threadIdx.x) + 6272)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 401408)];
    kernel_shared[(((int)threadIdx.x) + 6400)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 409600)];
    kernel_shared[(((int)threadIdx.x) + 6528)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 417792)];
    kernel_shared[(((int)threadIdx.x) + 6656)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 425984)];
    kernel_shared[(((int)threadIdx.x) + 6784)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 434176)];
    kernel_shared[(((int)threadIdx.x) + 6912)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 442368)];
    kernel_shared[(((int)threadIdx.x) + 7040)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 450560)];
    kernel_shared[(((int)threadIdx.x) + 7168)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 458752)];
    kernel_shared[(((int)threadIdx.x) + 7296)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 466944)];
    kernel_shared[(((int)threadIdx.x) + 7424)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 475136)];
    kernel_shared[(((int)threadIdx.x) + 7552)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 483328)];
    kernel_shared[(((int)threadIdx.x) + 7680)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 491520)];
    kernel_shared[(((int)threadIdx.x) + 7808)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 499712)];
    kernel_shared[(((int)threadIdx.x) + 7936)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 507904)];
    kernel_shared[(((int)threadIdx.x) + 8064)] = kernel[(((((((int)threadIdx.x) >> 4) * 1024) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 516096)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(rc_outer_inner * 16)] * kernel_shared[((((int)threadIdx.x) * 16) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 16) + 2)] * kernel_shared[((((int)threadIdx.x) * 16) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(rc_outer_inner * 16)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2048)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 16) + 2)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2048)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(rc_outer_inner * 16)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4096)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_outer_inner * 16) + 2)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4096)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(rc_outer_inner * 16)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6144)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 16) + 2)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6144)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 16) + 4)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 16) + 6)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 16) + 4)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2049)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 16) + 6)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2049)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_outer_inner * 16) + 4)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4097)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_outer_inner * 16) + 6)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4097)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_outer_inner * 16) + 4)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6145)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 16) + 6)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6145)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 16) + 8)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 16) + 10)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 16) + 8)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2050)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 16) + 10)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2050)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_outer_inner * 16) + 8)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4098)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_outer_inner * 16) + 10)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4098)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_outer_inner * 16) + 8)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6146)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 16) + 10)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6146)]));
      conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((rc_outer_inner * 16) + 12)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((rc_outer_inner * 16) + 14)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((rc_outer_inner * 16) + 12)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2051)]));
      conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((rc_outer_inner * 16) + 14)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2051)]));
      conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[((rc_outer_inner * 16) + 12)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4099)]));
      conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[((rc_outer_inner * 16) + 14)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4099)]));
      conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[((rc_outer_inner * 16) + 12)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6147)]));
      conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[((rc_outer_inner * 16) + 14)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6147)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 16) + 1)] * kernel_shared[((((int)threadIdx.x) * 16) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 16) + 3)] * kernel_shared[((((int)threadIdx.x) * 16) + (rc_outer_inner * 4))]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 16) + 1)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2048)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 16) + 3)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2048)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[((rc_outer_inner * 16) + 1)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4096)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[((rc_outer_inner * 16) + 3)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4096)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[((rc_outer_inner * 16) + 1)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6144)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[((rc_outer_inner * 16) + 3)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6144)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 16) + 5)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 16) + 7)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 1)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 16) + 5)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2049)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 16) + 7)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2049)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[((rc_outer_inner * 16) + 5)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4097)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[((rc_outer_inner * 16) + 7)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4097)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[((rc_outer_inner * 16) + 5)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6145)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[((rc_outer_inner * 16) + 7)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6145)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 16) + 9)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 16) + 11)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 16) + 9)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2050)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 16) + 11)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2050)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[((rc_outer_inner * 16) + 9)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4098)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[((rc_outer_inner * 16) + 11)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4098)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[((rc_outer_inner * 16) + 9)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6146)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[((rc_outer_inner * 16) + 11)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6146)]));
      conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((rc_outer_inner * 16) + 13)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((rc_outer_inner * 16) + 15)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 3)]));
      conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((rc_outer_inner * 16) + 13)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2051)]));
      conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[((rc_outer_inner * 16) + 15)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 2051)]));
      conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[((rc_outer_inner * 16) + 13)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4099)]));
      conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[((rc_outer_inner * 16) + 15)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 4099)]));
      conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[((rc_outer_inner * 16) + 13)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6147)]));
      conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[((rc_outer_inner * 16) + 15)] * kernel_shared[(((((int)threadIdx.x) * 16) + (rc_outer_inner * 4)) + 6147)]));
    }
  }
  for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
    conv2d_nchw[((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner)] = conv2d_nchw_local[xx_inner];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 14)] = conv2d_nchw_local[(xx_inner + 2)];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 25088)] = conv2d_nchw_local[(xx_inner + 4)];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 25102)] = conv2d_nchw_local[(xx_inner + 6)];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 50176)] = conv2d_nchw_local[(xx_inner + 8)];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 50190)] = conv2d_nchw_local[(xx_inner + 10)];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 75264)] = conv2d_nchw_local[(xx_inner + 12)];
    conv2d_nchw[(((((((int)threadIdx.x) * 196) + ((((int)blockIdx.x) / 7) * 28)) + ((((int)blockIdx.x) % 7) * 2)) + xx_inner) + 75278)] = conv2d_nchw_local[(xx_inner + 14)];
  }
}


