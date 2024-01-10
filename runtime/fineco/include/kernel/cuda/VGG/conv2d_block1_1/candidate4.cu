
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
extern "C" __global__ void __launch_bounds__(512) candidate4(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[64];
  __shared__ float pad_temp_shared[3264];
  __shared__ float kernel_shared[288];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[56] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[57] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[58] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[59] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[60] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[61] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[62] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[63] = 0.000000e+00f;
  for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= ((((((int)blockIdx.x) % 49) / 7) * 32) + (((int)threadIdx.x) >> 5))) && (1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[((((((((((int)blockIdx.x) % 49) / 7) * 7168) + ((((int)threadIdx.x) >> 5) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 512)] = (((1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[((((((((((int)blockIdx.x) % 49) / 7) * 7168) + ((((int)threadIdx.x) >> 5) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) + 3359)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((1 <= ((((((int)blockIdx.x) % 49) / 7) * 32) + (((((int)threadIdx.x) >> 5) + 32) % 34))) && (((((((int)blockIdx.x) % 49) / 7) * 32) + (((((int)threadIdx.x) >> 5) + 32) % 34)) < 225)) && (1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[(((((((((((int)threadIdx.x) + 1024) / 1088) * 50176) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + ((((((int)threadIdx.x) >> 5) + 32) % 34) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[(((((((((((int)threadIdx.x) + 1536) / 1088) * 50176) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + (((((int)threadIdx.x) >> 5) + 14) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((((1 <= ((((((int)blockIdx.x) % 49) / 7) * 32) + (((((int)threadIdx.x) >> 5) + 30) % 34))) && (((((((int)blockIdx.x) % 49) / 7) * 32) + (((((int)threadIdx.x) >> 5) + 30) % 34)) < 225)) && (1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[(((((((((((int)threadIdx.x) + 2048) / 1088) * 50176) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + ((((((int)threadIdx.x) >> 5) + 30) % 34) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) - 225)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2560)] = (((1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[(((((((((((int)threadIdx.x) + 2560) / 1088) * 50176) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + (((((int)threadIdx.x) >> 5) + 12) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) - 225)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 192) {
      pad_temp_shared[(((int)threadIdx.x) + 3072)] = ((((((((((int)blockIdx.x) % 49) / 7) * 32) + ((((int)threadIdx.x) >> 5) + 28)) < 225) && (1 <= ((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)))) && (((((((int)blockIdx.x) % 7) * 32) + rx_outer_outer) + (((int)threadIdx.x) & 31)) < 225)) ? data[(((((((((((int)threadIdx.x) + 3072) / 1088) * 50176) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + (((((int)threadIdx.x) >> 5) + 28) * 224)) + ((((int)blockIdx.x) % 7) * 32)) + rx_outer_outer) + (((int)threadIdx.x) & 31)) - 225)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 288) {
      kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) / 49) * 864) + (((int)threadIdx.x) * 3)) + rx_outer_outer)];
    }
    __syncthreads();
    for (int yy_c_outer_inner = 0; yy_c_outer_inner < 4; ++yy_c_outer_inner) {
      for (int rc_inner = 0; rc_inner < 3; ++rc_inner) {
        conv2d_nchw_local[(yy_c_outer_inner * 4)] = (conv2d_nchw_local[(yy_c_outer_inner * 4)] + (pad_temp_shared[((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31))] * kernel_shared[(((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3))]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 32)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 32)] + (pad_temp_shared[((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31))] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 144)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3))]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 33)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 33)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 144)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[(((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3))]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 34)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 34)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 144)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[(((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3))]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 35)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 35)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 144)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 16)] + (pad_temp_shared[((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31))] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 9)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 48)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 48)] + (pad_temp_shared[((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31))] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 153)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 17)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 9)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 49)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 49)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 153)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 18)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 9)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 50)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 50)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 153)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 19)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 9)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 51)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 51)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 153)]));
        conv2d_nchw_local[(yy_c_outer_inner * 4)] = (conv2d_nchw_local[(yy_c_outer_inner * 4)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 1)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 32)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 32)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 145)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 1)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 33)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 33)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 145)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 1)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 34)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 34)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 145)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 1)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 35)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 35)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 145)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 16)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 10)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 48)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 48)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 154)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 17)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 10)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 49)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 49)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 154)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 18)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 10)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 50)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 50)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 154)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 19)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 10)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 51)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 51)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 154)]));
        conv2d_nchw_local[(yy_c_outer_inner * 4)] = (conv2d_nchw_local[(yy_c_outer_inner * 4)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 2)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 32)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 32)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 146)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 1)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 1)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 2)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 33)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 33)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 146)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 2)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 2)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 2)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 34)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 34)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 146)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 3)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 3)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 2)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 35)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 35)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 146)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 16)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 16)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 11)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 48)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 48)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 155)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 17)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 17)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 11)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 49)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 49)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 155)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 18)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 18)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 11)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 50)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 50)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 155)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 19)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 19)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 11)]));
        conv2d_nchw_local[((yy_c_outer_inner * 4) + 51)] = (conv2d_nchw_local[((yy_c_outer_inner * 4) + 51)] + (pad_temp_shared[(((((rc_inner * 1088) + (((((int)threadIdx.x) & 63) >> 5) * 512)) + (yy_c_outer_inner * 128)) + (((int)threadIdx.x) & 31)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 6) * 18) + (rc_inner * 3)) + 155)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 16; ++yy_inner) {
      conv2d_nchw[(((((((((((int)blockIdx.x) / 49) * 1605632) + ((((int)threadIdx.x) >> 6) * 100352)) + (ff_inner * 50176)) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + (((((int)threadIdx.x) & 63) >> 5) * 3584)) + (yy_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 31))] = conv2d_nchw_local[((ff_inner * 16) + yy_inner)];
      conv2d_nchw[((((((((((((int)blockIdx.x) / 49) * 1605632) + ((((int)threadIdx.x) >> 6) * 100352)) + (ff_inner * 50176)) + (((((int)blockIdx.x) % 49) / 7) * 7168)) + (((((int)threadIdx.x) & 63) >> 5) * 3584)) + (yy_inner * 224)) + ((((int)blockIdx.x) % 7) * 32)) + (((int)threadIdx.x) & 31)) + 802816)] = conv2d_nchw_local[(((ff_inner * 16) + yy_inner) + 32)];
    }
  }
}


