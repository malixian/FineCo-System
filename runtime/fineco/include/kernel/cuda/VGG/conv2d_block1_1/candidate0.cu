
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
extern "C" __global__ void __launch_bounds__(28) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[64];
  __shared__ float pad_temp_shared[456];
  __shared__ float kernel_shared[72];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[32] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[33] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[34] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[35] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[36] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[37] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[38] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[39] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[40] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[41] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[42] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[43] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[44] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[45] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[46] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  conv2d_nchw_local[47] = 0.000000e+00f;
  conv2d_nchw_local[16] = 0.000000e+00f;
  conv2d_nchw_local[48] = 0.000000e+00f;
  conv2d_nchw_local[17] = 0.000000e+00f;
  conv2d_nchw_local[49] = 0.000000e+00f;
  conv2d_nchw_local[18] = 0.000000e+00f;
  conv2d_nchw_local[50] = 0.000000e+00f;
  conv2d_nchw_local[19] = 0.000000e+00f;
  conv2d_nchw_local[51] = 0.000000e+00f;
  conv2d_nchw_local[20] = 0.000000e+00f;
  conv2d_nchw_local[52] = 0.000000e+00f;
  conv2d_nchw_local[21] = 0.000000e+00f;
  conv2d_nchw_local[53] = 0.000000e+00f;
  conv2d_nchw_local[22] = 0.000000e+00f;
  conv2d_nchw_local[54] = 0.000000e+00f;
  conv2d_nchw_local[23] = 0.000000e+00f;
  conv2d_nchw_local[55] = 0.000000e+00f;
  conv2d_nchw_local[24] = 0.000000e+00f;
  conv2d_nchw_local[56] = 0.000000e+00f;
  conv2d_nchw_local[25] = 0.000000e+00f;
  conv2d_nchw_local[57] = 0.000000e+00f;
  conv2d_nchw_local[26] = 0.000000e+00f;
  conv2d_nchw_local[58] = 0.000000e+00f;
  conv2d_nchw_local[27] = 0.000000e+00f;
  conv2d_nchw_local[59] = 0.000000e+00f;
  conv2d_nchw_local[28] = 0.000000e+00f;
  conv2d_nchw_local[60] = 0.000000e+00f;
  conv2d_nchw_local[29] = 0.000000e+00f;
  conv2d_nchw_local[61] = 0.000000e+00f;
  conv2d_nchw_local[30] = 0.000000e+00f;
  conv2d_nchw_local[62] = 0.000000e+00f;
  conv2d_nchw_local[31] = 0.000000e+00f;
  conv2d_nchw_local[63] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 3; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[(((int)threadIdx.x) * 4)] = (((2 <= (((int)blockIdx.x) % 224)) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((int)threadIdx.x) * 4)))) ? data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 1)] = ((2 <= (((int)blockIdx.x) % 224)) ? data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) - 224)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 2)] = ((2 <= (((int)blockIdx.x) % 224)) ? data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) - 223)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 3)] = ((2 <= (((int)blockIdx.x) % 224)) ? data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) - 222)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 112)] = ((((1 <= ((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 56) / 57))) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 112) % 114)))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 112) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 56) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 112) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 113)] = ((((1 <= ((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 56) / 57))) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 113) % 114)))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 113) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 56) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 113) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 114)] = ((1 <= (((((int)blockIdx.x) & 1) * 112) + (((int)threadIdx.x) * 4))) ? data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) - 1)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 115)] = data[(((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((int)blockIdx.x) & 1) * 112)) + ((((int)threadIdx.x) * 4) + 1)) - 1)];
    pad_temp_shared[((((int)threadIdx.x) * 4) + 224)] = (((1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 110) % 114))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 110) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 112) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 110) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 225)] = (((1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 111) % 114))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 111) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 112) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 111) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 226)] = (((1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 112) % 114))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 112) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 113) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 112) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 227)] = (((1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 113) % 114))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 113) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 113) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 113) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 336)] = ((((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 168) / 57)) < 225) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 108) % 114)))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 108) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 168) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 108) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 337)] = ((((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 168) / 57)) < 225) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 109) % 114)))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 109) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 168) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 109) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 338)] = ((((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 169) / 57)) < 225) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 110) % 114)))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 110) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 169) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 110) % 114)) - 225)] : 0.000000e+00f);
    pad_temp_shared[((((int)threadIdx.x) * 4) + 339)] = ((((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 169) / 57)) < 225) && (1 <= (((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 111) % 114)))) && ((((((int)blockIdx.x) & 1) * 112) + (((((int)threadIdx.x) * 4) + 111) % 114)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 169) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((((int)threadIdx.x) * 4) + 111) % 114)) - 225)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 2) {
      pad_temp_shared[((((int)threadIdx.x) * 4) + 448)] = ((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 224) / 57)) < 225) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 224) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + ((((int)threadIdx.x) * 4) + 106)) - 225)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 2) {
      pad_temp_shared[((((int)threadIdx.x) * 4) + 449)] = ((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 224) / 57)) < 225) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 224) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + ((((int)threadIdx.x) * 4) + 107)) - 225)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 2) {
      pad_temp_shared[((((int)threadIdx.x) * 4) + 450)] = ((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 225) / 57)) < 225) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 225) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + ((((int)threadIdx.x) * 4) + 108)) - 225)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 2) {
      pad_temp_shared[((((int)threadIdx.x) * 4) + 451)] = (((((((((int)blockIdx.x) % 224) >> 1) * 2) + (((((int)threadIdx.x) * 2) + 225) / 57)) < 225) && ((((((int)blockIdx.x) & 1) * 112) + ((((int)threadIdx.x) * 4) + 109)) < 225)) ? data[((((((rc_outer_outer * 50176) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + ((((((int)threadIdx.x) * 2) + 225) / 57) * 224)) + ((((int)blockIdx.x) & 1) * 112)) + ((((int)threadIdx.x) * 4) + 109)) - 225)] : 0.000000e+00f);
    }
    if (((int)threadIdx.x) < 24) {
      *(float3*)(kernel_shared + (((int)threadIdx.x) * 3)) = *(float3*)(kernel + (((((((int)blockIdx.x) / 224) * 216) + ((((int)threadIdx.x) / 3) * 27)) + (rc_outer_outer * 9)) + ((((int)threadIdx.x) % 3) * 3)));
    }
    __syncthreads();
    for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 4; ++ff_c_outer_inner) {
        conv2d_nchw_local[(ff_c_outer_inner * 8)] = (conv2d_nchw_local[(ff_c_outer_inner * 8)] + (pad_temp_shared[((((int)threadIdx.x) * 4) + rx_outer_inner)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 32)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 32)] + (pad_temp_shared[((((int)threadIdx.x) * 4) + rx_outer_inner)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 1)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 1)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 33)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 33)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 1)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 2)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 2)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 34)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 34)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 2)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 3)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 3)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 35)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 35)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 3)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 4)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 114)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 36)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 36)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 114)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 5)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 115)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 37)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 37)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 115)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 6)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 116)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 38)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 38)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 116)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 7)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 7)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 117)] * kernel_shared[((ff_c_outer_inner * 9) + rx_outer_inner)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 39)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 39)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 117)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 36)]));
        conv2d_nchw_local[(ff_c_outer_inner * 8)] = (conv2d_nchw_local[(ff_c_outer_inner * 8)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 114)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 32)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 32)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 114)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 1)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 115)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 33)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 33)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 115)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 2)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 116)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 34)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 34)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 116)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 3)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 117)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 35)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 35)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 117)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 4)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 228)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 36)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 36)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 228)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 5)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 229)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 37)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 37)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 229)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 6)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 230)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 38)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 38)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 230)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 7)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 7)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 231)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 39)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 39)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 231)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 39)]));
        conv2d_nchw_local[(ff_c_outer_inner * 8)] = (conv2d_nchw_local[(ff_c_outer_inner * 8)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 228)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 32)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 32)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 228)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 1)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 229)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 33)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 33)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 229)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 2)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 2)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 230)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 34)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 34)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 230)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 3)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 3)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 231)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 35)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 35)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 231)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 4)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 4)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 342)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 36)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 36)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 342)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 5)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 5)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 343)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 37)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 37)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 343)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 6)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 6)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 344)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 38)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 38)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 344)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 7)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 7)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 345)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 8) + 39)] = (conv2d_nchw_local[((ff_c_outer_inner * 8) + 39)] + (pad_temp_shared[(((((int)threadIdx.x) * 4) + rx_outer_inner) + 345)] * kernel_shared[(((ff_c_outer_inner * 9) + rx_outer_inner) + 42)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 4; ++ff_inner) {
    for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
      for (int xx_inner = 0; xx_inner < 4; ++xx_inner) {
        conv2d_nchw[((((((((((int)blockIdx.x) / 224) * 401408) + (ff_inner * 50176)) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + (yy_inner * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) + xx_inner)] = conv2d_nchw_local[(((ff_inner * 8) + (yy_inner * 4)) + xx_inner)];
        conv2d_nchw[(((((((((((int)blockIdx.x) / 224) * 401408) + (ff_inner * 50176)) + (((((int)blockIdx.x) % 224) >> 1) * 448)) + (yy_inner * 224)) + ((((int)blockIdx.x) & 1) * 112)) + (((int)threadIdx.x) * 4)) + xx_inner) + 200704)] = conv2d_nchw_local[((((ff_inner * 8) + (yy_inner * 4)) + xx_inner) + 32)];
      }
    }
  }
}


