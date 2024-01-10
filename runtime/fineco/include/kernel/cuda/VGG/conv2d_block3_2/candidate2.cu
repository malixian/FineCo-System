
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
extern "C" __global__ void __launch_bounds__(256) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[112];
  __shared__ float pad_temp_shared[2320];
  __shared__ float kernel_shared[2304];
  for (int yy_c_inner_init = 0; yy_c_inner_init < 2; ++yy_c_inner_init) {
    for (int xx_c_inner_init = 0; xx_c_inner_init < 7; ++xx_c_inner_init) {
      conv2d_nchw_local[((yy_c_inner_init * 7) + xx_c_inner_init)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 14)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 28)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 42)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 56)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 70)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 84)] = 0.000000e+00f;
      conv2d_nchw_local[(((yy_c_inner_init * 7) + xx_c_inner_init) + 98)] = 0.000000e+00f;
    }
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 64; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) % 7) * 8) + (((int)threadIdx.x) / 58))) && (1 <= (((int)threadIdx.x) % 58))) && ((((int)threadIdx.x) % 58) < 57)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) % 7) * 448)) + ((((int)threadIdx.x) / 58) * 56)) + (((int)threadIdx.x) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 256)] = (((1 <= ((((int)threadIdx.x) + 24) % 58)) && (((((int)threadIdx.x) + 24) % 58) < 57)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) % 7) * 448)) + (((((int)threadIdx.x) + 256) / 58) * 56)) + ((((int)threadIdx.x) + 24) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 256) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 256) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 48) % 58))) && (((((int)threadIdx.x) + 48) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 512) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 256) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 48) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 768)] = (((1 <= ((((int)threadIdx.x) + 14) % 58)) && (((((int)threadIdx.x) + 14) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 768) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 94) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 14) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 222) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 222) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 38) % 58))) && (((((int)threadIdx.x) + 38) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1024) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 222) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 38) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((1 <= ((((int)threadIdx.x) + 4) % 58)) && (((((int)threadIdx.x) + 4) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1280) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 60) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 4) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 188) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 188) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 28) % 58))) && (((((int)threadIdx.x) + 28) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1536) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 188) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 28) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = ((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 26) % 290) / 29))) && (1 <= ((((int)threadIdx.x) + 52) % 58))) && (((((int)threadIdx.x) + 52) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 1792) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 26) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 52) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 154) % 290) / 29)) < 57) && (1 <= ((((int)threadIdx.x) + 18) % 58))) && (((((int)threadIdx.x) + 18) % 58) < 57)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2048) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 154) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 18) % 58)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 16) {
      pad_temp_shared[(((int)threadIdx.x) + 2304)] = ((((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 282) % 290) / 29)) < 57) && (((int)threadIdx.x) < 15)) ? data[((((((rc_outer_outer * 12544) + (((((int)threadIdx.x) + 2304) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 282) % 290) / 29) * 56)) + (((int)threadIdx.x) + 42)) - 57)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + ((((int)threadIdx.x) / 36) * 2304)) + (rc_outer_outer * 36)) + (((int)threadIdx.x) % 36))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 256) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 4) % 36))];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 512) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 8) % 36))];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 768) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 12) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1024) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 16) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1280) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 20) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1536)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1536) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 24) % 36))];
    kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 1792) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 28) % 36))];
    kernel_shared[(((int)threadIdx.x) + 2048)] = kernel[(((((((int)blockIdx.x) / 7) * 147456) + (((((int)threadIdx.x) + 2048) / 36) * 2304)) + (rc_outer_outer * 36)) + ((((int)threadIdx.x) + 32) % 36))];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        for (int rc_inner = 0; rc_inner < 2; ++rc_inner) {
          for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
            for (int yy_c_inner = 0; yy_c_inner < 2; ++yy_c_inner) {
              for (int xx_c_inner = 0; xx_c_inner < 7; ++xx_c_inner) {
                conv2d_nchw_local[((yy_c_inner * 7) + xx_c_inner)] = (conv2d_nchw_local[((yy_c_inner * 7) + xx_c_inner)] + (pad_temp_shared[(((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 14)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 14)] + (pad_temp_shared[((((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner) + 116)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 28)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 28)] + (pad_temp_shared[((((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner) + 232)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 42)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 42)] + (pad_temp_shared[((((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner) + 348)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 56)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 56)] + (pad_temp_shared[(((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 1152)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 70)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 70)] + (pad_temp_shared[((((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner) + 116)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 1152)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 84)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 84)] + (pad_temp_shared[((((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner) + 232)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 1152)]));
                conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 98)] = (conv2d_nchw_local[(((yy_c_inner * 7) + xx_c_inner) + 98)] + (pad_temp_shared[((((((((rc_outer_inner * 1160) + (rc_inner * 580)) + (yy_c_inner * 58)) + (ry_inner * 58)) + ((((int)threadIdx.x) & 7) * 7)) + xx_c_inner) + rx_outer_inner) + 348)] * kernel_shared[(((((((((int)threadIdx.x) >> 3) * 36) + (rc_outer_inner * 18)) + (rc_inner * 9)) + (ry_inner * 3)) + rx_outer_inner) + 1152)]));
              }
            }
          }
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 2; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 7; ++xx_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 7) + xx_inner)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 112)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 14)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 224)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 28)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 336)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 42)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 100352)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 56)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 100464)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 70)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 100576)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 84)];
      conv2d_nchw[((((((((((int)blockIdx.x) / 7) * 200704) + ((((int)threadIdx.x) >> 3) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (yy_inner * 56)) + ((((int)threadIdx.x) & 7) * 7)) + xx_inner) + 100688)] = conv2d_nchw_local[(((yy_inner * 7) + xx_inner) + 98)];
    }
  }
}


