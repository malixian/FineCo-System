
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
extern "C" __global__ void __launch_bounds__(896) candidate2(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[8];
  __shared__ float pad_temp_shared[9280];
  __shared__ float kernel_shared[2304];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) % 580) / 58))) && ((((((int)blockIdx.x) % 7) * 8) + ((((int)threadIdx.x) % 580) / 58)) < 57)) && (1 <= (((int)threadIdx.x) % 58))) && ((((int)threadIdx.x) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + ((((int)threadIdx.x) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((int)threadIdx.x) % 580) / 58) * 56)) + (((int)threadIdx.x) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 158) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 158) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 26) % 58))) && (((((int)threadIdx.x) + 26) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 896) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 158) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 26) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 26) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 26) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 52) % 58))) && (((((int)threadIdx.x) + 52) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 1792) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 26) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 52) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 2688)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 184) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 184) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 20) % 58))) && (((((int)threadIdx.x) + 20) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 2688) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 184) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 20) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 52) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 52) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 46) % 58))) && (((((int)threadIdx.x) + 46) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 3584) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 52) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 46) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 4480)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 210) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 210) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 14) % 58))) && (((((int)threadIdx.x) + 14) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 4480) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 210) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 14) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 5376)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 78) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 78) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 40) % 58))) && (((((int)threadIdx.x) + 40) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 5376) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 78) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 40) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 6272)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 236) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 236) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 8) % 58))) && (((((int)threadIdx.x) + 8) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 6272) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 236) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 8) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 7168)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 104) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 104) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 34) % 58))) && (((((int)threadIdx.x) + 34) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 7168) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 104) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 34) % 58)) - 57)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 8064)] = (((((1 <= (((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 262) % 290) / 29))) && ((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 262) % 290) / 29)) < 57)) && (1 <= ((((int)threadIdx.x) + 2) % 58))) && (((((int)threadIdx.x) + 2) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 8064) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 262) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 2) % 58)) - 57)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 320) {
      pad_temp_shared[(((int)threadIdx.x) + 8960)] = (((((((((int)blockIdx.x) % 7) * 8) + ((((((int)threadIdx.x) >> 1) + 130) % 290) / 29)) < 57) && (1 <= ((((int)threadIdx.x) + 28) % 58))) && (((((int)threadIdx.x) + 28) % 58) < 57)) ? data[((((((rc_outer_outer * 50176) + (((((int)threadIdx.x) + 8960) / 580) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + (((((((int)threadIdx.x) >> 1) + 130) % 290) / 29) * 56)) + ((((int)threadIdx.x) + 28) % 58)) - 57)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 9216) + ((((int)threadIdx.x) / 144) * 576)) + (rc_outer_outer * 144)) + (((int)threadIdx.x) % 144))];
    kernel_shared[(((int)threadIdx.x) + 896)] = kernel[(((((((int)blockIdx.x) / 7) * 9216) + (((((int)threadIdx.x) + 896) / 144) * 576)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 32) % 144))];
    if (((int)threadIdx.x) < 512) {
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[(((((((int)blockIdx.x) / 7) * 9216) + (((((int)threadIdx.x) + 1792) / 144) * 576)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) + 64) % 144))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 8; ++rc_outer_inner) {
      for (int ry_outer_inner = 0; ry_outer_inner < 3; ++ry_outer_inner) {
        for (int xx_c_outer_inner = 0; xx_c_outer_inner < 4; ++xx_c_outer_inner) {
          conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner)] * kernel_shared[((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3))]));
          conv2d_nchw_local[(xx_c_outer_inner + 4)] = (conv2d_nchw_local[(xx_c_outer_inner + 4)] + (pad_temp_shared[(((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1152)]));
          conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1)]));
          conv2d_nchw_local[(xx_c_outer_inner + 4)] = (conv2d_nchw_local[(xx_c_outer_inner + 4)] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1153)]));
          conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 2)]));
          conv2d_nchw_local[(xx_c_outer_inner + 4)] = (conv2d_nchw_local[(xx_c_outer_inner + 4)] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 2)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1154)]));
          conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 580)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 9)]));
          conv2d_nchw_local[(xx_c_outer_inner + 4)] = (conv2d_nchw_local[(xx_c_outer_inner + 4)] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 580)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1161)]));
          conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 581)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 10)]));
          conv2d_nchw_local[(xx_c_outer_inner + 4)] = (conv2d_nchw_local[(xx_c_outer_inner + 4)] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 581)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1162)]));
          conv2d_nchw_local[xx_c_outer_inner] = (conv2d_nchw_local[xx_c_outer_inner] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 582)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 11)]));
          conv2d_nchw_local[(xx_c_outer_inner + 4)] = (conv2d_nchw_local[(xx_c_outer_inner + 4)] + (pad_temp_shared[((((((rc_outer_inner * 1160) + (((((int)threadIdx.x) % 112) / 14) * 58)) + (ry_outer_inner * 58)) + ((((int)threadIdx.x) % 14) * 4)) + xx_c_outer_inner) + 582)] * kernel_shared[(((((((int)threadIdx.x) / 112) * 144) + (rc_outer_inner * 18)) + (ry_outer_inner * 3)) + 1163)]));
        }
      }
    }
  }
  for (int xx_inner = 0; xx_inner < 4; ++xx_inner) {
    conv2d_nchw[((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + ((((int)threadIdx.x) % 112) * 4)) + xx_inner)] = conv2d_nchw_local[xx_inner];
    conv2d_nchw[(((((((((int)blockIdx.x) / 7) * 50176) + ((((int)threadIdx.x) / 112) * 3136)) + ((((int)blockIdx.x) % 7) * 448)) + ((((int)threadIdx.x) % 112) * 4)) + xx_inner) + 25088)] = conv2d_nchw_local[(xx_inner + 4)];
  }
}


