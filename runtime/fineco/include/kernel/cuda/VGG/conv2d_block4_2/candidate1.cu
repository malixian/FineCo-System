
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
extern "C" __global__ void __launch_bounds__(448) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[56];
  __shared__ float pad_temp_shared[7168];
  __shared__ float kernel_shared[3072];
  for (int yy_c_inner_init = 0; yy_c_inner_init < 7; ++yy_c_inner_init) {
    conv2d_nchw_local[(yy_c_inner_init * 2)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 14)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 28)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 42)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 1)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 15)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 29)] = 0.000000e+00f;
    conv2d_nchw_local[((yy_c_inner_init * 2) + 43)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) - 29)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 755)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 1539)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1344)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 2323)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 3107)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2240)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 3891)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2688)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 4675)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 3136)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 5459)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 3584)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 6243)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4032)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 7027)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4480)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 7811)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 4928)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 8595)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 5376)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 9379)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 5824)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 10163)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 6272)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 10947)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 6720)] = (((((1 <= (((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28))) && ((((((int)blockIdx.x) & 1) * 14) + (((int)threadIdx.x) / 28)) < 29)) && (1 <= (rx_outer_outer + (((int)threadIdx.x) % 28)))) && ((rx_outer_outer + (((int)threadIdx.x) % 28)) < 29)) ? data[(((((rc_outer_outer * 12544) + ((((int)blockIdx.x) & 1) * 392)) + rx_outer_outer) + ((int)threadIdx.x)) + 11731)] : 0.000000e+00f);
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) >> 1) * 294912) + ((((int)threadIdx.x) / 48) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) >> 1) * 294912) + (((((int)threadIdx.x) + 448) / 48) * 4608)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 1) * 294912) + (((((int)threadIdx.x) + 896) / 48) * 4608)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1344)] = kernel[(((((((((int)blockIdx.x) >> 1) * 294912) + ((((int)threadIdx.x) / 48) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 129024)];
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 1) * 294912) + (((((int)threadIdx.x) + 1792) / 48) * 4608)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 2240)] = kernel[((((((((int)blockIdx.x) >> 1) * 294912) + (((((int)threadIdx.x) + 2240) / 48) * 4608)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      if (((int)threadIdx.x) < 384) {
        kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((((int)blockIdx.x) >> 1) * 294912) + ((((int)threadIdx.x) / 48) * 4608)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 258048)];
      }
      __syncthreads();
      for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
        for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
          for (int yy_c_inner = 0; yy_c_inner < 7; ++yy_c_inner) {
            conv2d_nchw_local[(yy_c_inner * 2)] = (conv2d_nchw_local[(yy_c_inner * 2)] + (pad_temp_shared[(((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 14)] = (conv2d_nchw_local[((yy_c_inner * 2) + 14)] + (pad_temp_shared[(((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner) + 768)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 28)] = (conv2d_nchw_local[((yy_c_inner * 2) + 28)] + (pad_temp_shared[(((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner) + 1536)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 42)] = (conv2d_nchw_local[((yy_c_inner * 2) + 42)] + (pad_temp_shared[(((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner) + 2304)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 1)] = (conv2d_nchw_local[((yy_c_inner * 2) + 1)] + (pad_temp_shared[((((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 15)] = (conv2d_nchw_local[((yy_c_inner * 2) + 15)] + (pad_temp_shared[((((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner) + 768)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 29)] = (conv2d_nchw_local[((yy_c_inner * 2) + 29)] + (pad_temp_shared[((((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner) + 1536)]));
            conv2d_nchw_local[((yy_c_inner * 2) + 43)] = (conv2d_nchw_local[((yy_c_inner * 2) + 43)] + (pad_temp_shared[((((((rc_inner * 448) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_c_inner * 28)) + (ry_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 48) + (rc_inner * 3)) + ry_inner) + 2304)]));
          }
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nchw[((((((((((int)blockIdx.x) >> 1) * 50176) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + xx_inner)] = conv2d_nchw_local[((yy_inner * 2) + xx_inner)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 1) * 50176) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + xx_inner) + 12544)] = conv2d_nchw_local[(((yy_inner * 2) + xx_inner) + 14)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 1) * 50176) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + xx_inner) + 25088)] = conv2d_nchw_local[(((yy_inner * 2) + xx_inner) + 28)];
      conv2d_nchw[(((((((((((int)blockIdx.x) >> 1) * 50176) + ((((int)threadIdx.x) / 28) * 784)) + ((((int)blockIdx.x) & 1) * 392)) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (yy_inner * 28)) + ((((int)threadIdx.x) % 14) * 2)) + xx_inner) + 37632)] = conv2d_nchw_local[(((yy_inner * 2) + xx_inner) + 42)];
    }
  }
}


