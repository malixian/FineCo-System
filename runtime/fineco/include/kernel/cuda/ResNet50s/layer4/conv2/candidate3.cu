
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
extern "C" __global__ void __launch_bounds__(224) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[7];
  __shared__ float pad_temp_shared[2048];
  __shared__ float kernel_shared[576];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 32; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = ((((16 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[((((rc_outer_outer * 1568) + ((((int)threadIdx.x) >> 4) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 224)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 14) & 15)) && ((((((int)threadIdx.x) >> 4) + 14) & 15) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 224) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 14) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 448)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 12) & 15)) && ((((((int)threadIdx.x) >> 4) + 12) & 15) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 448) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 12) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 672)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 10) & 15)) && ((((((int)threadIdx.x) >> 4) + 10) & 15) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 672) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 10) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 8) & 15)) && ((((((int)threadIdx.x) >> 4) + 8) & 15) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 896) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 8) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1120)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 6) & 15)) && ((((((int)threadIdx.x) >> 4) + 6) & 15) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1120) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 6) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1344)] = (((((1 <= (((((int)threadIdx.x) >> 4) + 4) & 15)) && ((((((int)threadIdx.x) >> 4) + 4) & 15) < 15)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1344) >> 8) * 196)) + ((((((int)threadIdx.x) >> 4) + 4) & 15) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = ((((((int)threadIdx.x) < 208) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((((rc_outer_outer * 1568) + (((((int)threadIdx.x) + 1568) >> 8) * 196)) + (((((int)threadIdx.x) >> 4) + 2) * 14)) + (((int)threadIdx.x) & 15)) - 15)] : 0.000000e+00f);
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = ((((16 <= ((int)threadIdx.x)) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[((((rc_outer_outer * 1568) + ((((int)threadIdx.x) >> 4) * 14)) + (((int)threadIdx.x) & 15)) + 1357)] : 0.000000e+00f);
    if (((int)threadIdx.x) < 32) {
      pad_temp_shared[(((int)threadIdx.x) + 2016)] = ((((((int)threadIdx.x) < 16) && (1 <= (((int)threadIdx.x) & 15))) && ((((int)threadIdx.x) & 15) < 15)) ? data[(((rc_outer_outer * 1568) + ((int)threadIdx.x)) + 1553)] : 0.000000e+00f);
    }
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)blockIdx.x) * 18432) + ((((int)threadIdx.x) / 72) * 2304)) + (rc_outer_outer * 72)) + (((int)threadIdx.x) % 72))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((int)blockIdx.x) * 18432) + (((((int)threadIdx.x) + 224) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 8) % 72))];
    if (((int)threadIdx.x) < 128) {
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((int)blockIdx.x) * 18432) + (((((int)threadIdx.x) + 448) / 72) * 2304)) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) + 16) % 72))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int rx_outer_inner = 0; rx_outer_inner < 3; ++rx_outer_inner) {
        for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[(((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 16)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 32)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 48)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 64)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 80)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 96)] * kernel_shared[(((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 16)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 32)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 48)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 64)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 80)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 96)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 3)]));
          conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 32)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
          conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 48)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
          conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 64)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
          conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 80)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
          conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 96)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
          conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
          conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[((((((rc_outer_inner * 1024) + (rc_inner * 256)) + (((((int)threadIdx.x) % 28) / 14) * 112)) + rx_outer_inner) + (((int)threadIdx.x) % 14)) + 128)] * kernel_shared[((((((((int)threadIdx.x) / 28) * 72) + (rc_outer_inner * 36)) + (rc_inner * 9)) + rx_outer_inner) + 6)]));
        }
      }
    }
  }
  for (int yy_inner = 0; yy_inner < 7; ++yy_inner) {
    conv2d_nchw[((((((int)blockIdx.x) * 1568) + ((((int)threadIdx.x) / 14) * 98)) + (yy_inner * 14)) + (((int)threadIdx.x) % 14))] = conv2d_nchw_local[yy_inner];
  }
}


