
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
extern "C" __global__ void __launch_bounds__(128) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[2560];
  __shared__ float kernel_shared[3072];
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
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer < 20; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer) {
        pad_temp_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 128) + ((int)threadIdx.x))] = (((((1 <= ((((((int)blockIdx.x) % 49) / 7) * 8) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16) + (((int)threadIdx.x) >> 3)) % 10))) && (((((((int)blockIdx.x) % 49) / 7) * 8) + (((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16) + (((int)threadIdx.x) >> 3)) % 10)) < 57)) && (1 <= ((((((int)blockIdx.x) % 7) * 8) + rx_outer_outer) + (((int)threadIdx.x) & 7)))) && (((((((int)blockIdx.x) % 7) * 8) + rx_outer_outer) + (((int)threadIdx.x) & 7)) < 57)) ? data[((((((((rc_outer_outer * 100352) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 8) + (((int)threadIdx.x) >> 4)) / 5) * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer * 16) + (((int)threadIdx.x) >> 3)) % 10) * 56)) + ((((int)blockIdx.x) % 7) * 8)) + rx_outer_outer) + (((int)threadIdx.x) & 7)) - 57)] : 0.000000e+00f);
      }
      for (int ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 = 0; ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 < 24; ++ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1) {
        kernel_shared[((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 128) + ((int)threadIdx.x))] = kernel[((((((((int)blockIdx.x) / 49) * 18432) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 4) + (((int)threadIdx.x) >> 5)) / 3) * 576)) + (rc_outer_outer * 288)) + ((((ax0_ax1_fused_ax2_fused_ax3_fused_outer_outer1 * 128) + ((int)threadIdx.x)) % 96) * 3)) + rx_outer_outer)];
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
            conv2d_nchw_local[0] = (conv2d_nchw_local[0] + (pad_temp_shared[((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[2] = (conv2d_nchw_local[2] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[4] = (conv2d_nchw_local[4] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[6] = (conv2d_nchw_local[6] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[8] = (conv2d_nchw_local[8] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[10] = (conv2d_nchw_local[10] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[12] = (conv2d_nchw_local[12] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[14] = (conv2d_nchw_local[14] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[(((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner)]));
            conv2d_nchw_local[1] = (conv2d_nchw_local[1] + (pad_temp_shared[((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7))] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[3] = (conv2d_nchw_local[3] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 8)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[5] = (conv2d_nchw_local[5] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 16)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[7] = (conv2d_nchw_local[7] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 24)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[9] = (conv2d_nchw_local[9] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 32)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[11] = (conv2d_nchw_local[11] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 40)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[13] = (conv2d_nchw_local[13] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 48)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
            conv2d_nchw_local[15] = (conv2d_nchw_local[15] + (pad_temp_shared[(((((rc_outer_inner * 1280) + (rc_inner * 80)) + (ry_inner * 8)) + (((int)threadIdx.x) & 7)) + 56)] * kernel_shared[((((((((int)threadIdx.x) >> 3) * 192) + (rc_outer_inner * 48)) + (rc_inner * 3)) + ry_inner) + 96)]));
          }
        }
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 2; ++ff_inner) {
    conv2d_nchw[(((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7))] = conv2d_nchw_local[ff_inner];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 56)] = conv2d_nchw_local[(ff_inner + 2)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 112)] = conv2d_nchw_local[(ff_inner + 4)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 168)] = conv2d_nchw_local[(ff_inner + 6)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 224)] = conv2d_nchw_local[(ff_inner + 8)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 280)] = conv2d_nchw_local[(ff_inner + 10)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 336)] = conv2d_nchw_local[(ff_inner + 12)];
    conv2d_nchw[((((((((((int)blockIdx.x) / 49) * 100352) + ((((int)threadIdx.x) >> 3) * 6272)) + (ff_inner * 3136)) + (((((int)blockIdx.x) % 49) / 7) * 448)) + ((((int)blockIdx.x) % 7) * 8)) + (((int)threadIdx.x) & 7)) + 392)] = conv2d_nchw_local[(ff_inner + 14)];
  }
}


