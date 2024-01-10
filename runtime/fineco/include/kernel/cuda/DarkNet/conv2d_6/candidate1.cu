
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
extern "C" __global__ void __launch_bounds__(896) candidate1(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[14];
  __shared__ float pad_temp_shared[3584];
  __shared__ float kernel_shared[3072];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 8; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = (((((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14))) && (((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14)) < 29)) && (1 <= ((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)))) && (((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)) < 29)) ? data[((((((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 224) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) % 224) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + rx_outer_outer) + (((int)threadIdx.x) % 14)) - 29)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 896)] = (((((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14))) && (((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14)) < 29)) && (1 <= ((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)))) && (((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)) < 29)) ? data[((((((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 224) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) % 224) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + rx_outer_outer) + (((int)threadIdx.x) % 14)) + 3107)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14))) && (((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14)) < 29)) && (1 <= ((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)))) && (((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)) < 29)) ? data[((((((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 224) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) % 224) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + rx_outer_outer) + (((int)threadIdx.x) % 14)) + 6243)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2688)] = (((((1 <= ((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14))) && (((((((int)blockIdx.x) & 3) >> 1) * 14) + ((((int)threadIdx.x) % 224) / 14)) < 29)) && (1 <= ((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)))) && (((((((int)blockIdx.x) & 1) * 14) + rx_outer_outer) + (((int)threadIdx.x) % 14)) < 29)) ? data[((((((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 224) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) % 224) / 14) * 28)) + ((((int)blockIdx.x) & 1) * 14)) + rx_outer_outer) + (((int)threadIdx.x) % 14)) + 9379)] : 0.000000e+00f);
      kernel_shared[((int)threadIdx.x)] = kernel[((((((((int)blockIdx.x) >> 2) * 73728) + ((((int)threadIdx.x) / 48) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 896)] = kernel[((((((((int)blockIdx.x) >> 2) * 73728) + (((((int)threadIdx.x) + 896) / 48) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 32) % 48) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1792)] = kernel[((((((((int)blockIdx.x) >> 2) * 73728) + (((((int)threadIdx.x) + 1792) / 48) * 1152)) + (rc_outer_outer * 144)) + (((((int)threadIdx.x) + 16) % 48) * 3)) + rx_outer_outer)];
      if (((int)threadIdx.x) < 384) {
        kernel_shared[(((int)threadIdx.x) + 2688)] = kernel[(((((((((int)blockIdx.x) >> 2) * 73728) + ((((int)threadIdx.x) / 48) * 1152)) + (rc_outer_outer * 144)) + ((((int)threadIdx.x) % 48) * 3)) + rx_outer_outer) + 64512)];
      }
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14))] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14))] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3))]));
        conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1536)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 14)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1)]));
        conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1537)]));
        conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 28)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
        conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 42)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
        conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 56)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
        conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 70)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
        conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 84)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
        conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 98)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
        conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 2)]));
        conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[((((rc_outer_inner * 224) + (((((int)threadIdx.x) % 28) / 14) * 98)) + (((int)threadIdx.x) % 14)) + 112)] * kernel_shared[((((((int)threadIdx.x) / 28) * 48) + (rc_outer_inner * 3)) + 1538)]));
      }
    }
  }
  for (int i2_inner = 0; i2_inner < 7; ++i2_inner) {
    compute[((((((((((int)blockIdx.x) >> 2) * 50176) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14))] = max((conv2d_nchw[i2_inner] + bias[(((((int)blockIdx.x) >> 2) * 64) + (((int)threadIdx.x) / 28))]), 0.000000e+00f);
    compute[(((((((((((int)blockIdx.x) >> 2) * 50176) + ((((int)threadIdx.x) / 28) * 784)) + (((((int)blockIdx.x) & 3) >> 1) * 392)) + (((((int)threadIdx.x) % 28) / 14) * 196)) + (i2_inner * 28)) + ((((int)blockIdx.x) & 1) * 14)) + (((int)threadIdx.x) % 14)) + 25088)] = max((conv2d_nchw[(i2_inner + 7)] + bias[((((((int)blockIdx.x) >> 2) * 64) + (((int)threadIdx.x) / 28)) + 32)]), 0.000000e+00f);
  }
}


