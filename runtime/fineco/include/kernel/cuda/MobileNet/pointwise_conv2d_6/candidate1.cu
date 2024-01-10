
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
extern "C" __global__ void __launch_bounds__(392) candidate1(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[16];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[4096];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98))];
    pad_temp_shared[(((int)threadIdx.x) + 392)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 784)];
    pad_temp_shared[(((int)threadIdx.x) + 784)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 1568)];
    pad_temp_shared[(((int)threadIdx.x) + 1176)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 2352)];
    pad_temp_shared[(((int)threadIdx.x) + 1568)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 3136)];
    pad_temp_shared[(((int)threadIdx.x) + 1960)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 3920)];
    pad_temp_shared[(((int)threadIdx.x) + 2352)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 4704)];
    pad_temp_shared[(((int)threadIdx.x) + 2744)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 5488)];
    pad_temp_shared[(((int)threadIdx.x) + 3136)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 6272)];
    pad_temp_shared[(((int)threadIdx.x) + 3528)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 7056)];
    pad_temp_shared[(((int)threadIdx.x) + 3920)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 7840)];
    pad_temp_shared[(((int)threadIdx.x) + 4312)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 8624)];
    pad_temp_shared[(((int)threadIdx.x) + 4704)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 9408)];
    pad_temp_shared[(((int)threadIdx.x) + 5096)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 10192)];
    pad_temp_shared[(((int)threadIdx.x) + 5488)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 10976)];
    pad_temp_shared[(((int)threadIdx.x) + 5880)] = Input[(((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 98) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + (((int)threadIdx.x) % 98)) + 11760)];
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 392) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 8) & 63))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 784) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 16) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1176)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 1176) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 24) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 1568) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1960)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 1960) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 40) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2352)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 2352) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 48) & 63))];
    kernel_shared[(((int)threadIdx.x) + 2744)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 2744) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 56) & 63))];
    kernel_shared[(((int)threadIdx.x) + 3136)] = kernel[((((((((int)blockIdx.x) >> 1) * 16384) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63)) + 12544)];
    kernel_shared[(((int)threadIdx.x) + 3528)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 3528) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 8) & 63))];
    if (((int)threadIdx.x) < 176) {
      kernel_shared[(((int)threadIdx.x) + 3920)] = kernel[(((((((int)blockIdx.x) >> 1) * 16384) + (((((int)threadIdx.x) + 3920) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 16) & 63))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 4; ++rc_outer_inner) {
      for (int ff_outer_inner = 0; ff_outer_inner < 2; ++ff_outer_inner) {
        for (int rc_inner = 0; rc_inner < 16; ++rc_inner) {
          conv2d_nchw[(ff_outer_inner * 8)] = (conv2d_nchw[(ff_outer_inner * 8)] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2))] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner)]));
          conv2d_nchw[((ff_outer_inner * 8) + 1)] = (conv2d_nchw[((ff_outer_inner * 8) + 1)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 1)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner)]));
          conv2d_nchw[((ff_outer_inner * 8) + 2)] = (conv2d_nchw[((ff_outer_inner * 8) + 2)] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner) + 64)]));
          conv2d_nchw[((ff_outer_inner * 8) + 3)] = (conv2d_nchw[((ff_outer_inner * 8) + 3)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner) + 64)]));
          conv2d_nchw[((ff_outer_inner * 8) + 4)] = (conv2d_nchw[((ff_outer_inner * 8) + 4)] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner) + 128)]));
          conv2d_nchw[((ff_outer_inner * 8) + 5)] = (conv2d_nchw[((ff_outer_inner * 8) + 5)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner) + 128)]));
          conv2d_nchw[((ff_outer_inner * 8) + 6)] = (conv2d_nchw[((ff_outer_inner * 8) + 6)] + (pad_temp_shared[(((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2))] * kernel_shared[((((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner) + 192)]));
          conv2d_nchw[((ff_outer_inner * 8) + 7)] = (conv2d_nchw[((ff_outer_inner * 8) + 7)] + (pad_temp_shared[((((rc_outer_inner * 1568) + (rc_inner * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 1)] * kernel_shared[((((((((int)threadIdx.x) / 49) * 512) + (ff_outer_inner * 256)) + (rc_outer_inner * 16)) + rc_inner) + 192)]));
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 8; ++i1_inner) {
    for (int i3_inner = 0; i3_inner < 2; ++i3_inner) {
      compute[(((((((((int)blockIdx.x) >> 1) * 12544) + ((((int)threadIdx.x) / 49) * 1568)) + (i1_inner * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + i3_inner)] = max(conv2d_nchw[((i1_inner * 2) + i3_inner)], 0.000000e+00f);
    }
  }
}


