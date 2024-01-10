
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
extern "C" __global__ void __launch_bounds__(256) candidate3(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ compute, float* __restrict__ bias) {
  float conv2d_nchw[64];
  __shared__ float pad_temp_shared[2304];
  __shared__ float kernel_shared[1536];
  for (int ff_inner_init = 0; ff_inner_init < 16; ++ff_inner_init) {
    conv2d_nchw[(ff_inner_init * 4)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 4) + 1)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 4) + 2)] = 0.000000e+00f;
    conv2d_nchw[((ff_inner_init * 4) + 3)] = 0.000000e+00f;
  }
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    for (int rx_outer_outer = 0; rx_outer_outer < 3; ++rx_outer_outer) {
      __syncthreads();
      pad_temp_shared[((int)threadIdx.x)] = ((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((int)threadIdx.x) >> 4))) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[(((((((rc_outer_outer * 100352) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 256)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 16) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 16) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 256) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 16) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 512)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 14) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 14) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 512) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 14) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 768)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 12) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 12) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 768) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 12) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1024)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 10) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 10) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1024) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 10) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1280)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 8) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 8) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1280) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 8) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1536)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 6) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 6) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1536) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 6) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 1792)] = (((((1 <= (((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 4) % 18))) && ((((((int)blockIdx.x) / 7) * 16) + (((((int)threadIdx.x) >> 4) + 4) % 18)) < 113)) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 1792) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + ((((((int)threadIdx.x) >> 4) + 4) % 18) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      pad_temp_shared[(((int)threadIdx.x) + 2048)] = (((((((((int)blockIdx.x) / 7) * 16) + ((((int)threadIdx.x) >> 4) + 2)) < 113) && (1 <= ((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)))) && (((((((int)blockIdx.x) % 7) * 16) + rx_outer_outer) + (((int)threadIdx.x) & 15)) < 113)) ? data[((((((((rc_outer_outer * 100352) + (((((int)threadIdx.x) + 2048) / 288) * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((int)threadIdx.x) >> 4) + 2) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + rx_outer_outer) + (((int)threadIdx.x) & 15)) - 113)] : 0.000000e+00f);
      kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)threadIdx.x) / 24) * 288) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 256)] = kernel[((((((((int)threadIdx.x) + 256) / 24) * 288) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 512)] = kernel[((((((((int)threadIdx.x) + 512) / 24) * 288) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 768)] = kernel[((((((((int)threadIdx.x) / 24) * 288) + (rc_outer_outer * 72)) + ((((int)threadIdx.x) % 24) * 3)) + rx_outer_outer) + 9216)];
      kernel_shared[(((int)threadIdx.x) + 1024)] = kernel[((((((((int)threadIdx.x) + 1024) / 24) * 288) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 16) % 24) * 3)) + rx_outer_outer)];
      kernel_shared[(((int)threadIdx.x) + 1280)] = kernel[((((((((int)threadIdx.x) + 1280) / 24) * 288) + (rc_outer_outer * 72)) + (((((int)threadIdx.x) + 8) % 24) * 3)) + rx_outer_outer)];
      __syncthreads();
      for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
        for (int rc_inner = 0; rc_inner < 4; ++rc_inner) {
          for (int ry_inner = 0; ry_inner < 3; ++ry_inner) {
            for (int ff_inner = 0; ff_inner < 16; ++ff_inner) {
              conv2d_nchw[(ff_inner * 4)] = (conv2d_nchw[(ff_inner * 4)] + (pad_temp_shared[(((((rc_outer_inner * 1152) + (rc_inner * 288)) + (((((int)threadIdx.x) & 63) >> 4) * 64)) + (ry_inner * 16)) + (((int)threadIdx.x) & 15))] * kernel_shared[((((((((int)threadIdx.x) >> 6) * 384) + (ff_inner * 24)) + (rc_outer_inner * 12)) + (rc_inner * 3)) + ry_inner)]));
              conv2d_nchw[((ff_inner * 4) + 1)] = (conv2d_nchw[((ff_inner * 4) + 1)] + (pad_temp_shared[((((((rc_outer_inner * 1152) + (rc_inner * 288)) + (((((int)threadIdx.x) & 63) >> 4) * 64)) + (ry_inner * 16)) + (((int)threadIdx.x) & 15)) + 16)] * kernel_shared[((((((((int)threadIdx.x) >> 6) * 384) + (ff_inner * 24)) + (rc_outer_inner * 12)) + (rc_inner * 3)) + ry_inner)]));
              conv2d_nchw[((ff_inner * 4) + 2)] = (conv2d_nchw[((ff_inner * 4) + 2)] + (pad_temp_shared[((((((rc_outer_inner * 1152) + (rc_inner * 288)) + (((((int)threadIdx.x) & 63) >> 4) * 64)) + (ry_inner * 16)) + (((int)threadIdx.x) & 15)) + 32)] * kernel_shared[((((((((int)threadIdx.x) >> 6) * 384) + (ff_inner * 24)) + (rc_outer_inner * 12)) + (rc_inner * 3)) + ry_inner)]));
              conv2d_nchw[((ff_inner * 4) + 3)] = (conv2d_nchw[((ff_inner * 4) + 3)] + (pad_temp_shared[((((((rc_outer_inner * 1152) + (rc_inner * 288)) + (((((int)threadIdx.x) & 63) >> 4) * 64)) + (ry_inner * 16)) + (((int)threadIdx.x) & 15)) + 48)] * kernel_shared[((((((((int)threadIdx.x) >> 6) * 384) + (ff_inner * 24)) + (rc_outer_inner * 12)) + (rc_inner * 3)) + ry_inner)]));
            }
          }
        }
      }
    }
  }
  for (int i1_inner = 0; i1_inner < 16; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 4; ++i2_inner) {
      compute[((((((((((int)threadIdx.x) >> 6) * 200704) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (((((int)threadIdx.x) & 63) >> 4) * 448)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15))] = max((conv2d_nchw[((i1_inner * 4) + i2_inner)] + bias[(((((int)threadIdx.x) >> 6) * 16) + i1_inner)]), 0.000000e+00f);
    }
  }
}


