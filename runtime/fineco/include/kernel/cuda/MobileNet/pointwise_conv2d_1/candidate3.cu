
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
extern "C" __global__ void __launch_bounds__(256) candidate3(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[64];
  __shared__ float pad_temp_shared[4096];
  __shared__ float kernel_shared[1024];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[16] = 0.000000e+00f;
  conv2d_nchw[32] = 0.000000e+00f;
  conv2d_nchw[48] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[17] = 0.000000e+00f;
  conv2d_nchw[33] = 0.000000e+00f;
  conv2d_nchw[49] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[18] = 0.000000e+00f;
  conv2d_nchw[34] = 0.000000e+00f;
  conv2d_nchw[50] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  conv2d_nchw[19] = 0.000000e+00f;
  conv2d_nchw[35] = 0.000000e+00f;
  conv2d_nchw[51] = 0.000000e+00f;
  conv2d_nchw[4] = 0.000000e+00f;
  conv2d_nchw[20] = 0.000000e+00f;
  conv2d_nchw[36] = 0.000000e+00f;
  conv2d_nchw[52] = 0.000000e+00f;
  conv2d_nchw[5] = 0.000000e+00f;
  conv2d_nchw[21] = 0.000000e+00f;
  conv2d_nchw[37] = 0.000000e+00f;
  conv2d_nchw[53] = 0.000000e+00f;
  conv2d_nchw[6] = 0.000000e+00f;
  conv2d_nchw[22] = 0.000000e+00f;
  conv2d_nchw[38] = 0.000000e+00f;
  conv2d_nchw[54] = 0.000000e+00f;
  conv2d_nchw[7] = 0.000000e+00f;
  conv2d_nchw[23] = 0.000000e+00f;
  conv2d_nchw[39] = 0.000000e+00f;
  conv2d_nchw[55] = 0.000000e+00f;
  conv2d_nchw[8] = 0.000000e+00f;
  conv2d_nchw[24] = 0.000000e+00f;
  conv2d_nchw[40] = 0.000000e+00f;
  conv2d_nchw[56] = 0.000000e+00f;
  conv2d_nchw[9] = 0.000000e+00f;
  conv2d_nchw[25] = 0.000000e+00f;
  conv2d_nchw[41] = 0.000000e+00f;
  conv2d_nchw[57] = 0.000000e+00f;
  conv2d_nchw[10] = 0.000000e+00f;
  conv2d_nchw[26] = 0.000000e+00f;
  conv2d_nchw[42] = 0.000000e+00f;
  conv2d_nchw[58] = 0.000000e+00f;
  conv2d_nchw[11] = 0.000000e+00f;
  conv2d_nchw[27] = 0.000000e+00f;
  conv2d_nchw[43] = 0.000000e+00f;
  conv2d_nchw[59] = 0.000000e+00f;
  conv2d_nchw[12] = 0.000000e+00f;
  conv2d_nchw[28] = 0.000000e+00f;
  conv2d_nchw[44] = 0.000000e+00f;
  conv2d_nchw[60] = 0.000000e+00f;
  conv2d_nchw[13] = 0.000000e+00f;
  conv2d_nchw[29] = 0.000000e+00f;
  conv2d_nchw[45] = 0.000000e+00f;
  conv2d_nchw[61] = 0.000000e+00f;
  conv2d_nchw[14] = 0.000000e+00f;
  conv2d_nchw[30] = 0.000000e+00f;
  conv2d_nchw[46] = 0.000000e+00f;
  conv2d_nchw[62] = 0.000000e+00f;
  conv2d_nchw[15] = 0.000000e+00f;
  conv2d_nchw[31] = 0.000000e+00f;
  conv2d_nchw[47] = 0.000000e+00f;
  conv2d_nchw[63] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 2; ++rc_outer_outer) {
    __syncthreads();
    pad_temp_shared[((int)threadIdx.x)] = Input[(((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15))];
    pad_temp_shared[(((int)threadIdx.x) + 256)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 12544)];
    pad_temp_shared[(((int)threadIdx.x) + 512)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 25088)];
    pad_temp_shared[(((int)threadIdx.x) + 768)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 37632)];
    pad_temp_shared[(((int)threadIdx.x) + 1024)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 50176)];
    pad_temp_shared[(((int)threadIdx.x) + 1280)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 62720)];
    pad_temp_shared[(((int)threadIdx.x) + 1536)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 75264)];
    pad_temp_shared[(((int)threadIdx.x) + 1792)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 87808)];
    pad_temp_shared[(((int)threadIdx.x) + 2048)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 100352)];
    pad_temp_shared[(((int)threadIdx.x) + 2304)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 112896)];
    pad_temp_shared[(((int)threadIdx.x) + 2560)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 125440)];
    pad_temp_shared[(((int)threadIdx.x) + 2816)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 137984)];
    pad_temp_shared[(((int)threadIdx.x) + 3072)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 150528)];
    pad_temp_shared[(((int)threadIdx.x) + 3328)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 163072)];
    pad_temp_shared[(((int)threadIdx.x) + 3584)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 175616)];
    pad_temp_shared[(((int)threadIdx.x) + 3840)] = Input[((((((rc_outer_outer * 200704) + ((((int)blockIdx.x) / 7) * 1792)) + ((((int)threadIdx.x) >> 4) * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 188160)];
    kernel_shared[((int)threadIdx.x)] = kernel[((((((int)threadIdx.x) >> 4) * 32) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15))];
    kernel_shared[(((int)threadIdx.x) + 256)] = kernel[(((((((int)threadIdx.x) >> 4) * 32) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 512)];
    kernel_shared[(((int)threadIdx.x) + 512)] = kernel[(((((((int)threadIdx.x) >> 4) * 32) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 1024)];
    kernel_shared[(((int)threadIdx.x) + 768)] = kernel[(((((((int)threadIdx.x) >> 4) * 32) + (rc_outer_outer * 16)) + (((int)threadIdx.x) & 15)) + 1536)];
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 16; ++rc_outer_inner) {
      conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 15))] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[16] = (conv2d_nchw[16] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 128)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[32] = (conv2d_nchw[32] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 15))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[48] = (conv2d_nchw[48] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 16)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[17] = (conv2d_nchw[17] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 144)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[33] = (conv2d_nchw[33] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[49] = (conv2d_nchw[49] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 32)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[18] = (conv2d_nchw[18] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 160)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[34] = (conv2d_nchw[34] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[50] = (conv2d_nchw[50] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 48)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[19] = (conv2d_nchw[19] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 176)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[35] = (conv2d_nchw[35] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[51] = (conv2d_nchw[51] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[4] = (conv2d_nchw[4] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 64)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[20] = (conv2d_nchw[20] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 192)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[36] = (conv2d_nchw[36] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[52] = (conv2d_nchw[52] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[5] = (conv2d_nchw[5] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 80)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[21] = (conv2d_nchw[21] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 208)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[37] = (conv2d_nchw[37] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[53] = (conv2d_nchw[53] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[6] = (conv2d_nchw[6] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 96)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[22] = (conv2d_nchw[22] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 224)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[38] = (conv2d_nchw[38] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[54] = (conv2d_nchw[54] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[7] = (conv2d_nchw[7] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 112)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[23] = (conv2d_nchw[23] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 240)] * kernel_shared[(((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner)]));
      conv2d_nchw[39] = (conv2d_nchw[39] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[55] = (conv2d_nchw[55] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 512)]));
      conv2d_nchw[8] = (conv2d_nchw[8] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 15))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[24] = (conv2d_nchw[24] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[40] = (conv2d_nchw[40] + (pad_temp_shared[((rc_outer_inner * 256) + (((int)threadIdx.x) & 15))] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[56] = (conv2d_nchw[56] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 128)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[9] = (conv2d_nchw[9] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[25] = (conv2d_nchw[25] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[41] = (conv2d_nchw[41] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 16)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[57] = (conv2d_nchw[57] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 144)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[10] = (conv2d_nchw[10] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[26] = (conv2d_nchw[26] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[42] = (conv2d_nchw[42] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 32)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[58] = (conv2d_nchw[58] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 160)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[11] = (conv2d_nchw[11] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[27] = (conv2d_nchw[27] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[43] = (conv2d_nchw[43] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 48)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[59] = (conv2d_nchw[59] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 176)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[12] = (conv2d_nchw[12] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[28] = (conv2d_nchw[28] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[44] = (conv2d_nchw[44] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 64)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[60] = (conv2d_nchw[60] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 192)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[13] = (conv2d_nchw[13] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[29] = (conv2d_nchw[29] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[45] = (conv2d_nchw[45] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 80)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[61] = (conv2d_nchw[61] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 208)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[14] = (conv2d_nchw[14] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[30] = (conv2d_nchw[30] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[46] = (conv2d_nchw[46] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 96)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[62] = (conv2d_nchw[62] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 224)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[15] = (conv2d_nchw[15] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[31] = (conv2d_nchw[31] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 16)]));
      conv2d_nchw[47] = (conv2d_nchw[47] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 112)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
      conv2d_nchw[63] = (conv2d_nchw[63] + (pad_temp_shared[(((rc_outer_inner * 256) + (((int)threadIdx.x) & 15)) + 240)] * kernel_shared[((((((int)threadIdx.x) >> 4) * 32) + rc_outer_inner) + 528)]));
    }
  }
  for (int i1_inner = 0; i1_inner < 2; ++i1_inner) {
    for (int i2_inner = 0; i2_inner < 8; ++i2_inner) {
      compute[(((((((((int)threadIdx.x) >> 4) * 25088) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15))] = max(conv2d_nchw[((i1_inner * 8) + i2_inner)], 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 4) * 25088) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 896)] = max(conv2d_nchw[(((i1_inner * 8) + i2_inner) + 16)], 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 4) * 25088) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 401408)] = max(conv2d_nchw[(((i1_inner * 8) + i2_inner) + 32)], 0.000000e+00f);
      compute[((((((((((int)threadIdx.x) >> 4) * 25088) + (i1_inner * 12544)) + ((((int)blockIdx.x) / 7) * 1792)) + (i2_inner * 112)) + ((((int)blockIdx.x) % 7) * 16)) + (((int)threadIdx.x) & 15)) + 402304)] = max(conv2d_nchw[(((i1_inner * 8) + i2_inner) + 48)], 0.000000e+00f);
    }
  }
}


