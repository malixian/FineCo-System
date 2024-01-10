
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
extern "C" __global__ void __launch_bounds__(112) candidate0(float* __restrict__ Input, float* __restrict__ kernel, float* __restrict__ compute) {
  float conv2d_nchw[4];
  __shared__ float pad_temp_shared[896];
  __shared__ float kernel_shared[512];
  conv2d_nchw[0] = 0.000000e+00f;
  conv2d_nchw[1] = 0.000000e+00f;
  conv2d_nchw[2] = 0.000000e+00f;
  conv2d_nchw[3] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 16; ++rc_outer_outer) {
    __syncthreads();
    *(float4*)(pad_temp_shared + (((int)threadIdx.x) * 4)) = *(float4*)(Input + ((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 7) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 7) * 4)));
    *(float4*)(pad_temp_shared + ((((int)threadIdx.x) * 4) + 448)) = *(float4*)(Input + (((((rc_outer_outer * 6272) + ((((int)threadIdx.x) / 7) * 196)) + ((((int)blockIdx.x) % 7) * 28)) + ((((int)threadIdx.x) % 7) * 4)) + 3136));
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) / 7) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31))];
    kernel_shared[(((int)threadIdx.x) + 112)] = kernel[(((((((int)blockIdx.x) / 7) * 8192) + (((((int)threadIdx.x) + 112) >> 5) * 512)) + (rc_outer_outer * 32)) + ((((int)threadIdx.x) + 16) & 31))];
    kernel_shared[(((int)threadIdx.x) + 224)] = kernel[((((((((int)blockIdx.x) / 7) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 3584)];
    kernel_shared[(((int)threadIdx.x) + 336)] = kernel[(((((((int)blockIdx.x) / 7) * 8192) + (((((int)threadIdx.x) + 336) >> 5) * 512)) + (rc_outer_outer * 32)) + ((((int)threadIdx.x) + 16) & 31))];
    if (((int)threadIdx.x) < 64) {
      kernel_shared[(((int)threadIdx.x) + 448)] = kernel[((((((((int)blockIdx.x) / 7) * 8192) + ((((int)threadIdx.x) >> 5) * 512)) + (rc_outer_outer * 32)) + (((int)threadIdx.x) & 31)) + 7168)];
    }
    __syncthreads();
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[(((int)threadIdx.x) % 28)] * kernel_shared[((((int)threadIdx.x) / 28) * 128)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[(((int)threadIdx.x) % 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 32)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[(((int)threadIdx.x) % 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 64)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[(((int)threadIdx.x) % 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 96)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 1)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 33)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 65)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 28)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 97)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 2)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 34)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 66)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 56)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 98)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 3)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 35)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 67)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 84)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 99)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 112)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 4)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 112)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 36)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 112)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 68)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 112)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 100)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 140)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 5)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 140)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 37)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 140)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 69)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 140)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 101)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 168)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 6)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 168)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 38)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 168)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 70)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 168)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 102)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 196)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 7)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 196)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 39)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 196)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 71)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 196)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 103)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 224)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 8)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 224)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 40)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 224)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 72)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 224)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 104)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 252)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 9)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 252)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 41)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 252)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 73)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 252)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 105)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 280)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 10)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 280)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 42)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 280)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 74)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 280)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 106)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 308)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 11)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 308)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 43)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 308)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 75)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 308)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 107)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 336)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 12)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 336)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 44)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 336)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 76)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 336)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 108)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 364)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 13)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 364)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 45)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 364)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 77)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 364)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 109)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 392)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 14)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 392)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 46)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 392)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 78)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 392)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 110)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 420)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 15)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 420)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 47)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 420)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 79)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 420)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 111)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 448)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 16)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 448)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 48)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 448)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 80)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 448)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 112)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 476)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 17)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 476)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 49)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 476)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 81)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 476)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 113)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 504)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 18)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 504)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 50)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 504)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 82)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 504)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 114)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 532)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 19)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 532)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 51)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 532)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 83)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 532)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 115)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 560)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 20)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 560)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 52)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 560)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 84)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 560)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 116)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 588)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 21)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 588)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 53)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 588)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 85)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 588)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 117)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 616)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 22)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 616)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 54)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 616)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 86)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 616)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 118)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 644)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 23)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 644)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 55)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 644)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 87)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 644)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 119)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 672)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 24)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 672)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 56)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 672)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 88)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 672)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 120)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 700)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 25)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 700)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 57)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 700)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 89)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 700)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 121)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 728)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 26)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 728)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 58)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 728)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 90)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 728)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 122)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 756)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 27)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 756)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 59)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 756)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 91)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 756)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 123)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 784)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 28)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 784)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 60)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 784)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 92)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 784)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 124)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 812)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 29)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 812)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 61)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 812)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 93)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 812)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 125)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 840)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 30)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 840)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 62)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 840)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 94)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 840)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 126)]));
    conv2d_nchw[0] = (conv2d_nchw[0] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 868)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 31)]));
    conv2d_nchw[1] = (conv2d_nchw[1] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 868)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 63)]));
    conv2d_nchw[2] = (conv2d_nchw[2] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 868)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 95)]));
    conv2d_nchw[3] = (conv2d_nchw[3] + (pad_temp_shared[((((int)threadIdx.x) % 28) + 868)] * kernel_shared[(((((int)threadIdx.x) / 28) * 128) + 127)]));
  }
  for (int i1_inner = 0; i1_inner < 4; ++i1_inner) {
    compute[((((((((int)blockIdx.x) / 7) * 3136) + ((((int)threadIdx.x) / 28) * 784)) + (i1_inner * 196)) + ((((int)blockIdx.x) % 7) * 28)) + (((int)threadIdx.x) % 28))] = max(conv2d_nchw[i1_inner], 0.000000e+00f);
  }
}


