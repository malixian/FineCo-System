
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
extern "C" __global__ void __launch_bounds__(196) candidate0(float* __restrict__ data, float* __restrict__ kernel, float* __restrict__ conv2d_nchw) {
  float conv2d_nchw_local[16];
  __shared__ float pad_temp_shared[6272];
  __shared__ float kernel_shared[2048];
  conv2d_nchw_local[0] = 0.000000e+00f;
  conv2d_nchw_local[1] = 0.000000e+00f;
  conv2d_nchw_local[2] = 0.000000e+00f;
  conv2d_nchw_local[3] = 0.000000e+00f;
  conv2d_nchw_local[4] = 0.000000e+00f;
  conv2d_nchw_local[5] = 0.000000e+00f;
  conv2d_nchw_local[6] = 0.000000e+00f;
  conv2d_nchw_local[7] = 0.000000e+00f;
  conv2d_nchw_local[8] = 0.000000e+00f;
  conv2d_nchw_local[9] = 0.000000e+00f;
  conv2d_nchw_local[10] = 0.000000e+00f;
  conv2d_nchw_local[11] = 0.000000e+00f;
  conv2d_nchw_local[12] = 0.000000e+00f;
  conv2d_nchw_local[13] = 0.000000e+00f;
  conv2d_nchw_local[14] = 0.000000e+00f;
  conv2d_nchw_local[15] = 0.000000e+00f;
  for (int rc_outer_outer = 0; rc_outer_outer < 4; ++rc_outer_outer) {
    __syncthreads();
    *(float2*)(pad_temp_shared + (((int)threadIdx.x) * 2)) = *(float2*)(data + ((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 392)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 784));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 784)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 1568));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 1176)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 2352));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 1568)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 3136));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 1960)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 3920));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 2352)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 4704));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 2744)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 5488));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 3136)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 6272));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 3528)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 7056));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 3920)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 7840));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 4312)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 8624));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 4704)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 9408));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 5096)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 10192));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 5488)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 10976));
    *(float2*)(pad_temp_shared + ((((int)threadIdx.x) * 2) + 5880)) = *(float2*)(data + (((((rc_outer_outer * 12544) + ((((int)threadIdx.x) / 49) * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + 11760));
    kernel_shared[((int)threadIdx.x)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + ((((int)threadIdx.x) >> 6) * 256)) + (rc_outer_outer * 64)) + (((int)threadIdx.x) & 63))];
    kernel_shared[(((int)threadIdx.x) + 196)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 196) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 4) & 63))];
    kernel_shared[(((int)threadIdx.x) + 392)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 392) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 8) & 63))];
    kernel_shared[(((int)threadIdx.x) + 588)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 588) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 12) & 63))];
    kernel_shared[(((int)threadIdx.x) + 784)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 784) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 16) & 63))];
    kernel_shared[(((int)threadIdx.x) + 980)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 980) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 20) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1176)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 1176) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 24) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1372)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 1372) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 28) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1568)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 1568) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 32) & 63))];
    kernel_shared[(((int)threadIdx.x) + 1764)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 1764) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 36) & 63))];
    if (((int)threadIdx.x) < 88) {
      kernel_shared[(((int)threadIdx.x) + 1960)] = kernel[(((((((int)blockIdx.x) >> 1) * 8192) + (((((int)threadIdx.x) + 1960) >> 6) * 256)) + (rc_outer_outer * 64)) + ((((int)threadIdx.x) + 40) & 63))];
    }
    __syncthreads();
    for (int rc_outer_inner = 0; rc_outer_inner < 2; ++rc_outer_inner) {
      for (int ff_c_outer_inner = 0; ff_c_outer_inner < 8; ++ff_c_outer_inner) {
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2))] * kernel_shared[((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32))]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 98)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 1)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 196)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 2)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 294)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 3)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 392)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 4)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 490)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 5)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 588)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 6)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 686)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 7)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 784)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 8)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 882)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 9)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 980)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 10)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1078)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 11)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1176)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 12)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1274)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 13)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1372)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 14)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1470)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 15)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1568)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 16)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1666)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 17)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1764)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 18)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1862)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 19)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1960)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 20)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2058)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 21)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2156)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 22)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2254)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 23)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2352)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 24)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2450)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 25)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2548)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 26)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2646)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 27)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2744)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 28)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2842)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 29)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2940)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 30)]));
        conv2d_nchw_local[(ff_c_outer_inner * 2)] = (conv2d_nchw_local[(ff_c_outer_inner * 2)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 3038)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 31)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1)] * kernel_shared[((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32))]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 99)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 1)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 197)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 2)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 295)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 3)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 393)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 4)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 491)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 5)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 589)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 6)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 687)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 7)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 785)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 8)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 883)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 9)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 981)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 10)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1079)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 11)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1177)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 12)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1275)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 13)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1373)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 14)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1471)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 15)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1569)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 16)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1667)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 17)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1765)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 18)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1863)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 19)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 1961)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 20)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2059)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 21)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2157)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 22)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2255)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 23)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2353)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 24)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2451)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 25)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2549)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 26)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2647)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 27)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2745)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 28)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2843)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 29)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 2941)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 30)]));
        conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] = (conv2d_nchw_local[((ff_c_outer_inner * 2) + 1)] + (pad_temp_shared[(((rc_outer_inner * 3136) + ((((int)threadIdx.x) % 49) * 2)) + 3039)] * kernel_shared[(((((((int)threadIdx.x) / 49) * 512) + (ff_c_outer_inner * 64)) + (rc_outer_inner * 32)) + 31)]));
      }
    }
  }
  for (int ff_inner = 0; ff_inner < 8; ++ff_inner) {
    for (int xx_inner = 0; xx_inner < 2; ++xx_inner) {
      conv2d_nchw[(((((((((int)blockIdx.x) >> 1) * 6272) + ((((int)threadIdx.x) / 49) * 1568)) + (ff_inner * 196)) + ((((int)blockIdx.x) & 1) * 98)) + ((((int)threadIdx.x) % 49) * 2)) + xx_inner)] = conv2d_nchw_local[((ff_inner * 2) + xx_inner)];
    }
  }
}


