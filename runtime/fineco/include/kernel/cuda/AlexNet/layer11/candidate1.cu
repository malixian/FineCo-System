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
extern "C" __global__ void __launch_bounds__(4) candidate1(float* __restrict__ data, float* __restrict__ weight, float* __restrict__ compute, float* __restrict__ bias) {
  float T_matmul_NT[1];
  __shared__ float data_shared[1024];
  __shared__ float weight_shared[4096];
  T_matmul_NT[0] = 0.000000e+00f;
  for (int k_outer_outer = 0; k_outer_outer < 2; ++k_outer_outer) {
    __syncthreads();
    *(float4*)(data_shared + (((int)threadIdx.x) * 4)) = *(float4*)(data + ((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 16)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 16));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 32)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 32));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 48)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 48));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 64)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 64));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 80)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 80));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 96)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 96));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 112)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 112));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 128)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 128));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 144)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 144));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 160)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 160));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 176)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 176));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 192)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 192));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 208)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 208));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 224)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 224));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 240)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 240));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 256)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 256));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 272)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 272));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 288)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 288));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 304)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 304));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 320)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 320));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 336)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 336));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 352)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 352));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 368)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 368));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 384)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 384));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 400)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 400));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 416)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 416));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 432)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 432));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 448)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 448));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 464)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 464));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 480)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 480));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 496)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 496));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 512)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 512));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 528)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 528));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 544)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 544));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 560)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 560));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 576)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 576));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 592)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 592));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 608)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 608));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 624)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 624));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 640)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 640));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 656)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 656));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 672)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 672));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 688)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 688));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 704)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 704));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 720)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 720));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 736)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 736));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 752)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 752));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 768)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 768));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 784)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 784));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 800)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 800));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 816)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 816));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 832)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 832));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 848)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 848));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 864)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 864));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 880)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 880));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 896)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 896));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 912)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 912));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 928)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 928));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 944)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 944));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 960)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 960));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 976)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 976));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 992)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 992));
    *(float4*)(data_shared + ((((int)threadIdx.x) * 4) + 1008)) = *(float4*)(data + (((k_outer_outer * 1024) + (((int)threadIdx.x) * 4)) + 1008));
    for (int ax0_ax1_fused_outer_outer = 0; ax0_ax1_fused_outer_outer < 256; ++ax0_ax1_fused_outer_outer) {
      *(float4*)(weight_shared + ((ax0_ax1_fused_outer_outer * 16) + (((int)threadIdx.x) * 4))) = *(float4*)(weight + (((((((int)blockIdx.x) * 8192) + ((ax0_ax1_fused_outer_outer >> 6) * 2048)) + (k_outer_outer * 1024)) + ((ax0_ax1_fused_outer_outer & 63) * 16)) + (((int)threadIdx.x) * 4)));
    }
    __syncthreads();
    for (int k_outer_inner = 0; k_outer_inner < 64; ++k_outer_inner) {
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[(k_outer_inner * 16)] * weight_shared[((((int)threadIdx.x) * 1024) + (k_outer_inner * 16))]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 1)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 1)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 2)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 2)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 3)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 3)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 4)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 4)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 5)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 5)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 6)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 6)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 7)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 7)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 8)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 8)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 9)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 9)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 10)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 10)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 11)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 11)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 12)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 12)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 13)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 13)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 14)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 14)]));
      T_matmul_NT[0] = (T_matmul_NT[0] + (data_shared[((k_outer_inner * 16) + 15)] * weight_shared[(((((int)threadIdx.x) * 1024) + (k_outer_inner * 16)) + 15)]));
    }
  }
  compute[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))] = (T_matmul_NT[0] + bias[((((int)blockIdx.x) * 4) + ((int)threadIdx.x))]);
}


