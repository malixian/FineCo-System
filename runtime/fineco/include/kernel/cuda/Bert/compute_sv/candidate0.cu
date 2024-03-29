
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
extern "C" __global__ void __launch_bounds__(48) candidate0(float* __restrict__ Q, float* __restrict__ K, float* __restrict__ T_batch_matmul_NT) {
  float T_batch_matmul_NT_local[40];
  __shared__ float Q_shared[5];
  __shared__ float K_shared[384];
  T_batch_matmul_NT_local[0] = 0.000000e+00f;
  T_batch_matmul_NT_local[5] = 0.000000e+00f;
  T_batch_matmul_NT_local[10] = 0.000000e+00f;
  T_batch_matmul_NT_local[15] = 0.000000e+00f;
  T_batch_matmul_NT_local[20] = 0.000000e+00f;
  T_batch_matmul_NT_local[25] = 0.000000e+00f;
  T_batch_matmul_NT_local[30] = 0.000000e+00f;
  T_batch_matmul_NT_local[35] = 0.000000e+00f;
  T_batch_matmul_NT_local[1] = 0.000000e+00f;
  T_batch_matmul_NT_local[6] = 0.000000e+00f;
  T_batch_matmul_NT_local[11] = 0.000000e+00f;
  T_batch_matmul_NT_local[16] = 0.000000e+00f;
  T_batch_matmul_NT_local[21] = 0.000000e+00f;
  T_batch_matmul_NT_local[26] = 0.000000e+00f;
  T_batch_matmul_NT_local[31] = 0.000000e+00f;
  T_batch_matmul_NT_local[36] = 0.000000e+00f;
  T_batch_matmul_NT_local[2] = 0.000000e+00f;
  T_batch_matmul_NT_local[7] = 0.000000e+00f;
  T_batch_matmul_NT_local[12] = 0.000000e+00f;
  T_batch_matmul_NT_local[17] = 0.000000e+00f;
  T_batch_matmul_NT_local[22] = 0.000000e+00f;
  T_batch_matmul_NT_local[27] = 0.000000e+00f;
  T_batch_matmul_NT_local[32] = 0.000000e+00f;
  T_batch_matmul_NT_local[37] = 0.000000e+00f;
  T_batch_matmul_NT_local[3] = 0.000000e+00f;
  T_batch_matmul_NT_local[8] = 0.000000e+00f;
  T_batch_matmul_NT_local[13] = 0.000000e+00f;
  T_batch_matmul_NT_local[18] = 0.000000e+00f;
  T_batch_matmul_NT_local[23] = 0.000000e+00f;
  T_batch_matmul_NT_local[28] = 0.000000e+00f;
  T_batch_matmul_NT_local[33] = 0.000000e+00f;
  T_batch_matmul_NT_local[38] = 0.000000e+00f;
  T_batch_matmul_NT_local[4] = 0.000000e+00f;
  T_batch_matmul_NT_local[9] = 0.000000e+00f;
  T_batch_matmul_NT_local[14] = 0.000000e+00f;
  T_batch_matmul_NT_local[19] = 0.000000e+00f;
  T_batch_matmul_NT_local[24] = 0.000000e+00f;
  T_batch_matmul_NT_local[29] = 0.000000e+00f;
  T_batch_matmul_NT_local[34] = 0.000000e+00f;
  T_batch_matmul_NT_local[39] = 0.000000e+00f;
  if (((int)threadIdx.x) < 5) {
    Q_shared[((int)threadIdx.x)] = Q[(((int)threadIdx.x) * 5)];
  }
  K_shared[((int)threadIdx.x)] = K[((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5))];
  K_shared[(((int)threadIdx.x) + 48)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 240)];
  K_shared[(((int)threadIdx.x) + 96)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 480)];
  K_shared[(((int)threadIdx.x) + 144)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 720)];
  K_shared[(((int)threadIdx.x) + 192)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 960)];
  K_shared[(((int)threadIdx.x) + 240)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1200)];
  K_shared[(((int)threadIdx.x) + 288)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1440)];
  K_shared[(((int)threadIdx.x) + 336)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1680)];
  __syncthreads();
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (Q_shared[0] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[5] = (T_batch_matmul_NT_local[5] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[10] = (T_batch_matmul_NT_local[10] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[15] = (T_batch_matmul_NT_local[15] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[20] = (T_batch_matmul_NT_local[20] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[25] = (T_batch_matmul_NT_local[25] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[30] = (T_batch_matmul_NT_local[30] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[35] = (T_batch_matmul_NT_local[35] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (Q_shared[1] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[6] = (T_batch_matmul_NT_local[6] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[11] = (T_batch_matmul_NT_local[11] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[16] = (T_batch_matmul_NT_local[16] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[21] = (T_batch_matmul_NT_local[21] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[26] = (T_batch_matmul_NT_local[26] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[31] = (T_batch_matmul_NT_local[31] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[36] = (T_batch_matmul_NT_local[36] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (Q_shared[2] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[7] = (T_batch_matmul_NT_local[7] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[12] = (T_batch_matmul_NT_local[12] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[17] = (T_batch_matmul_NT_local[17] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[22] = (T_batch_matmul_NT_local[22] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[27] = (T_batch_matmul_NT_local[27] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[32] = (T_batch_matmul_NT_local[32] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[37] = (T_batch_matmul_NT_local[37] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (Q_shared[3] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[8] = (T_batch_matmul_NT_local[8] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[13] = (T_batch_matmul_NT_local[13] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[18] = (T_batch_matmul_NT_local[18] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[23] = (T_batch_matmul_NT_local[23] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[28] = (T_batch_matmul_NT_local[28] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[33] = (T_batch_matmul_NT_local[33] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[38] = (T_batch_matmul_NT_local[38] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (Q_shared[4] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[9] = (T_batch_matmul_NT_local[9] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[14] = (T_batch_matmul_NT_local[14] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[19] = (T_batch_matmul_NT_local[19] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[24] = (T_batch_matmul_NT_local[24] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[29] = (T_batch_matmul_NT_local[29] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[34] = (T_batch_matmul_NT_local[34] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[39] = (T_batch_matmul_NT_local[39] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 336)]));
  __syncthreads();
  if (((int)threadIdx.x) < 5) {
    Q_shared[((int)threadIdx.x)] = Q[((((int)threadIdx.x) * 5) + 1)];
  }
  K_shared[((int)threadIdx.x)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1)];
  K_shared[(((int)threadIdx.x) + 48)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 241)];
  K_shared[(((int)threadIdx.x) + 96)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 481)];
  K_shared[(((int)threadIdx.x) + 144)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 721)];
  K_shared[(((int)threadIdx.x) + 192)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 961)];
  K_shared[(((int)threadIdx.x) + 240)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1201)];
  K_shared[(((int)threadIdx.x) + 288)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1441)];
  K_shared[(((int)threadIdx.x) + 336)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1681)];
  __syncthreads();
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (Q_shared[0] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[5] = (T_batch_matmul_NT_local[5] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[10] = (T_batch_matmul_NT_local[10] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[15] = (T_batch_matmul_NT_local[15] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[20] = (T_batch_matmul_NT_local[20] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[25] = (T_batch_matmul_NT_local[25] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[30] = (T_batch_matmul_NT_local[30] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[35] = (T_batch_matmul_NT_local[35] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (Q_shared[1] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[6] = (T_batch_matmul_NT_local[6] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[11] = (T_batch_matmul_NT_local[11] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[16] = (T_batch_matmul_NT_local[16] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[21] = (T_batch_matmul_NT_local[21] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[26] = (T_batch_matmul_NT_local[26] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[31] = (T_batch_matmul_NT_local[31] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[36] = (T_batch_matmul_NT_local[36] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (Q_shared[2] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[7] = (T_batch_matmul_NT_local[7] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[12] = (T_batch_matmul_NT_local[12] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[17] = (T_batch_matmul_NT_local[17] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[22] = (T_batch_matmul_NT_local[22] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[27] = (T_batch_matmul_NT_local[27] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[32] = (T_batch_matmul_NT_local[32] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[37] = (T_batch_matmul_NT_local[37] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (Q_shared[3] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[8] = (T_batch_matmul_NT_local[8] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[13] = (T_batch_matmul_NT_local[13] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[18] = (T_batch_matmul_NT_local[18] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[23] = (T_batch_matmul_NT_local[23] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[28] = (T_batch_matmul_NT_local[28] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[33] = (T_batch_matmul_NT_local[33] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[38] = (T_batch_matmul_NT_local[38] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (Q_shared[4] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[9] = (T_batch_matmul_NT_local[9] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[14] = (T_batch_matmul_NT_local[14] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[19] = (T_batch_matmul_NT_local[19] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[24] = (T_batch_matmul_NT_local[24] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[29] = (T_batch_matmul_NT_local[29] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[34] = (T_batch_matmul_NT_local[34] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[39] = (T_batch_matmul_NT_local[39] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 336)]));
  __syncthreads();
  if (((int)threadIdx.x) < 5) {
    Q_shared[((int)threadIdx.x)] = Q[((((int)threadIdx.x) * 5) + 2)];
  }
  K_shared[((int)threadIdx.x)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 2)];
  K_shared[(((int)threadIdx.x) + 48)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 242)];
  K_shared[(((int)threadIdx.x) + 96)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 482)];
  K_shared[(((int)threadIdx.x) + 144)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 722)];
  K_shared[(((int)threadIdx.x) + 192)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 962)];
  K_shared[(((int)threadIdx.x) + 240)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1202)];
  K_shared[(((int)threadIdx.x) + 288)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1442)];
  K_shared[(((int)threadIdx.x) + 336)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1682)];
  __syncthreads();
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (Q_shared[0] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[5] = (T_batch_matmul_NT_local[5] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[10] = (T_batch_matmul_NT_local[10] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[15] = (T_batch_matmul_NT_local[15] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[20] = (T_batch_matmul_NT_local[20] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[25] = (T_batch_matmul_NT_local[25] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[30] = (T_batch_matmul_NT_local[30] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[35] = (T_batch_matmul_NT_local[35] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (Q_shared[1] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[6] = (T_batch_matmul_NT_local[6] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[11] = (T_batch_matmul_NT_local[11] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[16] = (T_batch_matmul_NT_local[16] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[21] = (T_batch_matmul_NT_local[21] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[26] = (T_batch_matmul_NT_local[26] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[31] = (T_batch_matmul_NT_local[31] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[36] = (T_batch_matmul_NT_local[36] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (Q_shared[2] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[7] = (T_batch_matmul_NT_local[7] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[12] = (T_batch_matmul_NT_local[12] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[17] = (T_batch_matmul_NT_local[17] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[22] = (T_batch_matmul_NT_local[22] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[27] = (T_batch_matmul_NT_local[27] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[32] = (T_batch_matmul_NT_local[32] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[37] = (T_batch_matmul_NT_local[37] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (Q_shared[3] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[8] = (T_batch_matmul_NT_local[8] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[13] = (T_batch_matmul_NT_local[13] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[18] = (T_batch_matmul_NT_local[18] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[23] = (T_batch_matmul_NT_local[23] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[28] = (T_batch_matmul_NT_local[28] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[33] = (T_batch_matmul_NT_local[33] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[38] = (T_batch_matmul_NT_local[38] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (Q_shared[4] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[9] = (T_batch_matmul_NT_local[9] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[14] = (T_batch_matmul_NT_local[14] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[19] = (T_batch_matmul_NT_local[19] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[24] = (T_batch_matmul_NT_local[24] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[29] = (T_batch_matmul_NT_local[29] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[34] = (T_batch_matmul_NT_local[34] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[39] = (T_batch_matmul_NT_local[39] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 336)]));
  __syncthreads();
  if (((int)threadIdx.x) < 5) {
    Q_shared[((int)threadIdx.x)] = Q[((((int)threadIdx.x) * 5) + 3)];
  }
  K_shared[((int)threadIdx.x)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 3)];
  K_shared[(((int)threadIdx.x) + 48)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 243)];
  K_shared[(((int)threadIdx.x) + 96)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 483)];
  K_shared[(((int)threadIdx.x) + 144)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 723)];
  K_shared[(((int)threadIdx.x) + 192)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 963)];
  K_shared[(((int)threadIdx.x) + 240)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1203)];
  K_shared[(((int)threadIdx.x) + 288)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1443)];
  K_shared[(((int)threadIdx.x) + 336)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1683)];
  __syncthreads();
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (Q_shared[0] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[5] = (T_batch_matmul_NT_local[5] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[10] = (T_batch_matmul_NT_local[10] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[15] = (T_batch_matmul_NT_local[15] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[20] = (T_batch_matmul_NT_local[20] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[25] = (T_batch_matmul_NT_local[25] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[30] = (T_batch_matmul_NT_local[30] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[35] = (T_batch_matmul_NT_local[35] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (Q_shared[1] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[6] = (T_batch_matmul_NT_local[6] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[11] = (T_batch_matmul_NT_local[11] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[16] = (T_batch_matmul_NT_local[16] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[21] = (T_batch_matmul_NT_local[21] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[26] = (T_batch_matmul_NT_local[26] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[31] = (T_batch_matmul_NT_local[31] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[36] = (T_batch_matmul_NT_local[36] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (Q_shared[2] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[7] = (T_batch_matmul_NT_local[7] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[12] = (T_batch_matmul_NT_local[12] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[17] = (T_batch_matmul_NT_local[17] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[22] = (T_batch_matmul_NT_local[22] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[27] = (T_batch_matmul_NT_local[27] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[32] = (T_batch_matmul_NT_local[32] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[37] = (T_batch_matmul_NT_local[37] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (Q_shared[3] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[8] = (T_batch_matmul_NT_local[8] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[13] = (T_batch_matmul_NT_local[13] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[18] = (T_batch_matmul_NT_local[18] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[23] = (T_batch_matmul_NT_local[23] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[28] = (T_batch_matmul_NT_local[28] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[33] = (T_batch_matmul_NT_local[33] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[38] = (T_batch_matmul_NT_local[38] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (Q_shared[4] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[9] = (T_batch_matmul_NT_local[9] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[14] = (T_batch_matmul_NT_local[14] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[19] = (T_batch_matmul_NT_local[19] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[24] = (T_batch_matmul_NT_local[24] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[29] = (T_batch_matmul_NT_local[29] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[34] = (T_batch_matmul_NT_local[34] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[39] = (T_batch_matmul_NT_local[39] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 336)]));
  __syncthreads();
  if (((int)threadIdx.x) < 5) {
    Q_shared[((int)threadIdx.x)] = Q[((((int)threadIdx.x) * 5) + 4)];
  }
  K_shared[((int)threadIdx.x)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 4)];
  K_shared[(((int)threadIdx.x) + 48)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 244)];
  K_shared[(((int)threadIdx.x) + 96)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 484)];
  K_shared[(((int)threadIdx.x) + 144)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 724)];
  K_shared[(((int)threadIdx.x) + 192)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 964)];
  K_shared[(((int)threadIdx.x) + 240)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1204)];
  K_shared[(((int)threadIdx.x) + 288)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1444)];
  K_shared[(((int)threadIdx.x) + 336)] = K[(((((int)blockIdx.x) * 1920) + (((int)threadIdx.x) * 5)) + 1684)];
  __syncthreads();
  T_batch_matmul_NT_local[0] = (T_batch_matmul_NT_local[0] + (Q_shared[0] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[5] = (T_batch_matmul_NT_local[5] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[10] = (T_batch_matmul_NT_local[10] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[15] = (T_batch_matmul_NT_local[15] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[20] = (T_batch_matmul_NT_local[20] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[25] = (T_batch_matmul_NT_local[25] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[30] = (T_batch_matmul_NT_local[30] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[35] = (T_batch_matmul_NT_local[35] + (Q_shared[0] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[1] = (T_batch_matmul_NT_local[1] + (Q_shared[1] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[6] = (T_batch_matmul_NT_local[6] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[11] = (T_batch_matmul_NT_local[11] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[16] = (T_batch_matmul_NT_local[16] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[21] = (T_batch_matmul_NT_local[21] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[26] = (T_batch_matmul_NT_local[26] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[31] = (T_batch_matmul_NT_local[31] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[36] = (T_batch_matmul_NT_local[36] + (Q_shared[1] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[2] = (T_batch_matmul_NT_local[2] + (Q_shared[2] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[7] = (T_batch_matmul_NT_local[7] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[12] = (T_batch_matmul_NT_local[12] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[17] = (T_batch_matmul_NT_local[17] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[22] = (T_batch_matmul_NT_local[22] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[27] = (T_batch_matmul_NT_local[27] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[32] = (T_batch_matmul_NT_local[32] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[37] = (T_batch_matmul_NT_local[37] + (Q_shared[2] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[3] = (T_batch_matmul_NT_local[3] + (Q_shared[3] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[8] = (T_batch_matmul_NT_local[8] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[13] = (T_batch_matmul_NT_local[13] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[18] = (T_batch_matmul_NT_local[18] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[23] = (T_batch_matmul_NT_local[23] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[28] = (T_batch_matmul_NT_local[28] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[33] = (T_batch_matmul_NT_local[33] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[38] = (T_batch_matmul_NT_local[38] + (Q_shared[3] * K_shared[(((int)threadIdx.x) + 336)]));
  T_batch_matmul_NT_local[4] = (T_batch_matmul_NT_local[4] + (Q_shared[4] * K_shared[((int)threadIdx.x)]));
  T_batch_matmul_NT_local[9] = (T_batch_matmul_NT_local[9] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 48)]));
  T_batch_matmul_NT_local[14] = (T_batch_matmul_NT_local[14] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 96)]));
  T_batch_matmul_NT_local[19] = (T_batch_matmul_NT_local[19] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 144)]));
  T_batch_matmul_NT_local[24] = (T_batch_matmul_NT_local[24] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 192)]));
  T_batch_matmul_NT_local[29] = (T_batch_matmul_NT_local[29] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 240)]));
  T_batch_matmul_NT_local[34] = (T_batch_matmul_NT_local[34] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 288)]));
  T_batch_matmul_NT_local[39] = (T_batch_matmul_NT_local[39] + (Q_shared[4] * K_shared[(((int)threadIdx.x) + 336)]));
  for (int i_inner = 0; i_inner < 5; ++i_inner) {
    T_batch_matmul_NT[(((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x))] = T_batch_matmul_NT_local[i_inner];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 48)] = T_batch_matmul_NT_local[(i_inner + 5)];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 96)] = T_batch_matmul_NT_local[(i_inner + 10)];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 144)] = T_batch_matmul_NT_local[(i_inner + 15)];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 192)] = T_batch_matmul_NT_local[(i_inner + 20)];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 240)] = T_batch_matmul_NT_local[(i_inner + 25)];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 288)] = T_batch_matmul_NT_local[(i_inner + 30)];
    T_batch_matmul_NT[((((i_inner * 768) + (((int)blockIdx.x) * 384)) + ((int)threadIdx.x)) + 336)] = T_batch_matmul_NT_local[(i_inner + 35)];
  }
}


