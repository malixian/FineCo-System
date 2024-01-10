#include "test/TestPairModel.cpp"
#include "test/TestTripleModels.cpp"
#include "test/TestQuadModels.cpp"
#include "test/TestCoKernel.cpp"
#include "test/TestOneModel.cpp"
#include "test/TestOneKernel.cpp"
// #include "test/sort_kernel_latency.cpp"
#include "test/TestKernelLatency.cpp"
#include "src/models/VGG.cpp"
#include "src/models/DarkNet.cpp"
#include "src/models/MobileNet.cpp"
#include "src/models/ResNet.cpp"
#include "src/models/Bert.cpp"
#include "src/services/fineco.cpp"
#include "src/services/fineco_v1.cpp"
#include "src/services/fcfs.cpp"
#include "src/services/abacus.cpp"
#include "src/services/priority.cpp"
#include "src/services/multi_stream.cpp"
#include "src/services/krisp.cpp"
#include "src/services/reef.cpp"

void test_co_kernels(int layer, int candidate) {
    TestCoKernel test("ResNet50", "MobileNet");
    test.KernelTest<ResNet, MobileNet>(layer, layer, candidate, candidate, 1);
}

int main(int argc, char *argv[]) {
    int layer = atoi(argv[1]), candidate = atoi(argv[2]);
    test_co_kernels(layer, candidate);
}