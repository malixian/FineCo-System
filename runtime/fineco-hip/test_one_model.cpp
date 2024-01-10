#include "test/TestPairModel.cpp"
#include "test/TestTripleModels.cpp"
#include "test/TestQuadModels.cpp"
#include "test/TestCoKernel.cpp"
#include "test/TestOneModel.cpp"
#include "test/sort_kernel_latency.cpp"
#include "src/models/VGG.cpp"
#include "src/models/DarkNet.cpp"
#include "src/models/MobileNet.cpp"
#include "src/models/ResNet.cpp"
#include "src/models/Bert.cpp"
#include "src/models/AlexNet.cpp"
#include "src/services/fineco.cpp"
#include "src/services/fcfs.cpp"
#include "src/services/abacus.cpp"

void test_vgg() {
    TestOneModel test;
    test.IsolationRun<VGG>("VGG19");
}
void test_darknet() {
    TestOneModel test;
    test.IsolationRun<DarkNet>("DarkNet");
}
void test_mobilenet() {
    TestOneModel test;
    test.IsolationRun<MobileNet>("MobileNet");
}
void test_resnet152() {
    TestOneModel test;
    test.IsolationRun<ResNet>("ResNet152");
}
void test_resnet50() {
    TestOneModel test;
    test.IsolationRun<ResNet>("ResNet50");
}
void test_bert() {
    TestOneModel test;
    test.IsolationRun<Bert>("Bert");
}

void test_alexnet() {
    TestOneModel test;
    test.IsolationRun<AlexNet>("AlexNet");
}

int main(int argc, char *argv[]) {
    void (*func[7])() = {test_vgg, test_darknet, test_mobilenet, test_resnet152, test_resnet50, test_alexnet, test_bert};
    int i = atoi(argv[1]);
    func[i]();
    return 0;
}

