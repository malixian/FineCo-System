#include "test/TestPairModel.cpp"
#include "test/TestTripleModels.cpp"
#include "test/TestQuadModels.cpp"
#include "test/TestCoKernel.cpp"
#include "test/TestOneModel.cpp"
#include "test/TestOneKernel.cpp"
#include "test/sort_kernel_latency.cpp"
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

template<typename M1, typename M2>
void test_priority_service(const string name1, const string name2, int right_size) {
    PriorityService test;
    test.Listen<M1, M2>(name1, name2, right_size);
    test.GenerateFixedSeq(100);
    test.Accept();
}

// x + VGG
void VGG19_VGG19(int right_size) {
    test_priority_service<VGG, VGG>("VGG19", "VGG19", right_size);
}
void DarkNet_VGG19(int right_size) {
    test_priority_service<DarkNet, VGG>("DarkNet", "VGG19", right_size);
}
void MobileNet_VGG19(int right_size) {
    test_priority_service<MobileNet, VGG>("MobileNet", "VGG19", right_size);
}
void ResNet152_VGG19(int right_size) {
    test_priority_service<ResNet, VGG>("ResNet152", "VGG19", right_size);
}
void ResNet50_VGG19(int right_size) {
    test_priority_service<ResNet, VGG>("ResNet50", "VGG19", right_size);
}
void AlexNet_VGG19(int right_size) {
    test_priority_service<AlexNet, VGG>("AlexNet", "VGG19", right_size);
}
void Bert_VGG19(int right_size) {
    test_priority_service<Bert, VGG>("Bert", "VGG19", right_size);
}

// x + DarkNet
void VGG19_DarkNet(int right_size) {
    test_priority_service<VGG, DarkNet>("VGG19", "DarkNet", right_size);
}
void DarkNet_DarkNet(int right_size) {
    test_priority_service<DarkNet, DarkNet>("DarkNet", "DarkNet", right_size);
}
void MobileNet_DarkNet(int right_size) {
    test_priority_service<MobileNet, DarkNet>("MobileNet", "DarkNet", right_size);
}
void ResNet152_DarkNet(int right_size) {
    test_priority_service<ResNet, DarkNet>("ResNet152", "DarkNet", right_size);
}
void ResNet50_DarkNet(int right_size) {
    test_priority_service<ResNet, DarkNet>("ResNet50", "DarkNet", right_size);
}
void AlexNet_DarkNet(int right_size) {
    test_priority_service<AlexNet, DarkNet>("AlexNet", "DarkNet", right_size);
}
void Bert_DarkNet(int right_size) {
    test_priority_service<Bert, DarkNet>("Bert", "DarkNet", right_size);
}

// x + MobileNet
void VGG19_MobileNet(int right_size) {
    test_priority_service<VGG, MobileNet>("VGG19", "MobileNet", right_size);
}
void DarkNet_MobileNet(int right_size) {
    test_priority_service<DarkNet, MobileNet>("DarkNet", "MobileNet", right_size);
}
void MobileNet_MobileNet(int right_size) {
    test_priority_service<MobileNet, MobileNet>("MobileNet", "MobileNet", right_size);
}
void ResNet152_MobileNet(int right_size) {
    test_priority_service<ResNet, MobileNet>("ResNet152", "MobileNet", right_size);
}
void ResNet50_MobileNet(int right_size) {
    test_priority_service<ResNet, MobileNet>("ResNet50", "MobileNet", right_size);
}
void AlexNet_MobileNet(int right_size) {
    test_priority_service<AlexNet, MobileNet>("AlexNet", "MobileNet", right_size);
}
void Bert_MobileNet(int right_size) {
    test_priority_service<Bert, MobileNet>("Bert", "MobileNet", right_size);
}

// x + ResNet152
void VGG19_ResNet152(int right_size) {
    test_priority_service<VGG, ResNet>("VGG19", "ResNet152", right_size);
}
void DarkNet_ResNet152(int right_size) {
    test_priority_service<DarkNet, ResNet>("DarkNet", "ResNet152", right_size);
}
void MobileNet_ResNet152(int right_size) {
    test_priority_service<MobileNet, ResNet>("MobileNet", "ResNet152", right_size);
}
void ResNet152_ResNet152(int right_size) {
    test_priority_service<ResNet, ResNet>("ResNet152", "ResNet152", right_size);
}
void ResNet50_ResNet152(int right_size) {
    test_priority_service<ResNet, ResNet>("ResNet50", "ResNet152", right_size);
}
void AlexNet_ResNet152(int right_size) {
    test_priority_service<AlexNet, ResNet>("AlexNet", "ResNet152", right_size);
}
void Bert_ResNet152(int right_size) {
    test_priority_service<Bert, ResNet>("Bert", "ResNet152", right_size);
}

void VGG19_ResNet50(int right_size) {
    test_priority_service<VGG, ResNet>("VGG19", "ResNet50", right_size);
}
void DarkNet_ResNet50(int right_size) {
    test_priority_service<DarkNet, ResNet>("DarkNet", "ResNet50", right_size);
}
void MobileNet_ResNet50(int right_size) {
    test_priority_service<MobileNet, ResNet>("MobileNet", "ResNet50", right_size);
}
void ResNet152_ResNet50(int right_size) {
    test_priority_service<ResNet, ResNet>("ResNet152", "ResNet50", right_size);
}
void ResNet50_ResNet50(int right_size) {
    test_priority_service<ResNet, ResNet>("ResNet50", "ResNet50", right_size);
}
void AlexNet_ResNet50(int right_size) {
    test_priority_service<AlexNet, ResNet>("AlexNet", "ResNet50", right_size);
}
void Bert_ResNet50(int right_size) {
    test_priority_service<Bert, ResNet>("Bert", "ResNet50", right_size);
}

void VGG19_AlexNet(int right_size) {
    test_priority_service<VGG, AlexNet>("VGG19", "AlexNet", right_size);
}
void DarkNet_AlexNet(int right_size) {
    test_priority_service<DarkNet, AlexNet>("DarkNet", "AlexNet", right_size);
}
void MobileNet_AlexNet(int right_size) {
    test_priority_service<MobileNet, AlexNet>("MobileNet", "AlexNet", right_size);
}
void ResNet152_AlexNet(int right_size) {
    test_priority_service<ResNet, AlexNet>("ResNet152", "AlexNet", right_size);
}
void ResNet50_AlexNet(int right_size) {
    test_priority_service<ResNet, AlexNet>("ResNet50", "AlexNet", right_size);
}
void AlexNet_AlexNet(int right_size) {
    test_priority_service<AlexNet, AlexNet>("AlexNet", "AlexNet", right_size);
}
void Bert_AlexNet(int right_size) {
    test_priority_service<Bert, AlexNet>("Bert", "AlexNet", right_size);
}

void VGG19_Bert(int right_size) {
    test_priority_service<VGG, Bert>("VGG19", "Bert", right_size);
}
void DarkNet_Bert(int right_size) {
    test_priority_service<DarkNet, Bert>("DarkNet", "Bert", right_size);
}
void MobileNet_Bert(int right_size) {
    test_priority_service<MobileNet, Bert>("MobileNet", "Bert", right_size);
}
void ResNet152_Bert(int right_size) {
    test_priority_service<ResNet, Bert>("ResNet152", "Bert", right_size);
}
void ResNet50_Bert(int right_size) {
    test_priority_service<ResNet, Bert>("ResNet50", "Bert", right_size);
}
void AlexNet_Bert(int right_size) {
    test_priority_service<AlexNet, Bert>("AlexNet", "Bert", right_size);
}
void Bert_Bert(int right_size) {
    test_priority_service<Bert, Bert>("Bert", "Bert", right_size);
}


int main(int argc, char *argv[]) {
    // bert 和其他模型共置会 HIP ERROR，爆显存（？
    void (*func[7][7])(int) = {
                {VGG19_VGG19, DarkNet_VGG19, MobileNet_VGG19, ResNet152_VGG19, ResNet50_VGG19, AlexNet_VGG19, Bert_VGG19},
                {VGG19_DarkNet, DarkNet_DarkNet, MobileNet_DarkNet, ResNet152_DarkNet, ResNet50_DarkNet, AlexNet_DarkNet, Bert_DarkNet},
                {VGG19_MobileNet, DarkNet_MobileNet, MobileNet_MobileNet, ResNet152_MobileNet, ResNet50_MobileNet, AlexNet_MobileNet, Bert_MobileNet},
                {VGG19_ResNet152, DarkNet_ResNet152, MobileNet_ResNet152, ResNet152_ResNet152, ResNet50_ResNet152, AlexNet_ResNet152, Bert_ResNet152},
                {VGG19_ResNet50, DarkNet_ResNet50, MobileNet_ResNet50, ResNet152_ResNet50, ResNet50_ResNet50, AlexNet_ResNet50, Bert_ResNet50},
                {VGG19_AlexNet, DarkNet_AlexNet, MobileNet_AlexNet, ResNet152_AlexNet, ResNet50_AlexNet, AlexNet_AlexNet, Bert_AlexNet},
                {VGG19_Bert, DarkNet_Bert, MobileNet_Bert, ResNet152_Bert, ResNet50_Bert, AlexNet_Bert, Bert_Bert}
                };
    int i = atoi(argv[1]), j = atoi(argv[2]), right_size = atoi(argv[3]);
    func[i][j](right_size);
    return 0;
}