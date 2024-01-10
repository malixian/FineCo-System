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

using namespace std;

template<typename M1, typename M2>
void test_multistream_service(const string& model_name_1, const string& model_name_2) {
    MultiStreamService pair_service;
    pair_service.Listen<M1, M2>(model_name_1, model_name_2);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept();
}

template<typename M1, typename M2>
void test_reef_service(const string& model_name_1, const string& model_name_2) {
    ReefService pair_service;
    pair_service.Listen<M1, M2>(model_name_1, model_name_2);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept();
}

template<typename M1, typename M2>
void test_krisp_service(const string& model_name_1, const string& model_name_2) {
    KrispService pair_service;
    pair_service.Listen<M1, M2>(model_name_1, model_name_2);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept();
}

template<typename M1, typename M2>
void test_fineco_service(const string& model_name_1, const string& model_name_2) {
    FineCoServiceV1 pair_service;
    pair_service.Listen<M1, M2>(model_name_1, model_name_2);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(2, false);
}

template<typename M1, typename M2>
void test_fineco_with_candidate_scheduler_service(const string& model_name_1, const string& model_name_2) {
    FineCoServiceV1 pair_service;
    pair_service.Listen<M1, M2>(model_name_1, model_name_2);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(2, true);
}

template<typename M1, typename M2>
void compare_service(const string& model_name_1, const string& model_name_2, int task) {
    switch (task) {
        case 0:
            cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            cout << model_name_1 << "    " << model_name_2 << endl;
            cout << "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++" << endl;
            test_multistream_service<M1, M2>(model_name_1, model_name_2);
            break;
        case 1:
            test_reef_service<M1, M2>(model_name_1, model_name_2);
            break;
        case 2:
            test_krisp_service<M1, M2>(model_name_1, model_name_2);
            break;
        case 3:
            test_fineco_service<M1, M2>(model_name_1, model_name_2);
            break;
        default:
            test_fineco_with_candidate_scheduler_service<M1, M2>(model_name_1, model_name_2);
    }
    cout << " " << endl;
    cout << " " << endl;
}

int main(int argc, char *argv[]) {
    int i = atoi(argv[1]), j = atoi(argv[2]);
    switch (i) {
        case 0:
            compare_service<ResNet, VGG>("ResNet50", "VGG19", j);
            break;
        case 1:
            compare_service<ResNet, MobileNet>("ResNet50", "MobileNet", j);
            break;
        case 2:
            compare_service<ResNet, DarkNet>("ResNet50", "DarkNet", j);
            break;
        case 3:
            compare_service<VGG, MobileNet>("VGG19", "MobileNet", j);
            break;
        case 4:
            compare_service<VGG, DarkNet>("VGG19", "DarkNet", j);
            break;
        default:
            compare_service<MobileNet, DarkNet>("MobileNet" , "DarkNet", j);
    }
}