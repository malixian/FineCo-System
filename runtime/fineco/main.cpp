#include "test/TestPairModel.cpp"
#include "test/TestTripleModels.cpp"
#include "test/TestQuadModels.cpp"
#include "test/TestCoKernel.cpp"
#include "test/TestOneModel.cpp"
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
#include "src/services/multi_stream.cpp"
#include "src/services/krisp.cpp"
#include "src/services/reef.cpp"

void test_triple_models() {
    for(int stream=1; stream<5; stream++) {
        TestTripleModel test_triple_model("VGG19", "AlexNet", "ResNet152");
        cout<<"stream number : " + to_string(stream) <<endl;
        int task_num = 10;
        test_triple_model.ServeBench<VGG, AlexNet, ResNet>(5, 20, 5, 100, stream);
    }
}

void test_triple_models_random() {
    int stream = 2;
    TestTripleModel test_triple_model("VGG19", "AlexNet", "ResNet152");
    cout<<"stream number : " + to_string(stream) <<endl;
    int task_num = 10;
    test_triple_model.ServeRandom<VGG, AlexNet, ResNet>(20, 80, 20, 100, stream);
}

void test_triple_models_default() {
    int stream = 1;
    TestTripleModel test_triple_model("VGG19", "AlexNet", "ResNet152");
    cout<<"stream number : " + to_string(stream) <<endl;
    int task_num = 10;
    test_triple_model.ServeDefault<VGG, AlexNet, ResNet>(10, 40, 10, 100, stream);
}

void test_quad_models() {
    for(int stream=1; stream<2; stream++) {
        TestQuadModel test_quad_model("DarkNet", "MobileNet", "VGG16", "ResNet50");
        cout<<"stream number : " + to_string(stream) <<endl;
        int task_num = 10;
        test_quad_model.ServeBench<DarkNet, MobileNet, VGG, ResNet>(10, 10, 10, 10, 100, stream);
    }
}

void test_co_kernels() {
    TestCoKernel test("VGG19", "VGG19");
    
    /*
    for(int layer=0; layer<10; layer++){
        cout<<"===== Layer :" << layer << "========="<<endl;
        for(int i=0; i<5; i++){
            test.KernelTest<ResNet, ResNet>(layer, layer, i, i);
        } 
    }
    */
    
    test.KernelBench<VGG, VGG>();
    /*
    test.KernelTest<ResNet, ResNet>(4, 4, 4, 1);
    
    test.KernelTest<ResNet, ResNet>(12, 12, 5, 1);
    test.KernelTest<ResNet, ResNet>(13, 13, 5, 1);
    test.KernelTest<ResNet, ResNet>(14, 14, 5, 1);
    test.KernelTest<ResNet, ResNet>(15, 15, 5, 1);

    test.KernelTest<ResNet, ResNet>(24, 24, 5, 1);
    test.KernelTest<ResNet, ResNet>(25, 25, 5, 1);
    test.KernelTest<ResNet, ResNet>(26, 26, 5, 1);
    test.KernelTest<ResNet, ResNet>(27, 27, 5, 1);

    test.KernelTest<ResNet, ResNet>(44, 44, 5, 1);
    test.KernelTest<ResNet, ResNet>(45, 45, 5, 1);
    
    test.KernelTest<ResNet, ResNet>(46, 46, 0, 1);
    */
    //test.KernelTest<ResNet, ResNet>(12, 12, 5, 1);
}

void test_co_single_kernel() {
    TestCoKernel test("ResNet34", "ResNet34");
    //test.KernelTest<VGG, DarkNet>(8, 9, 1, 1);
    test.KernelTest<ResNet, ResNet>(4, 4, 5, 2);

}

void test_one_model() {
    TestOneModel test;
    test.IsolationRun<ResNet>("ResNet50");
}

void model_pair_test() {
    TestPairModel test_co_model("ResNet50", "MobileNet");
    int stream_num_1 = 1;
    int stream_num_2 = 5;
    test_co_model.ServeBench<ResNet, MobileNet>(1, stream_num_1, stream_num_2);
}

void test_fcfs_service() {
    FCFSService pair_service;
    pair_service.Listen<ResNet, MobileNet>("ResNet50", "MobileNet");
    //pair_service.GeneratePoissonRequest(300);
    pair_service.GenerateTestSeq(30);
    //pair_service.GenerateQoSSeq(30);
    pair_service.Accept();
}

void test_multistream_service(int stream_num) {
    MultiStreamService pair_service;
    pair_service.Listen<ResNet, MobileNet>("ResNet50", "MobileNet");
    //pair_service.GenerateQoSSeq(30);
    //pair_service.GeneratePoissonRequest(300);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(stream_num);
}

void test_reef_service(int stream_num) {
    ReefService pair_service;
    pair_service.Listen<ResNet, MobileNet>("ResNet50", "MobileNet");
    //pair_service.GeneratePoissonRequest(300);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(stream_num);
}

void test_krisp_service(int stream_num) {
    KrispService pair_service;
    pair_service.Listen<ResNet, MobileNet>("ResNet50", "MobileNet");
    pair_service.GenerateTestSeq(30);
    //pair_service.GenerateQoSSeq(300);
    //pair_service.GeneratePoissonRequest(300);
    pair_service.Accept(stream_num);
}

void test_fineco_service(int stream_num) {
    FineCoServiceV1 pair_service;
    pair_service.Listen<ResNet, MobileNet>("ResNet50", "MobileNet");
    //pair_service.GeneratePoissonRequest(300);
    pair_service.GenerateTestSeq(30);
    //pair_service.GenerateQoSSeq(30);
    pair_service.Accept(stream_num, true);
}

void test_fineco_with_candidate_scheduler_service() {
    FineCoServiceV1 pair_service;
    pair_service.Listen<ResNet, DarkNet>("ResNet50", "DarkNet");
    pair_service.GenerateTestSeq(30);
    //pair_service.GenerateQoSSeq(30);
    pair_service.Accept(true);
}

void test_kernel_latency() {
    TestKernelLatency test;
    test.RunModel<MobileNet>("MobileNet");      
    test.RunModel<ResNet>("ResNet50");
    test.RunModel<ResNet>("ResNet152");
    test.RunModel<Bert>("Bert");
    test.RunModel<VGG>("VGG19");                      
    test.RunModel<DarkNet>("DarkNet");         
    test.RunModelBySpecComb<VGG>("VGG19");
}



int main(int argc, char *argv[]) 

{  
    test_fcfs_service(); 
    test_multistream_service(2);
    test_reef_service(2);
    test_krisp_service(2);
    test_fineco_service(3);

    return 0;
}

