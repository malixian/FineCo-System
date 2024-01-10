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
    TestCoKernel test("ResNet50", "MobileNet");
    
    
    for(int layer=2; layer<10; layer++){
        cout<<"===== Layer :" << layer << "========="<<endl;
        for(int i=0; i<5; i++){
            test.KernelTest<ResNet, MobileNet>(layer, layer, i, i, 1);
        }
    }
    
    //test.KernelBench<VGG, VGG>();
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
    TestCoKernel test("DarkNet", "DarkNet");
    //test.KernelTest<VGG, DarkNet>(8, 9, 1, 1);
    // for(int i=2; i<32; i++)
    // test.KernelTest<MobileNet, MobileNet>(i, i, 0, 0);
    int i[] = {8, 10, 19, 21, 23};
    for (int j = 0; j < 5; j++)
        test.KernelTest<DarkNet, DarkNet>(i[j], i[j]);
}
    // TestCoKernel test("ResNet34", "ResNet34");
    //test.KernelTest<VGG, DarkNet>(8, 9, 1, 1);
    // test.KernelTest<ResNet, ResNet>(4, 4, 0, 3);

void test_one_kernel() {
    TestOneKernel test;
    test.KernelBench<AlexNet>("AlexNet");
    cout << "MobileNet begin" << endl;
    test.KernelBench<MobileNet>("MobileNet");
    cout << "ResNet50 begin" << endl;
    test.KernelBench<ResNet>("ResNet50");
    cout << "ResNet152 begin" << endl;
    test.KernelBench<ResNet>("ResNet152");
    cout << "Bert begin" << endl;
    test.KernelBench<Bert>("Bert");
    cout << "VGG19 begin" << endl;
    test.KernelBench<VGG>("VGG19");
    // cout << "DarkNet begin" << endl;
    // test.KernelBench<DarkNet>("DarkNet");       // param error, block size > 256
}

void test_one_model() {
    TestOneModel test;
    test.IsolationRun<VGG>("VGG19");
    test.IsolationRun<DarkNet>("DarkNet");
    test.IsolationRun<MobileNet>("MobileNet");
    test.IsolationRun<ResNet>("ResNet50");
    // test.IsolationRun<ResNet>("ResNet152");         // ResNet50
    // test.IsolationRun<Bert>("Bert");
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

void test_multistream_service() {
    MultiStreamService pair_service;
    pair_service.Listen<ResNet, DarkNet>("ResNet50", "DarkNet");
    pair_service.GenerateTestSeq(30);
    pair_service.Accept();
}

void test_reef_service(int stream_num=2) {
    ReefService pair_service;
    pair_service.Listen<ResNet, MobileNet>("ResNet50", "MobileNet");
    //pair_service.GeneratePoissonRequest(300);
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(stream_num);
}

void test_krisp_service() {
    KrispService pair_service;
    pair_service.Listen<ResNet, DarkNet>("ResNet50", "DarkNet");
    pair_service.GenerateTestSeq(30);
    pair_service.Accept();
}

void test_fineco_service(int stream_num=2) {
    FineCoServiceV1 pair_service;
    pair_service.Listen<ResNet, DarkNet>("ResNet50", "DarkNet");
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(stream_num, false);
}

void test_fineco_with_candidate_scheduler_service() {
    FineCoServiceV1 pair_service;
    pair_service.Listen<ResNet, DarkNet>("ResNet50", "DarkNet");
    pair_service.GenerateTestSeq(30);
    pair_service.Accept(true);
}

// void test_kernel_latency() {
//     FineCoServiceV1 service;
//     //test.RunModelBySpecComb<DarkNet>("DarkNet");
//     service.Listen<ResNet, DarkNet>("ResNet50", "DarkNet");
//     vector<int> candidate_list_1;
//     service.GetBestEffectImplList(1.8, "ResNet50", candidate_list_1);

//     vector<int> candidate_list_2;
//     service.GetBestEffectImplList(1.8, "DarkNet", candidate_list_2);

//     service.CandidateSchedulerV2(candidate_list_1, candidate_list_2);

// }

void test_kernel_latency() {
    TestKernelLatency test;
    // test.RunModel<AlexNet>("AlexNet");
    cout << endl << endl << "MobileNet" << endl << endl;
    test.RunModel<MobileNet>("MobileNet");
    cout << endl << endl << "ResNet50" << endl << endl;
    test.RunModel<ResNet>("ResNet50");
    // test.RunModel<ResNet>("ResNet152");
    // test.RunModel<Bert>("Bert");
    cout << endl << endl << "VGG19" << endl << endl;
    test.RunModel<VGG>("VGG19");
    cout << endl << endl << "DarkNet" << endl << endl;
    test.RunModel<DarkNet>("DarkNet");
    // test.RunModelBySpecComb<VGG>("VGG19");
}

void test_kernel_cuMask() {
    TestKernelLatency test;
    // test.RunKernelCuMask<AlexNet>("AlexNet");
    // test.RunKernelCuMask<MobileNet>("MobileNet");
    test.RunKernelCuMask<ResNet>("ResNet50");
    // test.RunKernelCuMask<ResNet>("ResNet152");
    // test.RunKernelCuMask<Bert>("Bert");
    // test.RunKernelCuMask<VGG>("VGG19");
    // test.RunKernelCuMask<DarkNet>("DarkNet");
}

void test_priority_service() {
    PriorityService test;
    test.Listen<AlexNet, AlexNet>("AlexNet", "AlexNet");
    test.GenerateFixedSeq(100);
    test.Accept();
}


int main(int argc, char *argv[]) 

{
    
    test_co_kernels();
    // test_one_model();
    //model_pair_test();
    // test_co_single_kernel();
    // test_one_kernel();

    // test_fineco_service();
    //test_reef_service();
    //test_fcfs_service();

    // test_multistream_service();
    // cout << " " << endl;
    // test_reef_service();
    // cout << " " << endl;
    // test_krisp_service();
    // cout << " " << endl;
    // test_fineco_service();
    // cout << " " << endl;
    // test_fineco_with_candidate_scheduler_service();

    // test_one_model();
    // test_kernel_latency();

    // test_priority_service();
    // test_service();

    return 0;
}

