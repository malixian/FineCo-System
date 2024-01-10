#include <chrono>

class TestCoKernelGroup : public Test {
 public:

    TestCoKernelGroup(const string& task_1_name, const string& task_2_name) : _task_1_name(task_1_name),\
     _task_2_name(task_2_name) {}
     
    template<typename M1, typename M2>
    void Test(KernelSlice& model_1_kernel_slice, KernelSlice& model_2_kernel_slice) {
        
        shared_ptr<M1> _model_1(new M1(_task_1_name));
        shared_ptr<M2> _model_2(new M2(_task_2_name));

        _model_1->InitModel(ROCM);
        _model_2->InitModel(ROCM);

        // warm up
        auto a1 = async(&Model::RunKernelGroup, _model_1, 0, 4 );
        auto a2 = async(&Model::LayerRunWarmup, _model_2, 0, 4 );
        float lat_1 = a1.get();
        float lat_2 = a2.get();
        // Get Task Number
        a1 = async(&Model::LayerRunWarmup, _model_1, model_1_layer, model_1_layer_candidate);
        a2 = async(&Model::LayerRunWarmup, _model_2, model_2_layer, model_2_layer_candidate);
        lat_1 = a1.get();
        lat_2 = a2.get();

        int request_num_1 = 1;
        int request_num_2 = 1;
        if(lat_1 > lat_2) {
            request_num_1 = 100;
            request_num_2 = 100 * (lat_1/lat_2);
        } else {
            request_num_2 = 100;
            request_num_1 = 100 * (lat_2/lat_1);
        }

        auto start = chrono::steady_clock::now();
        float task_1_ret = 0;
        float task_2_ret = 0;
        auto stream1 =  GetBackendHandle(ROCM)->CreateStream(0);
        auto stream2 =  GetBackendHandle(ROCM)->CreateStream(0);
        for (int i=0; i<100; i++) {
            future<float> task_1_latency = async(&Model::LayerRun, _model_1, model_1_layer, model_1_layer_candidate, 5, stream1);
            future<float> task_2_latency = async(&Model::LayerRun, _model_2, model_2_layer, model_2_layer_candidate, 5, stream2);
            task_1_ret += task_1_latency.get();
            task_2_ret += task_2_latency.get();
        }
        
    
        

        float tp_1 = (request_num_1 / task_1_ret) * 1000;
        float tp_2 = (request_num_2 / task_2_ret) * 1000;
        cout<<"candidate: "<<model_1_layer_candidate<<" Latency: "<<task_1_ret<< " TP: "<<tp_1<<\
        " candidate: "<<model_2_layer_candidate<<" Latency: "<<task_2_ret<<" TP: "<<tp_2<<endl;

        auto end = chrono::steady_clock::now();
        auto co_latency = chrono::duration_cast<chrono::microseconds>(end - start).count() / 1000;
        cout << "Co Running time: "<< co_latency << " ms" << endl;

    }

 private:
   string _task_1_name;
   string _task_2_name;


};