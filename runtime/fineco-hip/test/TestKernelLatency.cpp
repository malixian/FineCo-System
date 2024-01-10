#include<algorithm>
using namespace std;

class TestKernelLatency : public Test {

 public:

    template<typename M>

    void RunModel(const string task_name) {
        shared_ptr<M> _model(new M(task_name));
        _model->InitModel(ROCM);
        vector<int> task_candidate_cnt;
        _model->GetAllLayerCandidateCnt(task_candidate_cnt);
        vector<vector<int>> candidate_size_list;
        _model->GetAllLayerCandidateSize(candidate_size_list);
        vector<int> ret_candidate_list;
        struct KernelAttr {
            int layer_idx;
            int candidate_idx;
            float latency;
            int size;
            
        };

        struct ImplAttr {
            int layer_idx;
            int candidate_idx;
            int size;
            float latency;
            float best_latency;
        };

        vector<KernelAttr> results;
        vector<ImplAttr> impl_list;

        // Layer warm up, init memory
        for (int layer_idx=0; layer_idx<task_candidate_cnt.size();layer_idx++) {
            for(int candidate_idx=0; candidate_idx<task_candidate_cnt[layer_idx]; candidate_idx++) {
                //cout<<"layer: "<<layer_idx<<" candidate: "<<candidate_idx<<endl;
                auto a1 = async(&Model::LayerRunWarmup, _model, layer_idx, candidate_idx, nullptr);
                a1.get();
            }   
        }

        cout<<"========= warm up finish ========" <<endl;

        float effect_ratio = 1.8;
        cout<<"SLO scale is: "<<effect_ratio<<endl;

        
        for(int layer_idx=0; layer_idx<task_candidate_cnt.size();layer_idx++) {
            int best_effect_candidate_idx = -1;
            int best_effect_candidate_size = -1;
            
            float default_impl_latency = 0.0;
            
            int repeat_num = 100;
            int iter_num = 1;
            //cout<<"layer id:"<<layer_idx<<" candidate cnt: "<< task_candidate_cnt[layer_idx]<<endl;
            /*
            if (task_candidate_cnt[layer_idx] <= 1){
                continue;
            }
            */ 
            for(int candidate_idx=0; candidate_idx<task_candidate_cnt[layer_idx]; candidate_idx++){
                //cout<<"candidate cnt :"<<task_candidate_cnt[layer_idx]<<" current candidate idx:"<<candidate_idx<<endl;
                for(int iter=0; iter<iter_num; iter++)
                    future<float> task_latency = async(&Model::LayerRun, _model, layer_idx, candidate_idx, repeat_num, nullptr);
                future<float> task_latency = async(&Model::LayerRun, _model, layer_idx, candidate_idx, repeat_num, nullptr);
                float ret = task_latency.get() / repeat_num * 1000;
                KernelAttr attr;
                int sm_size = candidate_size_list[layer_idx][candidate_idx];
                attr.layer_idx = layer_idx;
                attr.latency = ret;
		attr.size = candidate_size_list[layer_idx][candidate_idx];
                attr.candidate_idx = candidate_idx;
                attr.size = sm_size;
                if (candidate_idx == 0) {
                    default_impl_latency = ret;
                    //continue;
                    //cout<<"layer: "<<layer_idx<<" candidate 0 block: "<<candidate_size_list[layer_idx][candidate_idx]<<" lat: "<<ret<<endl;;
                } else {
                    //cout<<"effect ratio value: "<<effect_ratio * default_impl_latency<<endl;
                    if((ret <= (effect_ratio * default_impl_latency)) && best_effect_candidate_idx == -1) {
                        best_effect_candidate_idx = candidate_idx;
                        best_effect_candidate_size = candidate_size_list[layer_idx][candidate_idx];
                        ImplAttr impl_attr;
                        impl_attr.layer_idx = layer_idx;
                        impl_attr.size = best_effect_candidate_size;
                        impl_attr.candidate_idx = best_effect_candidate_idx;
                        impl_attr.latency = ret;
                        impl_attr.best_latency = default_impl_latency;
                        impl_list.emplace_back(impl_attr);
                        //break;
                    }
                }
                results.emplace_back(attr);
            }
            if (best_effect_candidate_idx == -1) {
                ImplAttr impl_attr;
                impl_attr.layer_idx = layer_idx;
                impl_attr.size = 100;
                impl_attr.candidate_idx = task_candidate_cnt[layer_idx]-1;
                impl_attr.latency = 0.1;
                impl_attr.best_latency = default_impl_latency;
                impl_list.emplace_back(impl_attr);
            }
        }

        //sort(results.begin(), results.end(), [](KernelAttr x, KernelAttr y) {return x.latency > y.latency;});

        
        for (auto r : results) {
            cout<<"layer idx: "<<r.layer_idx<<" candidate idx:"<<r.candidate_idx<<" CU size:"<<r.size<<" latency:"<<r.latency<<endl;
        }
        
        cout<<"=========== best effective =============="<<endl;
        for (auto& r : impl_list) {
            //cout<<"layer idx: "<<r.layer_idx<<" candidate idx:"<<r.candidate_idx<<" size:"<<r.size<<" effective latency:"<<r.latency<<" best latency:"<<r.best_latency<<endl;
            if(r.size != 100)
                cout<<"layer idx: "<<r.layer_idx<<" size: "<<r.size<<" effective latency: "<<r.latency<<endl;
            else
                cout<<"layer idx: "<<r.layer_idx<<" size: "<<r.size<<" effective latency: "<<r.best_latency<<endl;
        }

        for (auto& r : impl_list) {
            cout<<r.size<<", ";
        }
        cout<<"\n";

        for (auto& r : impl_list) {
            //cout<<"layer idx: "<<r.layer_idx<<" candidate idx:"<<r.candidate_idx<<" size:"<<r.size<<" effective latency:"<<r.latency<<" best latency:"<<r.best_latency<<endl;
            if(r.size != 100)
                cout<<r.latency<<", ";
            else
                cout<<r.best_latency<<", ";
        }

        cout<<"candidate idx "<<endl;
        for (auto& r : impl_list) {  
            cout<<r.candidate_idx<<", ";
        }

        
        cout<<"\n";
    }

    template<typename M>
    void RunModelBySpecComb(const string task_name) {

        vector<int> comb_list{3, 4, 0, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 6, 6, 0, 5, 5, 5, 5, 0, 0, 0, 0};
        shared_ptr<M> _model(new M(task_name));
        _model->InitModel(ROCM);
        
        vector<int> task_candidate_cnt;
        _model->GetAllLayerCandidateCnt(task_candidate_cnt);
        vector<vector<int>> candidate_size_list;
        _model->GetAllLayerCandidateSize(candidate_size_list);
        vector<int> ret_candidate_list;
        struct KernelAttr {
            int layer_idx;
            int candidate_idx;
            float latency;
            int size;
            
        };

        vector<KernelAttr> results;

        // Layer warm up, init memory
        for (int layer_idx=0; layer_idx<task_candidate_cnt.size();layer_idx++) {
            int candidate_idx = comb_list[layer_idx];
            auto a1 = async(&Model::LayerRunWarmup, _model, layer_idx, candidate_idx, nullptr);
            a1.get(); 
        }

        cout<<"========= warm up finish ========" <<endl;

        float effect_ratio = 1.8;
        cout<<"SLO scale is: "<<effect_ratio<<endl;

        for(int layer_idx=0; layer_idx<task_candidate_cnt.size();layer_idx++) {
            float default_impl_latency = 0.0;
            
            int repeat_num = 100;
            //cout<<"layer id:"<<layer_idx<<" candidate cnt: "<< task_candidate_cnt[layer_idx]<<endl;
            int candidate_idx = comb_list[layer_idx];
            future<float> task_latency = async(&Model::LayerRun, _model, layer_idx, candidate_idx, repeat_num, nullptr);
            float ret = task_latency.get() / repeat_num * 1000;
            KernelAttr attr;
            attr.layer_idx = layer_idx;
            attr.latency = ret;
            attr.candidate_idx = candidate_idx;
            auto sm_size = candidate_size_list[layer_idx][candidate_idx];
            if (sm_size >= 108)
                sm_size = 108;
            attr.size = sm_size;
            results.emplace_back(attr);
        }

        // print sm number
        for (auto r : results) {
            cout<<r.size<<", ";
        }
        cout<<"\n";

        for (auto r : results) {
            cout<<r.latency<<", ";
        }
        cout<<"\n";

    }
    
    template<typename M>
    void RunKernelCuMask(const string task_name) {
        shared_ptr<M> _model(new M(task_name));
        _model->InitModel(ROCM);
        vector<int> task_candidate_cnt;
        _model->GetAllLayerCandidateCnt(task_candidate_cnt);
        vector<vector<int>> candidate_size_list;
        _model->GetAllLayerCandidateSize(candidate_size_list);
        struct KernelAttr {
            int layer_idx;
            int candidate_idx;
            float latency;
            
        };

        struct ImplAttr {
            int layer_idx;
            int candidate_idx;
            int size;
            float latency;
            float best_latency;
        };

        vector<KernelAttr> results;
        vector<ImplAttr> impl_list;

        // Layer warm up, init memory
        for (int layer_idx = 0; layer_idx < task_candidate_cnt.size(); layer_idx++) {
            for (int candidate_idx = 0; candidate_idx < task_candidate_cnt[layer_idx]; candidate_idx++) {
                // cout << "layer: " << layer_idx << " candidate: " << candidate_idx << " candidate_size:" << candidate_size_list[layer_idx][candidate_idx] << endl;
                auto a1 = async(&Model::LayerRunWarmup, _model, layer_idx, candidate_idx, nullptr);
                a1.get();
            }   
        }

        // cout << "========= warm up finish ========\n" << endl;

        const uint32_t mask[][4] = {
            {0x1, 0x0, 0x0, 0x0},
            {0x11, 0x0, 0x0, 0x0},
            {0x111, 0x0, 0x0, 0x0},
            {0x1111, 0x0, 0x0, 0x0},
            {0x11111, 0x0, 0x0, 0x0},
            {0x111111, 0x0, 0x0, 0x0},
            {0x1111111, 0x0, 0x0, 0x0},
            {0x11111111, 0x0, 0x0, 0x0},
            {0x11111111, 0x1, 0x0, 0x0},
            {0x11111111, 0x11, 0x0, 0x0},
            {0x11111111, 0x111, 0x0, 0x0},
            {0x11111111, 0x1111, 0x0, 0x0},
            {0x11111111, 0x11111, 0x0, 0x0},
            {0x11111111, 0x111111, 0x0, 0x0},
            {0x11111111, 0x1111111, 0x0, 0x0},
            {0x11111111, 0x11111111, 0x0, 0x0},
            {0x11111111, 0x11111111, 0x1, 0x0},
            {0x11111111, 0x11111111, 0x11, 0x0},
            {0x11111111, 0x11111111, 0x111, 0x0},
            {0x11111111, 0x11111111, 0x1111, 0x0},
            {0x11111111, 0x11111111, 0x11111, 0x0},
            {0x11111111, 0x11111111, 0x111111, 0x0},
            {0x11111111, 0x11111111, 0x1111111, 0x0},
            {0x11111111, 0x11111111, 0x11111111, 0x0},
            {0x11111111, 0x11111111, 0x11111111, 0x1},
            {0x11111111, 0x11111111, 0x11111111, 0x11},
            {0x11111111, 0x11111111, 0x11111111, 0x111},
            {0x11111111, 0x11111111, 0x11111111, 0x1111},
            {0x11111111, 0x11111111, 0x11111111, 0x11111},
            {0x11111111, 0x11111111, 0x11111111, 0x111111},
            {0x33333333, 0x13333333, 0x0, 0x0},
            {0x33333333, 0x33333333, 0x0, 0x0},
            {0x33333333, 0x33333333, 0x1, 0x0},
            {0x33333333, 0x33333333, 0x3, 0x0},
            {0x33333333, 0x33333333, 0x13, 0x0},
            {0x33333333, 0x33333333, 0x33, 0x0},
            {0x33333333, 0x33333333, 0x133, 0x0},
            {0x33333333, 0x33333333, 0x333, 0x0},
            {0x33333333, 0x33333333, 0x1333, 0x0},
            {0x33333333, 0x33333333, 0x3333, 0x0},
            {0x33333333, 0x33333333, 0x13333, 0x0},
            {0x33333333, 0x33333333, 0x33333, 0x0},
            {0x33333333, 0x33333333, 0x133333, 0x0},
            {0x33333333, 0x33333333, 0x333333, 0x0},
            {0x33333333, 0x33333333, 0x1333333, 0x0},
            {0x33333333, 0x33333333, 0x3333333, 0x0},
            {0x33333333, 0x33333333, 0x13333333, 0x0},
            {0x33333333, 0x33333333, 0x33333333, 0x0},
            {0x33333333, 0x33333333, 0x33333333, 0x1},
            {0x33333333, 0x33333333, 0x33333333, 0x3},
            {0x33333333, 0x33333333, 0x33333333, 0x13},
            {0x33333333, 0x33333333, 0x33333333, 0x33},
            {0x33333333, 0x33333333, 0x33333333, 0x133},
            {0x33333333, 0x33333333, 0x33333333, 0x333},
            {0x33333333, 0x33333333, 0x33333333, 0x1333},
            {0x33333333, 0x33333333, 0x33333333, 0x3333},
            {0x33333333, 0x33333333, 0x33333333, 0x13333},
            {0x33333333, 0x33333333, 0x33333333, 0x33333},
            {0x33333333, 0x33333333, 0x33333333, 0x133333},
            {0x33333333, 0x33333333, 0x33333333, 0x333333},
            {0x77777777, 0x77777777, 0x17777, 0x0},
            {0x77777777, 0x77777777, 0x37777, 0x0},
            {0x77777777, 0x77777777, 0x77777, 0x0},
            {0x77777777, 0x77777777, 0x177777, 0x0},
            {0x77777777, 0x77777777, 0x377777, 0x0},
            {0x77777777, 0x77777777, 0x777777, 0x0},
            {0x77777777, 0x77777777, 0x1777777, 0x0},
            {0x77777777, 0x77777777, 0x3777777, 0x0},
            {0x77777777, 0x77777777, 0x7777777, 0x0},
            {0x77777777, 0x77777777, 0x17777777, 0x0},
            {0x77777777, 0x77777777, 0x37777777, 0x0},
            {0x77777777, 0x77777777, 0x77777777, 0x0},
            {0x77777777, 0x77777777, 0x77777777, 0x1},
            {0x77777777, 0x77777777, 0x77777777, 0x3},
            {0x77777777, 0x77777777, 0x77777777, 0x7},
            {0x77777777, 0x77777777, 0x77777777, 0x17},
            {0x77777777, 0x77777777, 0x77777777, 0x37},
            {0x77777777, 0x77777777, 0x77777777, 0x77},
            {0x77777777, 0x77777777, 0x77777777, 0x177},
            {0x77777777, 0x77777777, 0x77777777, 0x377},
            {0x77777777, 0x77777777, 0x77777777, 0x777},
            {0x77777777, 0x77777777, 0x77777777, 0x1777},
            {0x77777777, 0x77777777, 0x77777777, 0x3777},
            {0x77777777, 0x77777777, 0x77777777, 0x7777},
            {0x77777777, 0x77777777, 0x77777777, 0x17777},
            {0x77777777, 0x77777777, 0x77777777, 0x37777},
            {0x77777777, 0x77777777, 0x77777777, 0x77777},
            {0x77777777, 0x77777777, 0x77777777, 0x177777},
            {0x77777777, 0x77777777, 0x77777777, 0x377777},
            {0x77777777, 0x77777777, 0x77777777, 0x777777},
            {0xffffffff, 0xffffffff, 0x7ffffff, 0x0},
            {0xffffffff, 0xffffffff, 0xfffffff, 0x0},
            {0xffffffff, 0xffffffff, 0x1fffffff, 0x0},
            {0xffffffff, 0xffffffff, 0x3fffffff, 0x0},
            {0xffffffff, 0xffffffff, 0x7fffffff, 0x0},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x0},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x1},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x3},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x7},
            {0xffffffff, 0xffffffff, 0xffffffff, 0xf},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x1f},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x3f},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x7f},
            {0xffffffff, 0xffffffff, 0xffffffff, 0xff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x1ff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x3ff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x7ff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0xfff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x1fff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x3fff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x7fff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0xffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x1ffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x3ffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x7ffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0xfffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x1fffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x3fffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0x7fffff},
            {0xffffffff, 0xffffffff, 0xffffffff, 0xffffff}
        };

        auto handle = GetBackendHandle(_model->GetModelRuntime());
        for(int layer_idx = 0; layer_idx < task_candidate_cnt.size(); layer_idx++) {
            int best_effect_candidate_idx = -1;
            int best_effect_candidate_size = -1;
            float effect_ratio = 1.6;
            float default_impl_latency = 0.0;
            
            int repeat_num = 100;
            int iter_num = 1;
            //cout<<"layer id:"<<layer_idx<<" candidate cnt: "<< task_candidate_cnt[layer_idx]<<endl;
            if (task_candidate_cnt[layer_idx] <= 1){
                continue;
            } 
            for (int candidate_idx = 0; candidate_idx < task_candidate_cnt[layer_idx]; candidate_idx++) {             

                DLStream stream = nullptr;
                if (candidate_idx > 0 && candidate_size_list[layer_idx][candidate_idx] <= 120) {
                    uint32_t *cuMask = (uint32_t *)mask[candidate_size_list[layer_idx][candidate_idx]];
                    stream = handle->CreateStream(4, cuMask);
                }
                for (int iter = 0; iter < iter_num; iter++)
                    future<float> task_latency = async(&Model::LayerRun, _model, layer_idx, candidate_idx, repeat_num, stream);
                future<float> task_latency = async(&Model::LayerRun, _model, layer_idx, candidate_idx, repeat_num, stream);
                float ret = task_latency.get() / repeat_num * 1000;
                cout << dec << "layer: " << layer_idx << ", candidate: " << candidate_idx << ", latency: " << ret << endl;
                // cout << hex << "    " << cuMask[0] << " " << cuMask[1] << " " << cuMask[2] << " " << cuMask[3] << endl;
                if (stream)
                    handle->FreeStream(0, stream);
            }
            cout << endl;
        }
        
    }
};


