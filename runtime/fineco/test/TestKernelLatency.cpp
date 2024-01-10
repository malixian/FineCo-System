#include<algorithm>
using namespace std;

class TestKernelLatency : public Test {

 public:

    template<typename M>

    void RunModel(const string task_name) {
        shared_ptr<M> _model(new M(task_name));
        _model->InitModel(CUDA);
        
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
            cout<<"layer idx: "<<r.layer_idx<<" candidate idx:"<<r.candidate_idx<<" SM size:"<<r.size <<" latency:"<<r.latency<<endl;
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
        _model->InitModel(CUDA);
        
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

};
