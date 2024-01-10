#include "service/service.h"
#include "common/thread_pool.hpp"
#include <algorithm>

class PriorityService : public PairService {
 public:  

    template<typename M1, typename M2>
    void Listen(const string& model_name_1, const string& model_name_2) {
        _model_1 = std::make_shared<M1>(model_name_1);
        _model_1->InitModel(ROCM);

        _model_2 = std::make_shared<M2>(model_name_2);
        _model_2->InitModel(ROCM);
    }    
 
    // 重载，每次组合手动设置高优先级的 right size，用来测试更优的资源用量
    template<typename M1, typename M2>
    void Listen(const string& model_name_1, const string& model_name_2, int right_size) {
        _model_1 = std::make_shared<M1>(model_name_1);
        _model_1->InitModel(ROCM);

        _model_2 = std::make_shared<M2>(model_name_2);
        _model_2->InitModel(ROCM);
        _model_2->SetRightSize(right_size);
    }    
 
    void Accept() {
        // cout<<"==== Start Priority Service ========"<<endl;

        // acquire candidate cnt, then select the last candidate
        vector<int> candidate_cnt_1;
        _model_1->GetAllLayerCandidateCnt(candidate_cnt_1);
        vector<int> candidate_selector_1;
        ServiceGenHighQoSCombination(candidate_cnt_1, candidate_selector_1);

        vector<int> candidate_cnt_2;
        _model_2->GetAllLayerCandidateCnt(candidate_cnt_2);
        vector<int> candidate_selector_2;
        ServiceGenHighQoSCombination(candidate_cnt_2, candidate_selector_2);

        // prepare stream with cuMask
        // each cuMask_1[i] and cuMask_2[j] is isolated <-> i <= j
        auto deviceApi_1 = GetBackendHandle(_model_1->GetModelRuntime());
        auto deviceApi_2 = GetBackendHandle(_model_2->GetModelRuntime());

        // low priority, CU < 50%, num(CU) = 8 * (index + 1)
        uint32_t cuMask_1[][4] = {{0x0, 0x0, 0x0, 0xff},
                                  {0x0, 0x0, 0x0, 0xffff},
                                  {0x0, 0x0, 0x0, 0xffffff},
                                  {0x0, 0x0, 0xff, 0xffffff},
                                  {0x0, 0x0, 0xffff, 0xffffff},
                                  {0x0, 0x0, 0xffffff, 0xffffff},
                                  {0x0, 0x0, 0xffffffff, 0xffffff}
                                  };
        // high priority, CU > 50%, num(CU) = 8 * (14 - index)
        uint32_t cuMask_2[][4] = {{0xffffffff, 0xffffffff, 0xffffffff, 0x00ffff00},
                                  {0xffffffff, 0xffffffff, 0xffffffff, 0x00ff0000},
                                  {0xffffffff, 0xffffffff, 0xffffffff, 0x0},
                                  {0xffffffff, 0xffffffff, 0xffffff00, 0x0},
                                  {0xffffffff, 0xffffffff, 0xffff0000, 0x0},
                                  {0xffffffff, 0xffffffff, 0xff000000, 0x0},
                                  {0xffffffff, 0xffffffff, 0x0, 0x0}
                                  };

        int index = 14 - _model_2->GetRightSize() / 8;
        DLStream stream_1, stream_2;
        stream_1 = deviceApi_1->CreateStream(4, cuMask_1[index]);
        stream_2 = deviceApi_2->CreateStream(4, cuMask_2[index]);

        // prepare input and output memory
        float *_output_memory_pool_1, *_input_memory_pool_1;
        float *_output_memory_pool_2, *_input_memory_pool_2;

        size_t out_mem_size_1 = sizeof(float) * _model_1->GetOutputSize();
        size_t input_mem_size_1 = sizeof(float) * _model_1->GetInputSize();
        size_t out_mem_size_2 = sizeof(float) * _model_2->GetOutputSize();
        size_t input_mem_size_2 = sizeof(float) * _model_2->GetInputSize();
        
        _output_memory_pool_1 = (float*)malloc(out_mem_size_1);
        _input_memory_pool_1 = (float*)malloc(input_mem_size_1);
        RandomInit(_input_memory_pool_1, _model_1->GetInputSize());

        _output_memory_pool_2 = (float*)malloc(out_mem_size_2);
        _input_memory_pool_2 = (float*)malloc(input_mem_size_2);
        RandomInit(_input_memory_pool_1, _model_1->GetInputSize());

        vector<future<float>> results_1, results_2;
        // vector<float> results_2;
        vector<float> wait_time_1;
        vector<float> wait_time_2;
        int arrival_interval_us = 10000;
        int model_1_number = 0;
        int model_2_number = 0;
        int64_t first_submit_timestamp = 0;

        ThreadPool client_pool(2);

        // run
        auto start = system_clock::now();
        while(_high_request_queue.size() > 0) {
            auto cur_request = _high_request_queue[0];
            auto submit_time = chrono::duration_cast<chrono::microseconds>(chrono::system_clock::now().time_since_epoch()).count();
            if (first_submit_timestamp == 0) {
                first_submit_timestamp = submit_time;
            }
            auto arrivel_time = cur_request->GetGeneTimestep() * arrival_interval_us + first_submit_timestamp;
            float wait_time = (submit_time - arrivel_time)/1000;
            if (submit_time >= arrivel_time) {
                if (cur_request->GetModelKind() == M1) {
                    results_1.emplace_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_selector_1),
                        _output_memory_pool_1, _input_memory_pool_1, input_mem_size_1, stream_1, true));
                    wait_time_1.emplace_back(wait_time);
                } else if (cur_request->GetModelKind() == M2) {
                    results_2.emplace_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_selector_2),
                        _output_memory_pool_2, _input_memory_pool_2, input_mem_size_2, stream_2, true));
                    wait_time_2.emplace_back(wait_time);
                }
                _high_request_queue.pop_front();
            }
        }

        model_1_number = results_1.size();
        model_2_number = results_2.size();
        
        // statisic result data
        auto end   = system_clock::now();
        // auto duration = duration_cast<milliseconds>(end - start).count();
        // float throughput = (_model_1->GetModelComputation() * model_1_number + _model_2->GetModelComputation() * model_2_number) * 1000 / duration;
        // cout << "run cost " << duration << "ms" << " throughput:" << throughput << "GFLOPs" <<endl;
        
        
        float model_1_avg_lat = 0.0;
        float model_2_avg_lat = 0.0;
        float model_1_avg_lat_no_wait = 0.0;
        float model_2_avg_lat_no_wait = 0.0;
        float avg_wait_time = 0;

        for (int i = 0; i < model_1_number; i++) {
            float wait_time = wait_time_1[i];
            avg_wait_time += wait_time;
            float latency = results_1[i].get();
            model_1_avg_lat += (latency + wait_time);
            model_1_avg_lat_no_wait += latency;
        }
        model_1_avg_lat /= model_1_number;
        model_1_avg_lat_no_wait /= model_1_number;

        for (int i = 0; i < model_2_number; i++) {
            float wait_time = wait_time_2[i];
            avg_wait_time += wait_time;
            float latency = results_2[i].get();
            model_2_avg_lat += (latency + wait_time);
            model_2_avg_lat_no_wait += latency;
        }
        model_2_avg_lat /= model_2_number;
        model_2_avg_lat_no_wait /=  model_2_number;
        avg_wait_time /= (wait_time_1.size() + wait_time_2.size());

        cout << model_2_avg_lat << endl;
        // cout<<"model 1 " << _model_1->GetModelName() << " avg latency: " << model_1_avg_lat << "   model 2 " << _model_2->GetModelName() << " avg latency: " << model_2_avg_lat << endl;
        // cout<<"model 1 avg latency no wait: "<<model_1_avg_lat_no_wait<<"   model 2 avg latency no wait:" <<model_2_avg_lat_no_wait<<endl;
        // cout<<"average wait time: "<<avg_wait_time<<endl;

        
    }

};
