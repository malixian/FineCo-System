#include "service/service.h"
#include "common/thread_pool.hpp"
#include "common/gpu_stats.hpp"
#include <algorithm>
#include <unistd.h>

class FineCoServiceV1 : public PairService {
 public: 

    void Accept(int set_stream_num = 2, bool enable_cs=false) {
        cout<<"==== Start FineCoV1 Service ========"<<endl;

        DLStream test_stream = GetBackendHandle(_model_1->GetModelRuntime())->CreateStream(0);

        vector<int> candidate_cnt_1;
        _model_1->GetAllLayerCandidateCnt(candidate_cnt_1);
        vector<int> candidate_high_selector_1;
        this->GetBestEffectImplList(1.0, _model_name_1, candidate_high_selector_1, test_stream);
        vector<int> candidate_best_selector_1;
        ServiceGenBestCombination(_model_name_1, candidate_cnt_1, candidate_best_selector_1);
        
        vector<int> candidate_cnt_2;
        _model_2->GetAllLayerCandidateCnt(candidate_cnt_2);
        vector<int> candidate_high_selector_2;
        this->GetBestEffectImplList(1.0, _model_name_2, candidate_high_selector_2, test_stream);
        vector<int> candidate_best_selector_2;
        ServiceGenBestCombination(_model_name_2, candidate_cnt_2, candidate_best_selector_2);

        if (enable_cs) {
            cout<<"==== enable candidate scheduler ===="<<endl;
            this->CandidateSchedulerV3(candidate_high_selector_1, candidate_high_selector_2);
        }
        
        
        

        int low_stream_num = 1;
        int default_stream_num = set_stream_num;
        int pool_size = 5;
        ThreadPool client_pool(pool_size);

        
        vector<float*> _output_memory_pool_1(low_stream_num);
        vector<float*> _input_memory_pool_1(low_stream_num);
        vector<float*> _output_memory_pool_2(low_stream_num);
        vector<float*> _input_memory_pool_2(low_stream_num);

        vector<float*> h_output_memory_pool_1(default_stream_num);
        vector<float*> h_input_memory_pool_1(default_stream_num);
        vector<float*> h_output_memory_pool_2(default_stream_num);
        vector<float*> h_input_memory_pool_2(default_stream_num);



        vector<DLStream> _stream_pool(low_stream_num);
        vector<DLStream> _high_stream_pool(default_stream_num);


        size_t out_mem_size_1 = sizeof(float) * _model_1->GetOutputSize();
        size_t input_mem_size_1 = sizeof(float) * _model_1->GetInputSize();
        size_t out_mem_size_2 = sizeof(float) * _model_2->GetOutputSize();
        size_t input_mem_size_2 = sizeof(float) * _model_2->GetInputSize();


        for(int i=0; i<low_stream_num; i++) {
            float* out_h_1 = (float*)malloc(out_mem_size_1);
            _output_memory_pool_1[i] = out_h_1;
            float* in_h_1 = (float*)malloc(input_mem_size_2);
            RandomInit(in_h_1, _model_1->GetInputSize());
            _input_memory_pool_1[i] = in_h_1;

            float* out_h_2 = (float*)malloc(out_mem_size_2);
            _output_memory_pool_2[i] = out_h_2;
            float* in_h_2 = (float*)malloc(input_mem_size_2);
            RandomInit(in_h_2, _model_2->GetInputSize());
            _input_memory_pool_2[i] = in_h_2;

            _stream_pool[i] = GetBackendHandle(_model_1->GetModelRuntime())->CreateStream(0);
        }

        for(int i=0; i<default_stream_num; i++) {
            float* out_h_1 = (float*)malloc(out_mem_size_1);
            h_output_memory_pool_1[i] = out_h_1;
            float* in_h_1 = (float*)malloc(input_mem_size_2);
            RandomInit(in_h_1, _model_1->GetInputSize());
            h_input_memory_pool_1[i] = in_h_1;

            float* out_h_2 = (float*)malloc(out_mem_size_2);
            h_output_memory_pool_2[i] = out_h_2;
            float* in_h_2 = (float*)malloc(input_mem_size_2);
            RandomInit(in_h_2, _model_2->GetInputSize());
            h_input_memory_pool_2[i] = in_h_2;

            //_high_stream_pool[i] = GetBackendHandle(_model_1->GetModelRuntime())->CreateStreamWithPriority(-5, 0);
            _high_stream_pool[i] =  GetBackendHandle(_model_1->GetModelRuntime())->CreateStream(0);
        }
        

        
        
        int model_1_number = 0;
        int model_2_number = 0;

        // flexible sycn client threads
        int acc_model1_number = 0;
        

        float model_1_latency = 0.0 ;
        float model_2_latency = 0.0;
        float high_qos_latency = 0.0;
        float high_qos_latency_no_wait = 0.0;

        vector<float> model_1_lat_list_rt;
        vector<float> model_2_lat_list_rt;

        vector<float> model_1_lat_list_nowait_rt;
        vector<float> model_2_lat_list_nowait_rt;

        vector<float> model_1_lat_list_be;
        vector<float> model_2_lat_list_be;

        vector<float> model_1_wait_list;
        vector<float> model_2_wait_list;

        vector<float> tp_list;
        vector<float> co_latency_list;

        for(int i=0; i<20; i++) {
            auto results_1 = _model_1->RealRun(candidate_best_selector_1, _output_memory_pool_1[0],\
                    _input_memory_pool_1[0], input_mem_size_1, _stream_pool[0],true);
            if(i>=10) model_1_latency += results_1;
        }

        for(int i=0; i<20; i++) {
            auto results_2 = _model_2->RealRun(candidate_best_selector_2, _output_memory_pool_2[0],\
                    _input_memory_pool_2[0], input_mem_size_2, _stream_pool[0], true);
            if(i>=10) model_2_latency += results_2;
        }

        model_1_latency /= 10;
        model_2_latency /= 10;
        cout<<"model 1 isolation latency: "<<model_1_latency<<" model 2 isolation latency :"<<model_2_latency<<endl;
        
        int64_t first_submit_timestamp = 0;

        
        auto begin_all_ts = chrono::steady_clock::now();
        int total_requests_cnt = _total_request_queue.size();
        while(_total_request_queue.size() > 0) {
            //usleep(500);
            // concurrent submit request by stream number
            //cout<<"left request size: "<<_total_request_queue.size()<<endl;
            auto begin_ts = chrono::steady_clock::now();
            int submit_number = 0;
            int first_timestamp = _total_request_queue[0]->GetGeneTimestep();
            vector<future<float>> high_results_model;
            vector<future<float>> low_results_model;
            vector<shared_ptr<Request>> tmp_high_req;
            vector<shared_ptr<Request>> tmp_low_req;
            for(int ri=0; ri<_total_request_queue.size() && _total_request_queue[ri]->GetGeneTimestep() == first_timestamp; ri++) {
                auto cur_req = _total_request_queue[ri];
                if(cur_req->GetQoSKind() == HighQoS) {
                    if (cur_req->GetModelKind() == M1)
                        cur_req->SetEstimatedLatency(model_1_latency);
                    else if (cur_req->GetModelKind() == M2)
                        cur_req->SetEstimatedLatency(model_2_latency);
                    tmp_high_req.push_back(cur_req);
                } else if (cur_req->GetQoSKind() == LowQoS) {
                    if (cur_req->GetModelKind() == M1)
                        cur_req->SetEstimatedLatency(model_1_latency);
                    else if (cur_req->GetModelKind() == M2)
                        cur_req->SetEstimatedLatency(model_2_latency);
                    tmp_low_req.push_back(cur_req);
                }
            }


            // reorder high qos queue by latency from little to great
            sort(tmp_high_req.begin(), tmp_high_req.end(), \
                [](shared_ptr<Request> x,shared_ptr<Request> y){return x->GetEstimatedLatency()<y->GetEstimatedLatency();});

            sort(tmp_low_req.begin(), tmp_low_req.end(), \
                [](shared_ptr<Request> x,shared_ptr<Request> y){return x->GetEstimatedLatency()<y->GetEstimatedLatency();});
            
            int high_qos_cnt = tmp_high_req.size();
            int low_qos_cnt = tmp_low_req.size();
            int concurrent_cnt = high_qos_cnt + low_qos_cnt;

            for (int i=0; i<concurrent_cnt; i++) {
                _total_request_queue.pop_front();
            }

            int high_stream_id = 0;
            int low_stream_id = 0;

            if (concurrent_cnt == 1) {
                candidate_high_selector_1 =  candidate_best_selector_1;
                candidate_high_selector_2 = candidate_best_selector_2;
            }
            
             
            auto arrival_timestamp = chrono::steady_clock::now();
            for(int iter=0; iter < tmp_high_req.size(); iter++){
                auto cur_request = tmp_high_req[iter];
                if (high_stream_id >= default_stream_num)
                    high_stream_id = _gpu_stats.GetFillStreamId();             
                auto stream = _high_stream_pool[high_stream_id];
                if (cur_request->GetModelKind() == M1 && cur_request->GetQoSKind() != Empty) {  
                    high_results_model.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_high_selector_1), h_output_memory_pool_1[high_stream_id],\
                    h_input_memory_pool_1[high_stream_id], input_mem_size_1, stream, true));
                    cur_request->SetEstimatedLatency(model_1_latency);
                    _gpu_stats.Update(high_stream_id, cur_request);
                    
                    if (++high_stream_id >= default_stream_num)
                    {
                        high_stream_id = high_stream_id % default_stream_num;
                    }
                    model_1_number++;
                    //cout<<"request size: "<<_total_request_queue.size()<<endl;
                } else if (cur_request->GetModelKind() == M2 && cur_request->GetQoSKind() != Empty) {
                    high_results_model.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_high_selector_2), h_output_memory_pool_2[high_stream_id],\
                    h_input_memory_pool_2[high_stream_id], input_mem_size_2, stream, true));
                    cur_request->SetEstimatedLatency(model_2_latency);
                    _gpu_stats.Update(high_stream_id, cur_request);

                    if (++high_stream_id >= default_stream_num)
                    {
                        high_stream_id = high_stream_id % default_stream_num;
                    }
                    model_2_number++;
                }
            }

            int fill_low_qos_req_idx = 0;
            int sycn_high_request_idx = 0;
            int sycn_low_request_idx = 0;
            while (sycn_high_request_idx<high_qos_cnt || sycn_low_request_idx<fill_low_qos_req_idx) {
                int min_delay = -1;
                int stream_idx = _gpu_stats.GetNeedSyncStreamId();
                //cout<<"need stream idx: "<<stream_idx<<endl;
                //cout<<"sync high reqeuse idx: "<<sycn_high_request_idx<<" sync low request idx: "<<sycn_low_request_idx<<endl;
                for (int rid=0; rid< _gpu_stats._stream_mapping[stream_idx].size(); rid++) {
                    auto rs = _gpu_stats._stream_mapping[stream_idx][rid];
                    if(!rs->GetHasSync()) {
                        float ret_lat = 0.0;
                        if (rs->GetQoSKind() == HighQoS) {
                            //cout<<"sync high qos request "<<sycn_high_request_idx<<endl;
                            ret_lat = high_results_model[sycn_high_request_idx++].get();
                        }
                        else if (rs->GetQoSKind() == LowQoS) {
                            //cout<<"sync low qos request "<<sycn_low_request_idx<<endl;
                            ret_lat = low_results_model[sycn_low_request_idx++].get();
                        }
                        rs->SetSync();
                        auto finish_timestamp = chrono::steady_clock::now();
                        float duration = chrono::duration_cast<chrono::microseconds>(finish_timestamp - arrival_timestamp).count() * 1.0 / 1000;
                        float wait_time = duration - ret_lat;
                        if (wait_time < 0) {
                            cout<<"============= exception, wait time should greater than duration ============"<<endl;
                        }
                        if (rs->GetModelKind() == M1 && rs->GetQoSKind() == HighQoS) {
                            //cout<<"M1 and HighQos"<<endl;
                            model_1_lat_list_nowait_rt.push_back(ret_lat);
                            model_1_wait_list.push_back(wait_time);
                            model_1_lat_list_rt.push_back(duration);
                        } else if (rs->GetModelKind() == M2 && rs->GetQoSKind() == HighQoS) {
                            //cout<<"M2 and HighQos"<<endl;
                            model_2_lat_list_nowait_rt.push_back(ret_lat);
                            model_2_wait_list.push_back(wait_time);
                            model_2_lat_list_rt.push_back(duration);
                        } else if (rs->GetModelKind() == M1 && rs->GetQoSKind() == LowQoS) {
                            //cout<<"M1 and LowQos"<<endl;
                            model_1_wait_list.push_back(wait_time);
                            model_1_lat_list_be.push_back(duration);
                        } else if (rs->GetModelKind() == M2 && rs->GetQoSKind() == LowQoS) {
                            //cout<<"M2 and LowQos"<<endl;
                            model_2_wait_list.push_back(wait_time);
                            model_2_lat_list_be.push_back(duration);
                        } else {
                            cout<<"========= Invalid match ========"<<endl;
                        }
                        // fill a low  qos request
                        if(fill_low_qos_req_idx < low_qos_cnt) {
                            //cout<<"fill low qos request"<<endl;
                            low_stream_id = _gpu_stats.GetFillStreamId();
                            auto stream = _high_stream_pool[low_stream_id];
                            auto cur_request = tmp_low_req[fill_low_qos_req_idx++];
                            if (cur_request->GetModelKind() == M1 && cur_request->GetQoSKind() != Empty) {
                                //this->CandidateSliceScheduler(candidate_high_selector_1 ,candidate_best_selector_1, start_latency);  
                                low_results_model.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_best_selector_1), h_output_memory_pool_1[low_stream_id],\
                                h_input_memory_pool_1[low_stream_id], input_mem_size_1, stream, true));
                                cur_request->SetEstimatedLatency(model_1_latency);
                                _gpu_stats.Update(low_stream_id, cur_request);
                                model_1_number++;
                                //cout<<"request size: "<<_total_request_queue.size()<<endl;
                            } else if (cur_request->GetModelKind() == M2 && cur_request->GetQoSKind() != Empty) {
                                //this->CandidateSliceScheduler(candidate_high_selector_1 ,candidate_best_selector_2, start_latency); 
                                low_results_model.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_best_selector_2), h_output_memory_pool_2[low_stream_id],\
                                h_input_memory_pool_2[low_stream_id], input_mem_size_2, stream, true));
                                cur_request->SetEstimatedLatency(model_2_latency);
                                _gpu_stats.Update(low_stream_id, cur_request);
                                model_2_number++;
                            }
                        }
                        break;
                    }
                }
            }


            for(int iter=fill_low_qos_req_idx; iter < tmp_low_req.size(); iter++){
                auto cur_request = tmp_low_req[iter];             
                auto low_stream_id = _gpu_stats.GetFillStreamId();
                auto start_latency = _gpu_stats.GetStreamStartLatency();
                auto stream = _high_stream_pool[low_stream_id];
                if (cur_request->GetModelKind() == M1 && cur_request->GetQoSKind() != Empty) {
                    //this->CandidateSliceScheduler(candidate_high_selector_1 ,candidate_best_selector_1, start_latency);  
                    low_results_model.push_back(client_pool.enqueue(&Model::RealRun, _model_1, std::ref(candidate_best_selector_1), h_output_memory_pool_1[low_stream_id],\
                    h_input_memory_pool_1[low_stream_id], input_mem_size_1, stream, true));
                    cur_request->SetEstimatedLatency(model_1_latency);
                    _gpu_stats.Update(low_stream_id, cur_request);
                    model_1_number++;
                    //cout<<"request size: "<<_total_request_queue.size()<<endl;
                } else if (cur_request->GetModelKind() == M2 && cur_request->GetQoSKind() != Empty) {
                    //this->CandidateSliceScheduler(candidate_high_selector_1 ,candidate_best_selector_2, start_latency); 
                    low_results_model.push_back(client_pool.enqueue(&Model::RealRun, _model_2, std::ref(candidate_best_selector_2), h_output_memory_pool_2[low_stream_id],\
                    h_input_memory_pool_2[low_stream_id], input_mem_size_2, stream, true));
                    cur_request->SetEstimatedLatency(model_2_latency);
                    _gpu_stats.Update(low_stream_id, cur_request);
                    model_2_number++;
                }
            }

            while (sycn_low_request_idx<low_qos_cnt) {
                int min_delay = -1;
                int stream_idx = _gpu_stats.GetNeedSyncStreamId();
                for (int rid=0; rid< _gpu_stats._stream_mapping[stream_idx].size(); rid++) {
                    auto rs = _gpu_stats._stream_mapping[stream_idx][rid];
                    if(!rs->GetHasSync()) {
                        auto r = low_results_model[sycn_low_request_idx].get();
                        rs->SetSync();
                        auto finish_timestamp = chrono::steady_clock::now();
                        float duration = chrono::duration_cast<chrono::microseconds>(finish_timestamp - arrival_timestamp).count() * 1.0 / 1000;
                        float wait_time = duration - r;
                        if (wait_time < 0) {
                            cout<<"============= exception, wait time should greater than duration ============"<<endl;
                        }
                        if (rs->GetModelKind() == M1) {
                            model_1_lat_list_nowait_rt.push_back(r);
                            model_1_wait_list.push_back(wait_time);
                            model_1_lat_list_rt.push_back(duration);
                        } else if (rs->GetModelKind() == M2) {
                            model_2_lat_list_nowait_rt.push_back(r);
                            model_2_wait_list.push_back(wait_time);
                            model_2_lat_list_rt.push_back(duration);
                        }
                        sycn_low_request_idx++;
                        break;
                    }
                }
            }


            _gpu_stats.ClearStreamRuntime();

            auto end_ts = chrono::steady_clock::now();
            float total_dur = duration_cast<microseconds>(end_ts - begin_ts).count();
            float tp = (concurrent_cnt*1.0 / total_dur) * 1e+6 ;
            tp_list.push_back(tp);
            co_latency_list.push_back(total_dur*1.0/1000);
        }

        auto end_all_ts = chrono::steady_clock::now();
        float total_dur = duration_cast<microseconds>(end_all_ts - begin_all_ts).count();
        float avg_tp = (total_requests_cnt*1.0 / total_dur) * 1e+6;

        sort(tp_list.begin(), tp_list.end());
        sort(co_latency_list.begin(), co_latency_list.end());
        cout <<" Peak Throughput:" << tp_list[tp_list.size()/2] << "request/s" <<endl;
        cout <<" Avg Throughput:" << avg_tp << "request/s" <<endl;
        cout <<" Co lateny:" << co_latency_list[co_latency_list.size()/2] << "ms" <<endl;

        

        int warm_up = 0;
        int total_cnt_1 = model_1_lat_list_rt.size();
        int total_cnt_2 = model_2_lat_list_rt.size();
        
        float model_1_low_whisker = 0;
        float model_1_low_quartile = 0;
        float model_1_high_whisker = 0;
        float model_1_high_quartile = 0;
        float model_1_median = 0;

        float model_2_low_whisker = 0;
        float model_2_low_quartile = 0;
        float model_2_high_whisker = 0;
        float model_2_high_quartile = 0;
        float model_2_median = 0;
        
        sort(model_1_lat_list_rt.begin()+warm_up, model_1_lat_list_rt.end());
        sort(model_2_lat_list_rt.begin()+warm_up, model_2_lat_list_rt.end());

        sort(model_1_lat_list_nowait_rt.begin()+warm_up, model_1_lat_list_nowait_rt.end());
        sort(model_2_lat_list_nowait_rt.begin()+warm_up, model_2_lat_list_nowait_rt.end());

        model_1_low_whisker = model_1_lat_list_rt[warm_up+total_cnt_1*0.05];
        model_1_low_quartile = model_1_lat_list_rt[warm_up+total_cnt_1*0.25];
        model_1_high_whisker = model_1_lat_list_rt[warm_up+total_cnt_1*0.95];
        model_1_high_quartile = model_1_lat_list_rt[warm_up+total_cnt_1*0.75];
        model_1_median = model_1_lat_list_rt[warm_up+total_cnt_1*0.5];

        if (total_cnt_2 > 0) {
            model_2_low_whisker = model_2_lat_list_rt[warm_up+total_cnt_2*0.05];
            model_2_low_quartile = model_2_lat_list_rt[warm_up+total_cnt_2*0.25];
            model_2_high_whisker = model_2_lat_list_rt[warm_up+total_cnt_2*0.95];
            model_2_high_quartile = model_2_lat_list_rt[warm_up+total_cnt_2*0.75];
            model_2_median = model_2_lat_list_rt[warm_up+total_cnt_2*0.5];
        }

        cout<<"========== model 1 "<< _model_1->GetModelName() <<" box data: "<<model_1_low_whisker<<" "<<model_1_low_quartile<<" "\
        <<model_1_median<<" "<<model_1_high_quartile<<" "<<" "<<model_1_high_whisker<<endl;

        cout<<"========== model 2 "<< _model_2->GetModelName()<<" box data: "<<model_2_low_whisker<<" "<<model_2_low_quartile<<" "\
        <<model_2_median<<" "<<model_2_high_quartile<<" "<<" "<<model_2_high_whisker<<endl;

        model_1_low_whisker = model_1_lat_list_nowait_rt[warm_up+total_cnt_1*0.05];
        model_1_low_quartile = model_1_lat_list_nowait_rt[warm_up+total_cnt_1*0.25];
        model_1_high_whisker = model_1_lat_list_nowait_rt[warm_up+total_cnt_1*0.95];
        model_1_high_quartile = model_1_lat_list_nowait_rt[warm_up+total_cnt_1*0.75];
        model_1_median = model_1_lat_list_nowait_rt[warm_up+total_cnt_1*0.5];

        if (total_cnt_2 > 0) {
            model_2_low_whisker = model_2_lat_list_nowait_rt[warm_up+total_cnt_2*0.05];
            model_2_low_quartile = model_2_lat_list_nowait_rt[warm_up+total_cnt_2*0.25];
            model_2_high_whisker = model_2_lat_list_nowait_rt[warm_up+total_cnt_2*0.95];
            model_2_high_quartile = model_2_lat_list_nowait_rt[warm_up+total_cnt_2*0.75];
            model_2_median = model_2_lat_list_nowait_rt[warm_up+total_cnt_2*0.5];
        }

        cout<<"========== model 1 no wait "<< _model_1->GetModelName() <<" box data: "<<model_1_low_whisker<<" "<<model_1_low_quartile<<" "\
        <<model_1_median<<" "<<model_1_high_quartile<<" "<<" "<<model_1_high_whisker<<endl;

        cout<<"========== model 2 no wait "<< _model_2->GetModelName()<<" box data: "<<model_2_low_whisker<<" "<<model_2_low_quartile<<" "\
        <<model_2_median<<" "<<model_2_high_quartile<<" "<<" "<<model_2_high_whisker<<endl;
        
    }

 private:
    GPUStatus _gpu_stats;

};
