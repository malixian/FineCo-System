#include "models/model.h"
#include "kernel/layer.h"
#include "../backend/rocm/rocm_timer.cc"
#include "kernel/batch_matmul.h"
#include "kernel/add.h"
#include "kernel/concat.h"
#include <iostream>
#include <unistd.h>
#include <cstdlib>


class Bert final : public Model {
 public:
    
    Bert(const string& model_name = "Bert") : Model(model_name) {}


    void AddBatchMat(int B, int N, int M, int K, bool need_input1, bool need_input2, BERTPhase phase_name) {
        BatchMatmul* BM = new BatchMatmul(B, N, M, K);
        BM->InitParams(need_input1, need_input2);
        BM->SetPhaseName(phase_name);
        string kernel_name = "";
        if(phase_name == CQK) {
            kernel_name = "compute_qk";
        } else if (phase_name == CSV) {
            kernel_name = "compute_sv";
        } else if (phase_name == EnGenQ || phase_name == EnGenK || phase_name == EnGenV || \
            phase_name == DeGenK || phase_name == DeGenQ || phase_name == DeGenV) {
            kernel_name = "gene_qkv";
        } else if (phase_name == EnFF1 || phase_name == DeFF1) {
            kernel_name = "feed_forward_1";
        } else if (phase_name == EnFF2 || phase_name == DeFF2) {
            kernel_name = "feed_forward_2";
        }
        ConfigureLayerCandidateByName(BM, _model_name, kernel_name);
    }

    /*
    void AddConvCat(int B, int N, int M, int K) {
        Concat* concat = new Concat(B, N, M, K);
        concat->InitParams();
        ConfigureLayerCandidate(concat);
    }
    */

    /*
    void AddTensorAdd(int B, int N, int M, int K) {
        TensorAdd* add = new TensorAdd(B, N, M, K);
        add->InitParams(false);
        BM->SetPhaseName();
        ConfigureLayerCandidateByName(add, _model_name, "add");
    }*/
    
    void GenQKV(BERTPhase phase_name1, BERTPhase phase_name2, BERTPhase phase_name3) {
        // gene input, WQ, and Q
        AddBatchMat(_batch_size, _seq_len, _head_dim, _embed_size, true, true, phase_name1);

        // gene K
        AddBatchMat(_batch_size, _seq_len, _head_dim, _embed_size, false, true, phase_name2);

        // gene V
        AddBatchMat(_batch_size, _seq_len, _head_dim, _embed_size, false, true, phase_name3);

        
    }


    // HeadNum = 12 for small bert
    /*
    void GeneMultiHead(int Batch, int SeqLen, int Embed, int HeadNum) {
        int HeadDim = Embed / HeadNum;
        for(int i=0; i<HeadNum; i++) {
            AddBatchMat(Batch, SeqLen, HeadDim, Embed, false, true, GeneMultiHead);
        }
    }
    */


    void ComputeQK() {
        AddBatchMat(_batch_size, _seq_len, _seq_len, _head_dim, false, false, CQK);
    }

    void ComputeSV( ) {
        AddBatchMat(_batch_size, _seq_len, _embed_size, _seq_len, false, false, CSV);
        
    }

    void AddFeedForward() {
        AddBatchMat(_batch_size, _seq_len, _feedforward_size, _embed_size, false, true, DeFF1);
        AddBatchMat(_batch_size, _seq_len, _embed_size, _feedforward_size, false, true, DeFF2);
    }


    void AddMultiHead() {
        ComputeQK();
        ComputeSV();
    }

    void AddEncodedr() {
        for(int i=0; i<_head_num; i++){
            AddMultiHead();
            //AddTensorAdd(EnAdd1);
            AddFeedForward();
            //AddTensorAdd(EnAdd2);
        }
    }

    void AddDecodedr() {
        for(int i=0; i<_head_num; i++){
            AddMultiHead();
            //AddTensorAdd(DeAdd1);
            AddMultiHead();
        }
            
        //AddTensorAdd(DeAdd2);
        AddFeedForward();
        //AddTensorAdd(DeAdd3);
    }



    
    void InitModel(RUNTIME runtime) final {
        /*====== AlexNet Layer 1 =======*/
        // Alloc Memory
        _runtime = runtime;
        if (_runtime == ROCM)
            SetRightSize(72);

        GenQKV(EnGenQ, EnGenK, EnGenV);
        for (int i=0; i<_head_num; i++) {
            AddEncodedr();
        }

        /*
        GenQKV(DeGenQ, DeGenK, DeGenV);
        for (int i=0; i<BertSize; i++) {
           AddDecodedr();
       }
       */

       int output_size =  _batch_size * _seq_len * _embed_size;
       SetOutPutSize(output_size);
        
        // cout<<"Bert Model Init Successfully"<<endl;

    }


    float Run(vector<int>& selector, vector<int>& requests, const string& task_name, int stream_num, vector<int>& random_list, vector<float>& ret_list, bool print_latency) {
        float avg_latency = 0.0;

        for(auto r : requests) {
            AddRequest(r);
        }

        int request_size = GetRequestSize();
        //cout<<"Total Request Size: "<<request_size<<endl;

        // Default Max stream number is 5
        int streams = stream_num;

        
        for (int i=0; i<streams; i++) {
            auto device_api =  GetBackendHandle(_runtime);
            auto stream = device_api->CreateStream(0);
            _run_stream_list.push_back(stream);
        }

        ROCMTimer timer(_run_stream_list[0]);
        timer.Start();

        // Sleep sleep;
        // sleep.Load();

        bool has_sleep = false;
        int request_global_id = 0;
        float total_sleep_us = 1.0;

        while(!RequestIsEmpty()) { 
            int iter_stream_num = 0;
            for(int i=0; i<streams; i++) {
                if (request_global_id >= request_size) break;
                iter_stream_num++;
                auto stream = _run_stream_list[i];
                
                // add sleep to improve randomness of kernel overlap
                int random_us = random_list[request_global_id++];
                total_sleep_us += float(random_us);
                // sleep.Compute(stream, random_us);
                
                for(int layer_idx=0; layer_idx<selector.size(); layer_idx++) {
                    auto candidate_idx = selector[layer_idx];
                    auto layer = _super_model[layer_idx];
                    auto candidate_name = layer->GetCandidateNameById(candidate_idx);
                    auto layer_kind = layer->GetLayerKind();
                    float* intermediate_data;

                    bool is_first_engenq = true;
                    bool is_first_degenq = true;
                    
                    if (layer_kind == BATCH_MATMUL) {
                        try {
                            BatchMatmul* bm = dynamic_cast<BatchMatmul*>(layer);
                            //if(layer_idx > 0) bm->SetInput(intermediate_data);
                            if (bm->GetPhaseName() == EnGenQ) {
                                _encoder_input = bm->GetInput1();                                
                                bm->Compute(candidate_name, stream);
                                _q = bm->GetOutPut();
                                //cout<<"EnGenQ, _q:"<<_q<<" _encoder_input:"<<_encoder_input<<endl;
                            } else if (bm->GetPhaseName() == DeGenQ) {
                                _decoder_input = bm->GetInput1();                                
                                bm->Compute(candidate_name, stream);
                                _q = bm->GetOutPut();
                                //cout<<"DeGenQ, _q:"<<_q<<" input:"<<_decoder_input<< endl;
                            } else if (bm->GetPhaseName() == EnGenK || bm->GetPhaseName() == DeGenK) {
                                if (bm->GetPhaseName() == EnGenK)
                                    bm->SetInput1(_encoder_input);
                                if (bm->GetPhaseName() == DeGenK)
                                    bm->SetInput1(_decoder_input); 
                                bm->Compute(candidate_name, stream);
                                _k = bm->GetOutPut();
                                //cout<<"Gen K, _k:"<<_k<<endl;
                            } else if (bm->GetPhaseName() == EnGenV || bm->GetPhaseName() == DeGenV) {
                                if (bm->GetPhaseName() == EnGenV)
                                    bm->SetInput1(_encoder_input);
                                if (bm->GetPhaseName() == DeGenV)
                                    bm->SetInput1(_decoder_input); 
                                bm->Compute(candidate_name, stream);
                                _v = bm->GetOutPut();
                                //cout<<"Gen V, _v:"<<_v<<endl;
                            } else if (bm->GetPhaseName() == CQK) {
                                bm->SetInput1(_q);
                                bm->SetInput2(_k);
                                bm->Compute(candidate_name, stream);
                                //cout<<"CQK, _q:"<<_q<<" CQK, _k:"<<_k<<endl;
                                _s = bm->GetOutPut();
                            } else if (bm->GetPhaseName() == CSV) {
                                bm->SetInput1(_s);
                                bm->SetInput2(_v);
                                bm->Compute(candidate_name, stream);
                                _intermediate_data = bm->GetOutPut();
                            } else if (bm->GetPhaseName() == EnFF1) {
                                bm->SetInput1(_intermediate_data);
                                bm->Compute(candidate_name, stream);
                                _intermediate_data = bm->GetOutPut();
                            } else if (bm->GetPhaseName() == EnFF2) {
                                bm->SetInput1(_intermediate_data);
                                bm->Compute(candidate_name, stream);
                                _encoder_input = bm->GetOutPut();
                            } else if (bm->GetPhaseName() == DeFF1) {
                                bm->SetInput1(_intermediate_data);
                                bm->Compute(candidate_name, stream);
                                _intermediate_data = bm->GetOutPut();
                            } else if (bm->GetPhaseName() == DeFF2) {
                                bm->SetInput1(_intermediate_data);
                                bm->Compute(candidate_name, stream);
                                _decoder_input = bm->GetOutPut();
                            }
                            
                        } catch(std::bad_cast const& ex) {
                            cout << "[ Conv2D"<<ex.what()<<"]" << endl;
                        }    
                    } 
                } 
            }

            FinishRequest(iter_stream_num);
        }

        for(int i=0; i<streams; i++) {
            GetBackendHandle(_runtime)->StreamSync(_run_stream_list[i]);
        }

        timer.Stop();
        auto latency = timer.SyncAndGetElapsedms();


        float throughput = (request_size / (latency - total_sleep_us/1000)) * 1000; 
        ret_list.push_back(throughput);
    
        avg_latency = (latency - total_sleep_us/1000)  / (request_size / stream_num);
        if(print_latency) {
            // cout<<task_name + " inference avg latency: "<<avg_latency<<"ms"<<endl;
            // cout << avg_latency << "ms " << throughput << endl;
            cout << avg_latency << endl;
            // cout << throughput << endl;
        }

        return avg_latency;
            
  }





 private:
  int _batch_size = 1;
  int _seq_len = 128;
  int _embed_size = 768;
  int _feedforward_size = 3072;
  int _head_dim = 64;
  int _head_num = 12;


  bool _gene_encoder_input = false;
  bool _gene_decoder_input = false;

  float* _q;
  float* _k;
  float* _v;
  float* _s;
  float* _encoder_input;
  float* _decoder_input;
  float* _intermediate_data;


};