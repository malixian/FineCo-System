#ifndef PAIR_SERVICE_H_
#define PAIR_SERVICE_H_

#include <deque>
#include "models/model.h"
#include "common/request.hpp"
#include "service/possion.hpp"

class  PairService {
 public:
  virtual ~PairService() {}

  template<typename M1, typename M2>
  void Listen(const string& model_name_1, const string& model_name_2) {
    _model_name_1 = model_name_1;
    _model_name_2 = model_name_2;
    
    _model_1 = std::make_shared<M1>(model_name_1);
    _model_1->InitModel(ROCM);

    _model_2 = std::make_shared<M2>(model_name_2);
    _model_2->InitModel(ROCM);
  }    


  //virtual void Accept() {}
  
  void GeneratePoissonRequest(int total_request) {
    double high_arrival_rate = 2;
    double low_arrival_rate = 1.5;
    //double mid_arrival_rate = 

    Random high_random(1);
    Random low_random(3);

    vector<int> dist;
    int high_qos_cnt = 0;
    int low_qos_cnt = 0;

    int model_1_cnt = 0;
    int model_2_cnt = 0;

    for (int t=0; t<total_request; t++) {
      int t_cnt = 0;
      //cout<<"======== timestamp: "<<t<<" ======="<<endl;
      int h_request_num = high_random.poisson(high_arrival_rate);
      for (int i=0; i<h_request_num; i++) {
        int rn = rand()%4;
        if (rn % 2 == 0) {
          auto hr = make_shared<Request>(M1, LowQoS, t);
          _low_request_queue.push_back(hr);
          _total_request_queue.push_back(hr);
          model_1_cnt++;
        } else {
          auto hr = make_shared<Request>(M2, LowQoS, t);
          _low_request_queue.push_back(hr);
          _total_request_queue.push_back(hr);
          model_2_cnt++;
        } 
        //cout<<"low "<< "model: "<< rn%2<<" ";
      }
      
      t_cnt += h_request_num;
      high_qos_cnt += h_request_num;

      int l_request_num = low_random.poisson(low_arrival_rate);
      for (int i=0; i<l_request_num; i++) {
        int rn = rand()%4;
        if (rn % 2 == 0) {
          auto hr = make_shared<Request>(M1, HighQoS, t);
          _high_request_queue.push_back(hr);
          _total_request_queue.push_back(hr);
          model_1_cnt++;
        } else {
          auto hr = make_shared<Request>(M2, HighQoS, t);
          _high_request_queue.push_back(hr);
          _total_request_queue.push_back(hr);
          model_2_cnt++;
        } 
        //cout<<"high "<< "model: "<< rn%2<<" ";
      }
      t_cnt += l_request_num;
      low_qos_cnt += l_request_num;
      dist.push_back(t_cnt);
    }

    for(auto& t : dist) {
      cout<<t<<" ";
    }
    cout<<"\n";
    cout<<"low qos number: "<<high_qos_cnt<<" high qos number: "<<low_qos_cnt<<endl;
    cout<<"model 1 number: "<<model_1_cnt<<" model 2 number: "<<model_2_cnt<<endl;
    //for(auto& r : _high_request_queue) {
    //  cout<<" r qos: "<<r->GetQoSKind()<<" model: "<<r->GetModelKind()<<" timestap: "<<r->GetGeneTimestep()<<endl;
    //}
    //cout<<"\n";
  }


void GenerateRandomRequest(int request_num) {
  //srand(20);
  for (int num=0; num<request_num; num++) {
    int a = 1;
    int b = 2;
    // generate random number: [a, b)
    //srand((unsigned int)time(NULL));
    int tmp_co_request_num = (rand() % (b-a))+ a;
    cout<<tmp_co_request_num<<", ";
    if (tmp_co_request_num == 0) {
      auto r = make_shared<Request>(M1, Empty, num);
      _high_request_queue.push_back(r);
    }
    for (int i=0; i<tmp_co_request_num; i++) {
      if (rand() % 2 == 0) {
        auto r = make_shared<Request>(M1, HighQoS, num);
        _high_request_queue.push_back(r);
      } else {
        auto r = make_shared<Request>(M2, HighQoS, num);
        _high_request_queue.push_back(r);
      }
    }
  }
  cout<<"\n";
}



void GenerateFixedSeq(int seq_len) {
  for(int t=0; t<seq_len; t++) {
    
    auto r1 = make_shared<Request>(M1, HighQoS, t);
    _total_request_queue.push_back(r1);
    for(int i=0; i<2; i++) {
      auto r1 = make_shared<Request>(M1, LowQoS, t);
      auto r2 = make_shared<Request>(M2, LowQoS, t);
      _total_request_queue.push_back(r1);
      _total_request_queue.push_back(r2);
    }
  }
}

void GenerateTestSeq(int seq_len) {
  for(int t=0; t<seq_len; t++) {
    auto r1 = make_shared<Request>(M1, HighQoS, t);
    _total_request_queue.push_back(r1);
    auto r2 = make_shared<Request>(M2, HighQoS, t);
    _total_request_queue.push_back(r2);
    // auto r3 = make_shared<Request>(M2, HighQoS, t);
    // _total_request_queue.push_back(r3);
  }
}

void GenerateQoSSeq(int seq_len) {
  for(int t=0; t<seq_len; t++) {
    auto r1 = make_shared<Request>(M1, HighQoS, t);
    _total_request_queue.push_back(r1);
    auto r2 = make_shared<Request>(M2, LowQoS, t);
    _total_request_queue.push_back(r2);
    auto r4 = make_shared<Request>(M2, HighQoS, t);
    _total_request_queue.push_back(r4);
  }
}

void ServiceGenCoCombination(int co_num, const vector<int>& layer_count, vector<int>& random_comb) {
  for(int i=0; i<layer_count.size(); i++) {
      int cnt = layer_count[i];
      int sel = cnt / co_num;
      if (sel == 0) sel = 1;
      else if (sel >= cnt) sel = cnt - 1;
      // high qos response to high parallel candidate
      random_comb.push_back(sel);
  }
  
}


  void ServiceGenDefaultCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    for(int i=0; i<layer_count.size(); i++) {
        int cnt = layer_count[i];
        // high qos response to high parallel candidate
        random_comb.push_back(0);
    }
    
  }

  void ServiceGenHighQoSCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    for(int i=0; i<layer_count.size(); i++) {
        int cnt = layer_count[i];
        // high qos response to high parallel candidate
        random_comb.push_back(cnt-1);
    }
    
  }

  void ServiceGenLowQoSCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    for(int i=0; i<layer_count.size(); i++) {
        int cnt = layer_count[i];
        // low qos response to low parallel candidate
        if(cnt >= 2)
          random_comb.push_back(2);
        else
          random_comb.push_back(0);
    }
  }

  void ServiceGenRandomCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    for(int i=0; i<layer_count.size(); i++) {
        int cnt = layer_count[i];
        random_comb.push_back(rand() % cnt);
    }
  }

  void EqualAllocateCombination(int concurrent_num, vector<int>& cur_comb, vector<vector<int>>& candidate_size_list) {
    if(concurrent_num == 1) {
      return;
    }
    for(int layer_idx=0; layer_idx<cur_comb.size(); layer_idx++) {
      int cnt = candidate_size_list[layer_idx].size();
      cur_comb[layer_idx] = 0;
      for(int cid=0; cid<cnt; cid++) {
        //cout<<"block size: "<<candidate_size_list[layer_idx][cid]<<" concurrent num: "<<concurrent_num<<endl;
        if (cid > 0 && candidate_size_list[layer_idx][cid] * concurrent_num > 108) {
          cur_comb[layer_idx] = cid - 1;
          break;
        } 
      }
    }
  }

  void getCandidateImplByAvailabeResource(vector<pair<int, int>>& available_sm_seq, vector<int>& candidate_selector,\
   vector<vector<int>>& candidate_size_list) {
    int start_layer_idx = 0;
    int dnn_layer_len = candidate_selector.size();
    for(auto& item : available_sm_seq) {
      float ratio = (1.0 * item.first)/100;
      int available_sm = item.second;
      int end_layer_idx = dnn_layer_len * ratio;
      for(int layer_idx = start_layer_idx; layer_idx<=end_layer_idx; layer_idx++) {
        int cnt = candidate_size_list[layer_idx].size();
        candidate_selector[layer_idx] = 0;
        for(int cid=0; cid<cnt; cid++) {
          //cout<<"block size: "<<candidate_size_list[layer_idx][cid]<<" concurrent num: "<<concurrent_num<<endl;
          if (cid > 0 && candidate_size_list[layer_idx][cid]  > available_sm) {
            candidate_selector[layer_idx] = cid - 1;
            break;
          } 
        }
      }
      start_layer_idx = end_layer_idx + 1;
    }
  }

  // vgg19: 3, 5, 4, 4, 4, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2,
  // resnet50: 4, 1, 4, 4, 2, 4, 4, 2, 4, 4, 4, 2, 3, 2, 2, 3, 2, 2, 3, 2, 2, 3, 4, 2, 3, 2, 5, 3, 2, 5, 3, 2, 5, 3, 2, 5, 3, 2, 5, 3, 2, 1, 3, 1, 2, 3, 1, 2, 3,

  void ServiceGenCustomCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    // slo scale 1.8
    vector<int> vgg19{4, 6, 0, 4, 4, 0, 5, 6, 6, 6, 0, 3, 4, 4, 4, 0, 3, 3, 3, 3, 0, 0, 0, 0};
    //vector<int> vgg19{3, 4, 0, 4, 4, 0, 4, 4, 4, 4, 0, 4, 4, 6, 6, 0, 5, 5, 5, 5, 0, 0, 0, 0};
    vector<int> resnet50{5, 0, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 2, 4, 4, 2, 4, 4, 2, 4, 4, 4, 2, 3, 2, 5, 3, 2, 5, 5, 2, 5, 3, 2, 5, 5, 2, 5, 3, 3, 1, 3, 1, 2, 3, 1, 2, 3};
    cout<<layer_count.size()<<" "<<vgg19.size()<<" "<<resnet50.size()<<endl;
    
    if (layer_count.size() == vgg19.size()) {
      //cout<<"add vgg19"<<endl;
      for(int i=0; i<layer_count.size(); i++) {
        int candidate_idx = vgg19[i];
        random_comb.push_back(candidate_idx);
      }
    }
    

    if (layer_count.size() == 50) {
      cout<<"add resnet50"<<endl;
      for(int i=0; i<layer_count.size(); i++) {
        int candidate_idx = resnet50[i];
        random_comb.push_back(candidate_idx);
      }
    }
  }

  void GetBestEffectImplList(float qos_scale, const string sub_model_name, vector<int>& candidate_list, DLStream test_stream) {
      shared_ptr<Model> _model;
      if (sub_model_name == _model_name_1) {
        _model = _model_1;
      } else if (sub_model_name == _model_name_2) {
        _model = _model_2;
      } else {
        cout<<"None match model"<<endl;
        return;
      }

      vector<int> task_candidate_cnt;
      _model->GetAllLayerCandidateCnt(task_candidate_cnt);
      vector<vector<int>> candidate_size_list;
      _model->GetAllLayerCandidateSize(candidate_size_list);
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
              // cout<<"layer: "<<layer_idx<<" candidate: "<<candidate_idx<<endl;
              _model->LayerRunWarmup(layer_idx, candidate_idx, test_stream);
          }   
      }

      //cout<<"========= warm up finish ========" <<endl;

      //cout<<"SLO scale is: "<<qos_scale<<endl;

      for(int layer_idx=0; layer_idx<task_candidate_cnt.size();layer_idx++) {
          int best_effect_candidate_idx = -1;
          int best_effect_candidate_size = -1;
          
          float default_impl_latency = 0.0;
          
          int repeat_num = 10;
          int iter_num = 1;
          //cout<<"layer id:"<<layer_idx<<" candidate cnt: "<< task_candidate_cnt[layer_idx]<<endl;
          /*
          if (task_candidate_cnt[layer_idx] <= 1){
              continue;
          }
          */
          for(int candidate_idx=0; candidate_idx<task_candidate_cnt[layer_idx]; candidate_idx++){
              float task_latency = _model->LayerRun(layer_idx, candidate_idx, repeat_num, test_stream);
              best_effect_candidate_size = candidate_size_list[layer_idx][candidate_idx];
              float ret = task_latency / repeat_num * 1000;
              KernelAttr attr;
              attr.layer_idx = layer_idx;
              attr.latency = ret;
              attr.candidate_idx = candidate_idx;
              _model->SetLayerBlockLatency(layer_idx, best_effect_candidate_size, ret);
              if (candidate_idx == 0) {
                  default_impl_latency = ret;
                  //continue;
                  //cout<<"layer: "<<layer_idx<<" candidate 0 block: "<<candidate_size_list[layer_idx][candidate_idx]<<" lat: "<<ret<<endl;;
              } else {
                  if((ret <= (qos_scale * default_impl_latency)) && best_effect_candidate_idx == -1) {
                      best_effect_candidate_idx = candidate_idx;
                      ImplAttr impl_attr;
                      impl_attr.layer_idx = layer_idx;
                      impl_attr.size = best_effect_candidate_size;
                      impl_attr.candidate_idx = best_effect_candidate_idx;
                      impl_attr.latency = ret;
                      impl_attr.best_latency = default_impl_latency;
                      impl_list.emplace_back(impl_attr);
                      break;
                  }
              }
              results.emplace_back(attr);
          }
          if (best_effect_candidate_idx == -1) {
              ImplAttr impl_attr;
              impl_attr.layer_idx = layer_idx;
              impl_attr.size = 100;
              impl_attr.candidate_idx = 0;
              impl_attr.latency = default_impl_latency;
              impl_attr.best_latency = default_impl_latency;
              impl_list.emplace_back(impl_attr);
          }
      }

      for (auto& r : impl_list) {
        candidate_list.push_back(r.candidate_idx);
      }
       // print saved block-latency pair
      cout<<"================= "<<sub_model_name<<endl;
      for (auto& r : impl_list) {
        cout<<r.size<<", ";
      }
      cout<<"\n";

      // for (auto& r : impl_list) {
      //   //cout<<"layer idx: "<<r.layer_idx<<" candidate idx:"<<r.candidate_idx<<" size:"<<r.size<<" effective latency:"<<r.latency<<" best latency:"<<r.best_latency<<endl;
      //   if(r.size != 100)
      //       cout<<r.latency<<", ";
      //   else
      //       cout<<r.best_latency<<", ";
      // }
      // cout<<"\n";
      
  }

  void ServiceGenBestCombination(const string sub_model_name,  const vector<int>& layer_count, vector<int>& candidate_list) {
    shared_ptr<Model> tmp_model;
    if (sub_model_name == _model_name_1) {
        tmp_model = _model_1;
      } else if (sub_model_name == _model_name_2) {
        tmp_model = _model_2;
      } else {
        cout<<"None match model"<<endl;
        return;
    }

    for (int layer_idx=0; layer_idx<layer_count.size();layer_idx++) {
      //cout<<"layer: "<<layer_idx<<" candidate: "<<candidate_idx<<endl;
      candidate_list.push_back(tmp_model->GetBestCandidateIdx(layer_idx));
    }

  }

  void CandidateSchedulerV1(vector<int>& candidate_list_1, vector<int>& candidate_list_2) {
 
    

    int kernel_size_1 = candidate_list_1.size();
    int kernel_size_2 = candidate_list_2.size();

    int layer_idx_1 = 0, layer_idx_2 = 0;
    int timeline_1 = 0, timeline_2 = 0;
    int cur_sm_1 = 0, cur_sm_2 = 0;
    
    int candidate_idx_1=0, candidate_idx_2=0;
    int block_size_1=0, block_size_2 = 0;
    int before_layer_idx_1 = 0;
    int before_layer_idx_2 = 0;
    bool update_1=false, update_2 = false;

    // calculate resource fragment area
    float total_area_1 = 0, total_area_2 = 0;
    for(int layer_idx=0; layer_idx<candidate_list_1.size(); layer_idx++) {
      int candidate_idx = candidate_list_1[layer_idx];
      int block_cnt = _model_1->GetLayerBlock(layer_idx, candidate_idx);
      float latency = _model_1->GetLayerLatency(layer_idx, candidate_idx);
      int idle_sm = 108 - block_cnt;
      total_area_1 += idle_sm * latency;
    }

    for(int layer_idx=0; layer_idx<candidate_list_2.size(); layer_idx++) {
      int candidate_idx = candidate_list_2[layer_idx];
      int block_cnt = _model_2->GetLayerBlock(layer_idx, candidate_idx);
      float latency = _model_2->GetLayerLatency(layer_idx, candidate_idx);
      int idle_sm = 108 - block_cnt;
      total_area_2 += idle_sm * latency;
    }

    

    while(layer_idx_1<kernel_size_1 && layer_idx_2<kernel_size_2) {
      //cout<<"layer idx 1: "<< layer_idx_1<<" layer idx 2: "<< layer_idx_2<<endl;
      //cout<<"timeline 1: "<< timeline_1<<" timeline 2: "<< timeline_2<<endl;
      if (timeline_1 > timeline_2) {
        candidate_idx_2 = candidate_list_2[layer_idx_2];
        block_size_2 = _model_2->GetLayerBlock(layer_idx_2, candidate_idx_2);
        before_layer_idx_2 = layer_idx_2;
        layer_idx_2++;
        update_2 = true;
        update_1 = false;
      } else if (timeline_2 > timeline_1) {
        candidate_idx_1 = candidate_list_1[layer_idx_1];
        block_size_1 = _model_1->GetLayerBlock(layer_idx_1, candidate_idx_1);
        before_layer_idx_1 = layer_idx_1;
        layer_idx_1++;
        update_1 = true;
        update_2 = false;
      } else {
        candidate_idx_1 = candidate_list_1[layer_idx_1];
        block_size_1 = _model_1->GetLayerBlock(layer_idx_1, candidate_idx_1);
        before_layer_idx_1 = layer_idx_1;
        layer_idx_1++;
        update_1 = true;
        
        candidate_idx_2 = candidate_list_2[layer_idx_2];
        block_size_2 = _model_2->GetLayerBlock(layer_idx_2, candidate_idx_2);
        before_layer_idx_2 = layer_idx_2;
        layer_idx_2++;
        update_2 = true;
      }
      
      if (block_size_1 + block_size_2 <= 108) {
        if (update_1)
          timeline_1 += _model_1->GetLayerLatency(before_layer_idx_1, candidate_list_1[before_layer_idx_1]);
        if (update_2)
          timeline_2 += _model_2->GetLayerLatency(before_layer_idx_2, candidate_list_2[before_layer_idx_2]);
      }
      // when conflict, adjust used block number 
      // ** scheduling scheme 1: alternate adjustment, good for <resnet50, darknet>, bad for <resnet50, vgg19>
      else {
        if (timeline_1 <= timeline_2) {
          candidate_list_1[before_layer_idx_1] = _model_1->GetSatisfiedCandidateIdx(before_layer_idx_1, block_size_2);
          timeline_1 += _model_1->GetLayerLatency(before_layer_idx_1, candidate_list_1[before_layer_idx_1]);
          if (timeline_2 == 0) timeline_2 += _model_2->GetLayerLatency(before_layer_idx_2, candidate_idx_2);
        } else {
          candidate_list_2[before_layer_idx_2] = _model_2->GetSatisfiedCandidateIdx(before_layer_idx_2, block_size_1);
          timeline_2 += _model_2->GetLayerLatency(before_layer_idx_2, candidate_list_2[before_layer_idx_2]);
          if (timeline_1 == 0) timeline_1 += _model_1->GetLayerLatency(before_layer_idx_1, candidate_idx_1);
        }
      }
    }
    
    while(layer_idx_1<kernel_size_1) {
      candidate_list_1[layer_idx_1++] = 0;
    }

    while(layer_idx_2<kernel_size_2) {
      candidate_list_2[layer_idx_2++] = 0;
    }
    cout<<"======="<<_model_1->GetModelName()<<endl;
    for(auto& block : candidate_list_1) {
      cout<<block<<" ";
    }
    cout<<endl;

    cout<<"======="<<_model_2->GetModelName()<<endl;
    for(auto& block : candidate_list_2) {
      cout<<block<<" ";
    }
    cout<<endl;

  }


  void CandidateSchedulerV2(vector<int>& candidate_list_1, vector<int>& candidate_list_2) {

    int kernel_size_1 = candidate_list_1.size();
    int kernel_size_2 = candidate_list_2.size();

    int layer_idx_1 = 0, layer_idx_2 = 0;
    int timeline_1 = 0, timeline_2 = 0;
    int cur_sm_1 = 0, cur_sm_2 = 0;
    
    int candidate_idx_1=0, candidate_idx_2=0;
    int block_size_1=0, block_size_2 = 0;
    int before_layer_idx_1 = 0;
    int before_layer_idx_2 = 0;
    bool update_1=false, update_2 = false;

    // calculate resource fragment area
    float total_area_1 = 0, total_area_2 = 0;
    for(int layer_idx=0; layer_idx<candidate_list_1.size(); layer_idx++) {
      int candidate_idx = candidate_list_1[layer_idx];
      int block_cnt = _model_1->GetLayerBlock(layer_idx, candidate_idx);
      float latency = _model_1->GetLayerLatency(layer_idx, candidate_idx);
      int idle_sm = 108 - block_cnt;
      total_area_1 += idle_sm * latency;
    }

    for(int layer_idx=0; layer_idx<candidate_list_2.size(); layer_idx++) {
      int candidate_idx = candidate_list_2[layer_idx];
      int block_cnt = _model_2->GetLayerBlock(layer_idx, candidate_idx);
      float latency = _model_2->GetLayerLatency(layer_idx, candidate_idx);
      int idle_sm = 108 - block_cnt;
      total_area_2 += idle_sm * latency;
    }

    struct ConflictPair {
      int layer_idx_1;
      int layer_idx_2;
    };

    vector<ConflictPair> conflict_pair_list;
    vector<int> model_1_ava_seq;
    vector<int> model_2_ava_seq;

    // calculate conflict pair
    while(layer_idx_1<kernel_size_1 && layer_idx_2<kernel_size_2) {
      //cout<<"layer idx 1: "<< layer_idx_1<<" layer idx 2: "<< layer_idx_2<<endl;
      //cout<<"timeline 1: "<< timeline_1<<" timeline 2: "<< timeline_2<<endl;
      if (timeline_1 > timeline_2) {
        candidate_idx_2 = candidate_list_2[layer_idx_2];
        block_size_2 = _model_2->GetLayerBlock(layer_idx_2, candidate_idx_2);
        float latency = _model_2->GetLayerLatency(layer_idx_2, candidate_idx_2);
        int idle_block = 108 - block_size_2;
        total_area_2 -= idle_block * latency;
        before_layer_idx_2 = layer_idx_2;
        layer_idx_2++;
        update_2 = true;
        update_1 = false;
        model_2_ava_seq.push_back(total_area_2);
      } else if (timeline_2 > timeline_1) {
        candidate_idx_1 = candidate_list_1[layer_idx_1];
        block_size_1 = _model_1->GetLayerBlock(layer_idx_1, candidate_idx_1);
        float latency = _model_1->GetLayerLatency(layer_idx_1, candidate_idx_1);
        int idle_block = 108 - block_size_1;
        total_area_1 -= idle_block * latency;
        before_layer_idx_1 = layer_idx_1;
        layer_idx_1++;
        update_1 = true;
        update_2 = false;
        model_1_ava_seq.push_back(total_area_1); 
      } else {
        candidate_idx_1 = candidate_list_1[layer_idx_1];
        block_size_1 = _model_1->GetLayerBlock(layer_idx_1, candidate_idx_1);
        float latency = _model_1->GetLayerLatency(layer_idx_1, candidate_idx_1);
        int idle_block = 108 - block_size_1;
        total_area_1 -= idle_block * latency;
        before_layer_idx_1 = layer_idx_1;
        layer_idx_1++;
        update_1 = true;
        model_1_ava_seq.push_back(total_area_2); 
        
        candidate_idx_2 = candidate_list_2[layer_idx_2];
        block_size_2 = _model_2->GetLayerBlock(layer_idx_2, candidate_idx_2);
        latency = _model_2->GetLayerLatency(layer_idx_2, candidate_idx_2);
        idle_block = 108 - block_size_2;
        total_area_2 -= idle_block * latency;
        before_layer_idx_2 = layer_idx_2;
        layer_idx_2++;
        update_2 = true;
        model_2_ava_seq.push_back(total_area_2);
      }

      if (block_size_1 + block_size_2 <= 108) {
        if (update_1)
          timeline_1 += _model_1->GetLayerLatency(before_layer_idx_1, candidate_list_1[before_layer_idx_1]);
        if (update_2)
          timeline_2 += _model_2->GetLayerLatency(before_layer_idx_2, candidate_list_2[before_layer_idx_2]);
      } else {
        ConflictPair cp = {before_layer_idx_1, before_layer_idx_2};
        conflict_pair_list.push_back(cp);
      }
    }

    while(layer_idx_1<kernel_size_1) {
      candidate_list_1[layer_idx_1] = _model_1->GetBestCandidateIdx(layer_idx_1);
      //candidate_list_1[layer_idx_1] = 0;
      layer_idx_1++;
    }

    while(layer_idx_2<kernel_size_2) {
      candidate_list_2[layer_idx_2] = _model_2->GetBestCandidateIdx(layer_idx_2);
      //candidate_list_2[layer_idx_2] = 0;
      layer_idx_2++;
    }

    //reslove confilct
    for(auto& item : conflict_pair_list) {
      //cout<<item.layer_idx_1<<", "<<item.layer_idx_2<<"; avaliable sequence: "<<model_1_ava_seq[item.layer_idx_1]<<" "<<model_2_ava_seq[item.layer_idx_2]<<endl;
      if (candidate_list_1[item.layer_idx_1] + candidate_list_2[item.layer_idx_2] <= 108) continue;
      int ava_area_1 = model_1_ava_seq[item.layer_idx_1];
      int ava_area_2 = model_2_ava_seq[item.layer_idx_2];
      int sum_ava_area = ava_area_1 + ava_area_2;
      float ratio_1 = ava_area_1 / sum_ava_area;
      float ratio_2 = 1 - ratio_1;
      int limit_block_1 = 108 * (1 - ratio_1);
      int limit_block_2 = 108 * (1 - ratio_2);
      int new_cidx_1 = _model_1->GetLimitCandidateIdx(item.layer_idx_1, limit_block_1);
      int new_cidx_2 = _model_2->GetLimitCandidateIdx(item.layer_idx_2, limit_block_2);
      
      candidate_list_1[item.layer_idx_1] = new_cidx_1;
      candidate_list_2[item.layer_idx_2] = new_cidx_2;
    }
  }

  void CandidateSchedulerV3(vector<int>& candidate_list_1, vector<int>& candidate_list_2) {

    int kernel_size_1 = candidate_list_1.size();
    int kernel_size_2 = candidate_list_2.size();

    vector<vector<int>> candidate_size_list_1, candidate_size_list_2;
    _model_1->GetAllLayerCandidateSize(candidate_size_list_1);
    _model_2->GetAllLayerCandidateSize(candidate_size_list_2);

    // calculate resource fragment area
    float total_latency_1 = 0, total_latency_2 = 0;
    for(int layer_idx=0; layer_idx<candidate_list_1.size(); layer_idx++) {
      int candidate_idx = candidate_list_1[layer_idx];
      total_latency_1 += _model_1->GetLayerLatency(layer_idx, candidate_idx);
    }

    for(int layer_idx=0; layer_idx<candidate_list_2.size(); layer_idx++) {
      int candidate_idx = candidate_list_2[layer_idx];
      total_latency_2 += _model_2->GetLayerLatency(layer_idx, candidate_idx);
    }

    int layer_idx_1 = 0;
    int layer_idx_2 = 0;
    float timeline_1 = 0, timeline_2 = 0;

    // int before_layer_idx_1 = 0, before_layer_idx_2 = 0;
    int low_resource_idx = 3;
    bool update_1 = false, update_2 = false;
    int block_size_1 = 0, block_size_2 = 0;
    // calculate conflict pair
    while(layer_idx_1 < kernel_size_1 && layer_idx_2 < kernel_size_2) {
      int candidate_idx_1 = candidate_list_1[layer_idx_1];
      int candidate_idx_2 = candidate_list_2[layer_idx_2];
      if (timeline_1 > timeline_2) {
        block_size_2 = candidate_size_list_2[layer_idx_2][candidate_idx_2];

        if (total_latency_1 > total_latency_2) {
          if (candidate_size_list_2[layer_idx_2].size() >= low_resource_idx + 1) {
            candidate_idx_2 = min(max(candidate_idx_2, 1), low_resource_idx);
            block_size_2 = candidate_size_list_2[layer_idx_2][candidate_idx_2];
          }
        } else {
          while (block_size_1 + block_size_2 <= 120 && candidate_idx_2 < candidate_size_list_2[layer_idx_2].size() - 1) {
            candidate_idx_2++;
            block_size_2 = candidate_size_list_2[layer_idx_2][candidate_idx_2];
          }
        }

        update_2 = true;
        update_1 = false;
      } else if (timeline_2 > timeline_1) {
        block_size_1 = candidate_size_list_1[layer_idx_1][candidate_idx_1];

        if (total_latency_1 > total_latency_2) {
          while (block_size_1 + block_size_2 <= 120 && candidate_idx_1 < candidate_size_list_1[layer_idx_1].size() - 1) {
            candidate_idx_1++;
            block_size_1 = candidate_size_list_1[layer_idx_1][candidate_idx_1];
          }
        } else {
          if (candidate_size_list_1[layer_idx_1].size() >= low_resource_idx + 1) {
            candidate_idx_1 = min(max(candidate_idx_1, 1), low_resource_idx);
            block_size_1 = candidate_size_list_1[layer_idx_1][candidate_idx_1];
          }
        }

        update_1 = true;
        update_2 = false;
      } else {
        block_size_1 = candidate_size_list_1[layer_idx_1][candidate_idx_1];
        block_size_2 = candidate_size_list_2[layer_idx_2][candidate_idx_2];

        if (total_latency_1 > total_latency_2) {
          if (candidate_size_list_2[layer_idx_2].size() >= low_resource_idx + 1) {
            candidate_idx_2 = min(max(candidate_idx_2, 1), low_resource_idx);
            block_size_2 = candidate_size_list_2[layer_idx_2][candidate_idx_2];
          }
          while (block_size_1 + block_size_2 <= 120 && candidate_idx_1 < candidate_size_list_1[layer_idx_1].size() - 1) {
            candidate_idx_1++;
            block_size_1 = candidate_size_list_1[layer_idx_1][candidate_idx_1];
          }
        } else {
          if (candidate_size_list_1[layer_idx_1].size() >= low_resource_idx + 1) {
            candidate_idx_1 = min(max(candidate_idx_1, 1), low_resource_idx);
            block_size_1 = candidate_size_list_1[layer_idx_1][candidate_idx_1];
          }
          while (block_size_1 + block_size_2 <= 120 && candidate_idx_2 < candidate_size_list_2[layer_idx_2].size() - 1) {
            candidate_idx_2++;
            block_size_2 = candidate_size_list_2[layer_idx_2][candidate_idx_2];
          }
        }

        update_1 = true;
        update_2 = true;
      }

      if (update_1) {
        timeline_1 += _model_1->GetLayerLatency(layer_idx_1, candidate_list_1[layer_idx_1]);
        candidate_list_1[layer_idx_1] = candidate_idx_1;
        layer_idx_1++;
      }
      if (update_2) {
        timeline_2 += _model_2->GetLayerLatency(layer_idx_2, candidate_list_2[layer_idx_2]);
        candidate_list_2[layer_idx_2] = candidate_idx_2;
        layer_idx_2++;
      }
    }
    cout << "end co scheduling" << endl;

    while(layer_idx_1<kernel_size_1) {
      candidate_list_1[layer_idx_1] = _model_1->GetBestCandidateIdx(layer_idx_1);
      //candidate_list_1[layer_idx_1] = 0;
      layer_idx_1++;
    }

    cout << "================= " << _model_name_1 << endl;
    for (int layer_idx = 0; layer_idx < kernel_size_1; layer_idx++) {
      int block_size = candidate_size_list_1[layer_idx][candidate_list_1[layer_idx]];
      cout << block_size << ", ";
    }
    cout << endl;

    while(layer_idx_2<kernel_size_2) {
      candidate_list_2[layer_idx_2] = _model_2->GetBestCandidateIdx(layer_idx_2);
      //candidate_list_2[layer_idx_2] = 0;
      layer_idx_2++;
    }

    cout << "================= " << _model_name_2 << endl;
    for (int layer_idx = 0; layer_idx < kernel_size_2; layer_idx++) {
      int block_size = candidate_size_list_2[layer_idx][candidate_list_2[layer_idx]];
      cout << block_size << ", ";
    }
    cout << endl;
  }


  


  void CandidateSliceScheduler(vector<int>& exec_candidate_list, vector<int>& sch_candidate_list, float delay) {

    int kernel_size_1 = exec_candidate_list.size();
    int kernel_size_2 = sch_candidate_list.size();

    float tmp_exe_latency = 0;
    int tmp_layer_idx = -1;
    for(int layer_idx=0; layer_idx<exec_candidate_list.size(); layer_idx++) {
      int candidate_idx = exec_candidate_list[layer_idx];
      int block_cnt = _model_1->GetLayerBlock(layer_idx, candidate_idx);
      float latency = _model_1->GetLayerLatency(layer_idx, candidate_idx);
      if (tmp_exe_latency > delay) {
        tmp_layer_idx = layer_idx;
        break;
      }
      tmp_exe_latency += latency;
    }

    int layer_idx_1 = tmp_layer_idx;
    int layer_idx_2 = 0;

    while(layer_idx_1<kernel_size_1 && layer_idx_2<kernel_size_2) {
      int candidate_idx_1 = exec_candidate_list[layer_idx_1];
      int block_size_1 = _model_1->GetLayerBlock(layer_idx_1, candidate_idx_1);
      int idle_block = 108 - block_size_1;
      int ret_candidate_idx = _model_2->GetLimitCandidateIdx(layer_idx_2, idle_block);
      //cout<<"confilct idx: "<<ret_candidate_idx<<endl;
      sch_candidate_list[layer_idx_2] = ret_candidate_idx;
      layer_idx_1++;
      layer_idx_2++;
    }
  }


 protected:
  shared_ptr<Model> _model_1;
  shared_ptr<Model> _model_2;
  string _model_name_1;
  string _model_name_2;
  // real time QoS request
  deque<shared_ptr<Request>> _high_request_queue;
  // best effort QoS request
  deque<shared_ptr<Request>> _low_request_queue;
  // contain all RT and BE requests
  deque<shared_ptr<Request>> _total_request_queue;

};

#endif