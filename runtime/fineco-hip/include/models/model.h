#ifndef MODEL_H
#define MODEL_H

#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <cassert>
#include <fstream>
#include <iostream>
#include "kernel/layer.h"
#include "kernel/sleep.h"
#include "../src/backend/runtime.cpp"
#include "../src/kernels/conv2d_relu.cpp"
#include "../src/kernels/dense.cpp"
#include "../src/kernels/maxpool.cpp"
#include "../src/kernels/depthwise_conv.cpp"
#include "../src/backend/rocm/rocm_timer.cc"

class  Model {
 public:
  Model(const string& model_name) : _model_name(model_name) {}
  // Model(const string& model_name, const int right_size) : _model_name(model_name), _right_size(right_size) {}

  virtual ~Model() {
    // cout<<"model delete"<<endl;
    for(auto layer : _super_model) {
      delete layer;
    }
  }

  virtual void InitModel(RUNTIME runtime){};

  //virtual void Run(vector<size_t> selector){};

  string GetCandidatePath(const string& model_name, \
            int layer_idx, RUNTIME runtime = ROCM) {
    if(runtime != ROCM) {
      return "";
    }
    string runtime_str = "rocm";
    string base_path = "/home/chr/repos/fineco-hip/include/kernel/" + runtime_str + "/";
    string candidate_params_path = base_path + model_name + "/" + "layer" \
    + to_string(layer_idx) + "/" + "candidate_params.txt";
    return candidate_params_path;
  }

  int GetCandidateCount(const string& model_name, int layer_idx, RUNTIME runtime = ROCM) {
    string file = GetCandidatePath(model_name, layer_idx, runtime);
    // cout<<"candidiate path: "<<file<<endl;
    ifstream infile; 
    infile.open(file.data());   
    assert(infile.is_open());   

    int line = 0;
    string s;
    while(getline(infile,s))
    { 
        line++;
    }
    infile.close(); 
    return line - 1;     
  }

  int GetCandidateCountByPath(const string& file) {
    ifstream infile; 
    //cout<<"file path:"<<file<<endl;
    infile.open(file.data());   
    assert(infile.is_open());   

    int line = 0;
    string s;
    while(getline(infile,s))
    { 
        line++;
    }
    infile.close(); 
    return line - 1;     
  }


  void ReadCandidateParams(vector<vector<int>>&BT, const string& file) {
    ifstream infile; 
    infile.open(file.data());   
    assert(infile.is_open());   

    string s;
    int idx = 0;
    int line = 0;
    while(getline(infile,s))
    { 
        if(line == 0) {
          line++;
          continue;
        }

        string delimiter = " ";
        int deli_idx = s.find(delimiter);
        string block_str = s.substr(0, deli_idx);
        string thread_str = s.substr(deli_idx+1, s.length()-deli_idx-1);
        BT[idx][0] = stoi(block_str);
        BT[idx][1] = stoi(thread_str);
        //cout<<"read block: "<<BT[idx][0]<<" read thread: "<<BT[idx][1]<<endl;
        idx++;
    }
    infile.close();      
}


  void FindConfigureCandidate(vector<vector<int>>& BT, const string& model_name, \
            int layer_idx, RUNTIME runtime = ROCM) {
    string candidate_params_path = GetCandidatePath(model_name, layer_idx, runtime);
    
    if (!exists_file(candidate_params_path)) {
      cout<< "candidate params file not exist"<<" "<<candidate_params_path<<endl;
      return;
    }
    ReadCandidateParams(BT, candidate_params_path);
  }

  void FindConfigureCandidateByPath(vector<vector<int>>& BT, const string& path) {
    if (!exists_file(path)) {
      cout<< "candidate params file not exist"<<" "<<path<<endl;
      return;
    }
    ReadCandidateParams(BT, path);
  }

  void ConfigureLayerCandidate(Layer* layer) {
    int candidate_cnt = GetCandidateCount(_model_name, _kernel_idx, _runtime);
    vector<vector<int>> block_threads(candidate_cnt, vector<int>(2,0));
    FindConfigureCandidate( block_threads, _model_name, _kernel_idx, _runtime);

    // Register Candidate
    for (int candidate_idx=0; candidate_idx < block_threads.size(); candidate_idx++) {
        string kernel_name = "candidate" + to_string(candidate_idx);
        int block = block_threads[candidate_idx][0];
        int thread = block_threads[candidate_idx][1];
        string kenrel_idx_str = "layer" + to_string(_kernel_idx);
        layer->RegisterCandidate(_model_name, kenrel_idx_str, kernel_name, block, thread);
    }

    _kernel_idx++;
    _super_model.push_back(layer);
  }

  void ConfigureLayerCandidateByPath(Layer* layer, const string& sub_model_name,  const string& layer_name) {
        
    string base_path = "/home/chr/repos/fineco-hip/include/kernel/rocm/";
    // if (_runtime == CUDA) {
    //     base_path += "cuda/";
    // }
    
    base_path += sub_model_name + "/";
    string layer_idx_str = "";
    if (layer_name != "maxpool" && layer_name != "dense") {
      layer_idx_str =  "layer" + to_string(_layer_idx);
      base_path += layer_idx_str + "/";
    }
    //cout<<"layer :"<<layer_name<<" base_path: "<<base_path<<endl;
    base_path += layer_name + "/candidate_params.txt";
    int candidate_cnt = GetCandidateCountByPath(base_path);
    vector<vector<int>> block_threads(candidate_cnt, vector<int>(2, 0));
    
    FindConfigureCandidateByPath(block_threads, base_path);
    // Register Candidate
    for (int candidate_idx = 0; candidate_idx < block_threads.size(); candidate_idx++) {
        string kernel_name = "candidate" + to_string(candidate_idx);
        int block = block_threads[candidate_idx][0];
        int thread = block_threads[candidate_idx][1];
        layer->RegisterCandidate(sub_model_name, layer_idx_str, kernel_name, block, thread, layer_name);
    }
    _kernel_idx++;
    _super_model.push_back(layer);
  }

  void ConfigureLayerCandidateByName(Layer* layer, const string& sub_model_name,  const string& layer_name) {
        
    string base_path = "/home/chr/repos/fineco-hip/include/kernel/rocm/";
    // if (_runtime == CUDA) {
    //     base_path += "cuda/";
    // }
    
    base_path += sub_model_name + "/";
    base_path += layer_name + "/candidate_params.txt";
    int candidate_cnt = GetCandidateCountByPath(base_path);
    vector<vector<int>> block_threads(candidate_cnt, vector<int>(2,0));
    
    FindConfigureCandidateByPath(block_threads, base_path);
    // Register Candidate
    for (int candidate_idx=0; candidate_idx < block_threads.size(); candidate_idx++) {
        string kernel_name = "candidate" + to_string(candidate_idx);
        int block = block_threads[candidate_idx][0];
        int thread = block_threads[candidate_idx][1];
        layer->RegisterCandidate(sub_model_name, layer_name, kernel_name, block, thread);
    }
    _kernel_idx++;
    _super_model.push_back(layer);
  }

  void GetAllLayerCandidateCnt(vector<int>& ret) {
    for (auto layer : _super_model) {
      int layer_cnt = layer->GetLayerCandidateCnt();
      ret.push_back(layer_cnt);
    }
  }

  void GetAllLayerCandidateSize(vector<vector<int>>& ret) {
    for (auto layer : _super_model) {
      vector<int> candidate_list;
      layer->GetLayerCandidateSizeList(candidate_list);
      ret.push_back(candidate_list);
    }
  }

  void SetLayerBlockLatency(int layer_idx, int block, float latency) {
    _super_model[layer_idx]->AddBlockLatency(block, latency);
  }

  virtual float RealRun(vector<int>& selector, float* output, float* input, size_t input_size, DLStream stream, bool return_latency=false) {
    auto device_api =  GetBackendHandle(_runtime);
    if (stream == nullptr) {
      cout<<"======= stream is null ========"<<endl;
      stream = device_api->CreateStream(0);
    }

    shared_ptr<ROCMTimer> timer;
    if(return_latency) {
      timer = make_shared<ROCMTimer>(stream);
      timer->Start();
    }
    
    float* intermediate_data;

    //Sleep sleep;
    //sleep.Load();
    
    //for(int iter=0; iter<10; iter++){
    for(int layer_idx=0; layer_idx<_super_model.size(); layer_idx++) {
      auto candidate_idx = selector[layer_idx];
      auto layer = _super_model[layer_idx];
      auto candidate_name = layer->GetCandidateNameById(candidate_idx);
      auto layer_kind = layer->GetLayerKind();
      //if(sleep_us > 0 && (layer_idx == 19 || layer_idx == 21 || layer_idx == 22) ) {
      //  sleep.Compute(stream, sleep_us);
      //}
      if (layer_kind == CONV) {
          try {
              
              Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);
              // copy user input as first layer input
              if (layer_idx == 0) {
                conv2d->CopyAndSetInput(input, stream);
              } else conv2d->SetInput(intermediate_data);
              //if (layer_idx != 0 ) conv2d->SetInput(intermediate_data);
              conv2d->Compute(candidate_name, stream);
              intermediate_data = conv2d->GetOutPut();
              //cout<<layer_idx<<" conv "<<intermediate_data<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
              //cout<<"Layer Idx: "<<layer_idx<<" Computation: "<<conv2d->GetComputation()<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Conv2D"<<ex.what()<<"]" << endl;
          }    
      } else if (layer_kind == CONV_RELU) {
          try {
              //cout<<"layer:"<<layer<<endl;
              Conv2DRelu* conv2d = dynamic_cast<Conv2DRelu*>(layer);
              if (layer_idx == 0) {
                conv2d->CopyAndSetInput(input, stream);
              } else conv2d->SetInput(intermediate_data);
              if (layer_idx != 0 ) conv2d->SetInput(intermediate_data);
              conv2d->Compute(candidate_name, stream);
              intermediate_data = conv2d->GetOutPut();
	      // cout<<layer_idx<<"output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
              //cout<<layer_idx<<"conv2dRelu"<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Conv2D Relu"<<ex.what()<<"]" << endl;
          }    
      } else if (layer_kind == MAXPOOL) {
          try {
              MaxPool* maxpool = dynamic_cast<MaxPool*>(layer);
              
              if(layer_idx > 0) maxpool->SetInput(intermediate_data);
              maxpool->Compute(candidate_name, stream);
              intermediate_data = maxpool->GetOutPut();
              //cout<<layer_idx<<" maxpool "<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Maxpool"<<ex.what()<<"]" << endl;
          } 
      } else if (layer_kind == DENSE) {
          try {
              Dense* dense = dynamic_cast<Dense*>(layer);
              if(layer_idx > 0) dense->SetInput(intermediate_data);
              dense->Compute(candidate_name, stream);
              intermediate_data = dense->GetOutPut();
              //cout<<layer_idx<<" dense "<<intermediate_data<<" output: "<<dense->GetOutputTensorDim()<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "Dense "<<ex.what()<<"]" << endl;
          } 
      } else if (layer_kind == DEPTHWISECONV) {
          try {
              DepthwiseConv* conv = dynamic_cast<DepthwiseConv*>(layer);
              if(layer_idx > 0) conv->SetInput(intermediate_data);
              conv->Compute(candidate_name, stream);
              intermediate_data = conv->GetOutPut();
              //cout<<layer_idx<<"depthwise"<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "DepthwiseConv "<<ex.what()<<"]" << endl;
          }
      }
    }
    //}
    float latency = 0.0;;
    if(return_latency) {
      timer->Stop();
      latency = timer->SyncAndGetElapsedms();
    }
      
    GetBackendHandle(_runtime)->CopyDataFromToAsync(intermediate_data, output, _output_size, D2H, stream);
    //cout<<" inference latency: "<<latency<<"ms"<<" throughput: "<<throughput<<endl;
    //cout<<"finish"<<endl;
    return latency;
  }


  virtual float Run(vector<int>& selector, vector<int>& requests, const string& task_name, int stream_num, vector<int>& random_list, vector<float>& ret_list, bool print_latency) {
    float avg_latency = 0.0;

    for(auto r : requests) {
        AddRequest(r);
    }

    int request_size = GetRequestSize();
    
    auto stream = GetBackendHandle(_runtime)->CreateStream(0);

    ROCMTimer timer(stream);
    timer.Start();

    //Sleep sleep;
    //sleep.Load();

    int request_global_id = 0;
    float total_sleep_us = 1.0;

    while(!RequestIsEmpty()) { 
      float* input;
      for(int layer_idx=0; layer_idx<_super_model.size(); layer_idx++) {
        if (request_global_id >= request_size) break;
  
        float* intermediate_data;
        auto candidate_idx = selector[layer_idx];
        auto layer = _super_model[layer_idx];
        auto candidate_name = layer->GetCandidateNameById(candidate_idx);
        auto layer_kind = layer->GetLayerKind();
        
        if (layer_kind == CONV) {
            try {
                
                Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);
                if (layer_idx == 0) {
                  conv2d->GenInputAsync(stream);
                  input = conv2d->GetInput();
                } else conv2d->SetInput(intermediate_data);
                conv2d->Compute(candidate_name, stream);
                intermediate_data = conv2d->GetOutPut();
                //cout<<layer_idx<<" input tensor"<<conv2d->GetInputTensorDim()<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
                //cout<<"Layer Idx: "<<layer_idx<<" Computation: "<<conv2d->GetComputation()<<endl;
                
            } catch(std::bad_cast const& ex) {
                cout << "[ Conv2D"<<ex.what()<<"]" << endl;
            }    
        } else if (layer_kind == CONV_RELU) {
            try {
                //cout<<"layer:"<<layer<<endl;
                Conv2DRelu* conv2d = dynamic_cast<Conv2DRelu*>(layer);
                if (layer_idx == 0) {
                  conv2d->GenInputAsync(stream);
                  input = conv2d->GetInput();
                } else conv2d->SetInput(intermediate_data);
                conv2d->Compute(candidate_name, stream);
                intermediate_data = conv2d->GetOutPut();
		// cout<<intermediate_data<<layer_idx<<" input tensor: "<<conv2d->GetInputTensorDim()<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
                //cout<<layer_idx<<"conv2dRelu"<<intermediate_data<<endl;
            } catch(std::bad_cast const& ex) {
                cout << "[ Conv2D Relu"<<ex.what()<<"]" << endl;
            }    
        } else if (layer_kind == MAXPOOL) {
            try {
                MaxPool* maxpool = dynamic_cast<MaxPool*>(layer);
                
                if(layer_idx > 0) maxpool->SetInput(intermediate_data);
                maxpool->Compute(candidate_name, stream);
                intermediate_data = maxpool->GetOutPut();
                //cout<<layer_idx<<" maxpool "<<intermediate_data<<endl;
            } catch(std::bad_cast const& ex) {
                cout << "[ Maxpool"<<ex.what()<<"]" << endl;
            } 
        } else if (layer_kind == DENSE) {
            try {
                Dense* dense = dynamic_cast<Dense*>(layer);
                if(layer_idx > 0) dense->SetInput(intermediate_data);
                dense->Compute(candidate_name, stream);
                intermediate_data = dense->GetOutPut();
                //cout<<layer_idx<<" dense "<<intermediate_data<<" output: "<<dense->GetOutputTensorDim()<<endl;
            } catch(std::bad_cast const& ex) {
                cout << "Dense "<<ex.what()<<"]" << endl;
            } 
        } else if (layer_kind == DEPTHWISECONV) {
            try {
                DepthwiseConv* conv = dynamic_cast<DepthwiseConv*>(layer);
                if(layer_idx > 0) conv->SetInput(intermediate_data);
                conv->Compute(candidate_name, stream);
                intermediate_data = conv->GetOutPut();
                //cout<<layer_idx<<"depthwise"<<intermediate_data<<endl;
            } catch(std::bad_cast const& ex) {
                cout << "DepthwiseConv "<<ex.what()<<"]" << endl;
            }
        }
      }
      FinishRequest(1);
    }

    timer.Stop();
    auto latency = timer.SyncAndGetElapsedms();
    //GetBackendHandle(_runtime)->StreamSync(stream);
    //float latency = 1.0;    

    float throughput = (request_size / (latency - total_sleep_us/1000)) * 1000; 
    ret_list.push_back(throughput);
    
    avg_latency = (latency - total_sleep_us/1000)  / (request_size / stream_num);
    if(print_latency) {
      // cout<<task_name + " inference avg latency: "<<avg_latency<<"ms"<<" throughput: "<<throughput<<endl;
      // cout << avg_latency << "ms " << throughput << endl;
      cout << avg_latency << endl;
      // cout << throughput << endl;
    }

    return avg_latency;
            
  }


  virtual float RunKernelGroup(int start, int end, DLStream stream, float* input, float** output, vector<int>& selector) {

    ROCMTimer timer(stream);
    timer.Start();
    float* intermediate_data;
    for(int layer_idx=start; layer_idx<end; layer_idx++) {
      auto candidate_idx = selector[layer_idx];
      auto layer = _super_model[layer_idx];
      auto candidate_name = layer->GetCandidateNameById(candidate_idx);
      auto layer_kind = layer->GetLayerKind();
      
      if (layer_kind == CONV) {
          try {
              
              Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);
              if (layer_idx == start && start != 0 ) conv2d->SetInput(input);
              else if(layer_idx == start && start == 0 ) conv2d->CopyAndSetInput(input, stream);
              else  conv2d->SetInput(intermediate_data);
              conv2d->Compute(candidate_name, stream);
              intermediate_data = conv2d->GetOutPut();
              //cout<<layer_idx<<" conv "<<intermediate_data<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
              //cout<<"Layer Idx: "<<layer_idx<<" Computation: "<<conv2d->GetComputation()<<endl;
              
          } catch(std::bad_cast const& ex) {
              cout << "[ Conv2D"<<ex.what()<<"]" << endl;
          }    
      } else if (layer_kind == CONV_RELU) {
          try {
              //cout<<"layer:"<<layer<<endl;
              Conv2DRelu* conv2d = dynamic_cast<Conv2DRelu*>(layer);
              if (layer_idx == start && start != 0 ) conv2d->SetInput(input);
              else if(layer_idx == start && start == 0 ) conv2d->CopyAndSetInput(input, stream);
              else  conv2d->SetInput(intermediate_data);
              conv2d->Compute(candidate_name, stream);
              intermediate_data = conv2d->GetOutPut();
              //cout<<layer_idx<<"conv2dRelu"<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Conv2D Relu"<<ex.what()<<"]" << endl;
          }    
      } else if (layer_kind == MAXPOOL) {
          try {
              MaxPool* maxpool = dynamic_cast<MaxPool*>(layer);
              //if(layer_idx == start && !maxpool->hasInput()) maxpool->GenInputAsync(stream);
              //else if (layer_idx > start) maxpool->SetInput(intermediate_data);
              if (layer_idx == start && start != 0 ) maxpool->SetInput(input);
              else  maxpool->SetInput(intermediate_data);
              maxpool->Compute(candidate_name, stream);
              //cout<<layer_idx<<" maxpool "<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Maxpool"<<ex.what()<<"]" << endl;
          } 
      } else if (layer_kind == DENSE) {
          try {
              Dense* dense = dynamic_cast<Dense*>(layer);
              if (layer_idx == start && start != 0 ) dense->SetInput(input);
              else  dense->SetInput(intermediate_data);
              dense->Compute(candidate_name, stream);
              intermediate_data = dense->GetOutPut();
              //cout<<layer_idx<<" dense "<<intermediate_data<<" output: "<<dense->GetOutputTensorDim()<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "Dense "<<ex.what()<<"]" << endl;
          } 
      } else if (layer_kind == DEPTHWISECONV) {
          try {
              DepthwiseConv* conv2d = dynamic_cast<DepthwiseConv*>(layer);
              if (layer_idx == start && start != 0 ) conv2d->SetInput(input);
              else if(layer_idx == start && start == 0 ) conv2d->CopyAndSetInput(input, stream);
              else  conv2d->SetInput(intermediate_data);
              conv2d->Compute(candidate_name, stream);
              intermediate_data = conv2d->GetOutPut();
          } catch(std::bad_cast const& ex) {
              cout << "DepthwiseConv "<<ex.what()<<"]" << endl;
          }
      }
    }

    float latency = 0;
    //timer.Stop();
    //latency = timer.SyncAndGetElapsedms();
    GetBackendHandle(_runtime)->StreamSync(stream);
    if (end == this->GetLayerSize())
      GetBackendHandle(_runtime)->CopyDataFromToAsync(intermediate_data, output, _output_size, D2H, stream);
    
    return latency;
            
  }





  
  virtual float RunWithMultiStream(vector<int>& selector, vector<int>& requests, const string& task_name, int stream_num, vector<int>& random_list, vector<float>& ret_list, bool print_latency) {
    float avg_latency = 0.0;

    for(auto r : requests) {
        AddRequest(r);
    }

    int request_size = GetRequestSize();
    //cout<<"Total Request Size: "<<request_size<<endl;

    // Default Max stream number is 5
    int streams = stream_num;
    if (_run_stream_list.size() < stream_num) {
      for (int i=_run_stream_list.size(); i<stream_num; i++) {
        auto device_api =  GetBackendHandle(_runtime);
        auto stream = device_api->CreateStream(0);
        _run_stream_list.push_back(stream);
      }
    }
    

    ROCMTimer timer(_run_stream_list[0]);
    timer.Start();

    float total_sleep_us = 1.0;

    while(!RequestIsEmpty()) { 
      for(int layer_idx=0; layer_idx<_super_model.size(); layer_idx++) {
        // add sleep to improve randomness of kernel overlap
        //int random_us = random_list[request_global_id++];
        //total_sleep_us += float(random_us);
        //sleep.Compute(stream, random_us);
        float* intermediate_data[streams];
        int new_layer_idx = layer_idx;
        for(int pipeline_idx=0; pipeline_idx<streams; pipeline_idx++) {
          auto stream = _run_stream_list[pipeline_idx];
          auto candidate_idx = selector[layer_idx];
          Layer* layer;
          if (layer_idx >= pipeline_idx) {
            new_layer_idx = layer_idx- pipeline_idx;
            layer = _super_model[new_layer_idx] ;
          } else {
            break;
          }
          auto candidate_name = layer->GetCandidateNameById(candidate_idx);
          auto layer_kind = layer->GetLayerKind();
          cout<<" layer :"<<layer_idx<<" pipeline idx: "<<pipeline_idx<<" layer kind: "<<layer_kind<<endl;
          
          if (layer_kind == CONV) {
              try {
                  
                  Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);
                  if (new_layer_idx == 0) conv2d->GenInputAsync(stream);
                  else conv2d->SetInput(intermediate_data[pipeline_idx]);
                  conv2d->Compute(candidate_name, stream);
                  
                  intermediate_data[pipeline_idx] = conv2d->GetOutPut();
                  //cout<<new_layer_idx<<" conv "<<intermediate_data[pipeline_idx]<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
                  //cout<<"Layer Idx: "<<layer_idx<<" Computation: "<<conv2d->GetComputation()<<endl;
                  
              } catch(std::bad_cast const& ex) {
                  cout << "[ Conv2D"<<ex.what()<<"]" << endl;
              }    
          } else if (layer_kind == CONV_RELU) {
              try {
                  //cout<<"layer:"<<layer<<endl;
                  Conv2DRelu* conv2d = dynamic_cast<Conv2DRelu*>(layer);
                  if (new_layer_idx == 0) conv2d->GenInputAsync(stream);
                  else conv2d->SetInput(intermediate_data[pipeline_idx]);
                  conv2d->Compute(candidate_name, stream);
                  intermediate_data[pipeline_idx] = conv2d->GetOutPut();
                  cout<<new_layer_idx<<"conv2dRelu"<<intermediate_data[pipeline_idx]<<endl;
              } catch(std::bad_cast const& ex) {
                  cout << "[ Conv2D Relu"<<ex.what()<<"]" << endl;
              }    
          } else if (layer_kind == MAXPOOL) {
              try {
                  MaxPool* maxpool = dynamic_cast<MaxPool*>(layer);
                  
                  if(new_layer_idx > 0) maxpool->SetInput(intermediate_data[pipeline_idx]);
                  maxpool->Compute(candidate_name, stream);
                  intermediate_data[pipeline_idx] = maxpool->GetOutPut();
                  cout<<new_layer_idx<<" maxpool "<<intermediate_data[pipeline_idx]<<endl;
              } catch(std::bad_cast const& ex) {
                  cout << "[ Maxpool"<<ex.what()<<"]" << endl;
              } 
          } else if (layer_kind == DENSE) {
              try {
                  Dense* dense = dynamic_cast<Dense*>(layer);
                  if(new_layer_idx > 0) dense->SetInput(intermediate_data[pipeline_idx]);
                  dense->Compute(candidate_name, stream);
                  intermediate_data[pipeline_idx] = dense->GetOutPut();
                  cout<<new_layer_idx<<" dense "<<intermediate_data[pipeline_idx]<<" output: "<<dense->GetOutputTensorDim()<<endl;
              } catch(std::bad_cast const& ex) {
                  cout << "Dense "<<ex.what()<<"]" << endl;
              } 
          } else if (layer_kind == DEPTHWISECONV) {
             try {
                  DepthwiseConv* conv = dynamic_cast<DepthwiseConv*>(layer);
                  if(new_layer_idx > 0) conv->SetInput(intermediate_data[pipeline_idx]);
                  conv->Compute(candidate_name, stream);
                  intermediate_data[pipeline_idx] = conv->GetOutPut();
                  //cout<<layer_idx<<"depthwise"<<intermediate_data<<endl;
              } catch(std::bad_cast const& ex) {
                  cout << "DepthwiseConv "<<ex.what()<<"]" << endl;
              }
          }
        } 
      }

      for(int i=0; i<streams; i++) {
          GetBackendHandle(_runtime)->StreamSync(_run_stream_list[i]);
      }    

      FinishRequest(stream_num);
    }

    timer.Stop();
    auto latency = timer.SyncAndGetElapsedms();

    float throughput = (request_size / (latency - total_sleep_us/1000)) * 1000; 
    ret_list.push_back(throughput);
    
    avg_latency = (latency - total_sleep_us/1000)  / (request_size / stream_num);
    if(print_latency) {
      cout<<task_name + " inference avg latency: "<<avg_latency<<"ms"<<" throughput: "<<throughput<<endl;
    }

    return avg_latency;
            
  }


  virtual float LayerRunWarmup(int layer_idx, int candidate_idx, DLStream stream=nullptr) {
    auto layer = _super_model[layer_idx];
    auto candidate_name = layer->GetCandidateNameById(candidate_idx);
    auto layer_kind = layer->GetLayerKind();
    auto device_api =  GetBackendHandle(_runtime);
    bool need_free = (stream == nullptr);
    if (stream == nullptr)
      stream = device_api->CreateStream(0);

    ROCMTimer timer(stream);
    timer.Start();

    // Default Warm Up Set 100
    int warm_up = 500;
    for(int i=0; i<warm_up; i++) {
      if (layer_kind == CONV) {
        try {
            
            Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);
            if(!conv2d->hasInput()) conv2d->GenInputAsync(stream);
            conv2d->Compute(candidate_name, stream);
            //cout<<layer_idx<<" input tensor:"<<conv2d->GetInputTensorDim()<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
            //cout<<"Layer Idx: "<<layer_idx<<" Computation: "<<conv2d->GetComputation()<<endl;
            
        } catch(std::bad_cast const& ex) {
            cout << "[ Conv2D"<<ex.what()<<"]" << endl;
        }    
      } else if (layer_kind == CONV_RELU) {
          try {
              //cout<<"layer:"<<layer<<endl;
              Conv2DRelu* conv2d = dynamic_cast<Conv2DRelu*>(layer);
              if(!conv2d->hasInput()) conv2d->GenInputAsync(stream);
              conv2d->Compute(candidate_name, stream);
              //cout<<layer_idx<<"conv2dRelu"<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Conv2D Relu"<<ex.what()<<"]" << endl;
          }    
      } else if (layer_kind == DENSE) {
          try {
              Dense* dense = dynamic_cast<Dense*>(layer);
              if(!dense->hasInput()) dense->GenInputAsync(stream);
              dense->Compute(candidate_name, stream);
              //cout<<layer_idx<<"dense"<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "Dense "<<ex.what()<<"]" << endl;
          } 
      } else if (layer_kind == DEPTHWISECONV) {
          try {
              DepthwiseConv* conv = dynamic_cast<DepthwiseConv*>(layer);
              if(!conv->hasInput()) conv->GenInputAsync(stream);
              conv->Compute(candidate_name, stream);
              //cout<<layer_idx<<"depthwise"<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "DepthwiseConv "<<ex.what()<<"]" << endl;
          }
      } 
    }
    
    device_api->StreamSync(stream);
    timer.Stop();
    auto latency = timer.SyncAndGetElapsedms();
    // cout<<" inference total latency: "<<latency<<"ms"<<endl;
    if (need_free)
      device_api->FreeStream(0, stream);
    return latency;
  }

  virtual float LayerRun(int layer_idx, int candidate_idx, int repeat_num,  DLStream stream) {
    auto layer = _super_model[layer_idx];
    auto candidate_name = layer->GetCandidateNameById(candidate_idx);
    auto layer_kind = layer->GetLayerKind();

    auto device_api =  GetBackendHandle(_runtime);
    if (stream == nullptr)
      stream = device_api->CreateStream(0);

    ROCMTimer timer(stream);
    timer.Start();
    //cout<<"layer type:"<<layer_kind<<endl;
    for(int i=0; i<repeat_num; i++) {
      if (layer_kind == CONV) {
        try {
            
            Conv2D* conv2d = dynamic_cast<Conv2D*>(layer);

            if(!conv2d->hasInput()) conv2d->GenInputAsync(stream);
            
            conv2d->Compute(candidate_name, stream);
            //cout<<"layer name: "<<conv2d->GetLayerName()<<endl;
            //cout<<layer_idx<<" output tensor:"<<conv2d->GetOutputTensorDim()<<endl;
            //cout<<"Layer Idx: "<<layer_idx<<" Computation: "<<conv2d->GetComputation()<<endl;
            
        } catch(std::bad_cast const& ex) {
            cout << "[ Conv2D"<<ex.what()<<"]" << endl;
        }    
      } else if (layer_kind == CONV_RELU) {
          try {
              //cout<<"layer:"<<layer<<endl;
              Conv2DRelu* conv2d = dynamic_cast<Conv2DRelu*>(layer);
              if(!conv2d->hasInput()) conv2d->GenInputAsync(stream);
              conv2d->Compute(candidate_name, stream);
              //cout<<layer_idx<<"conv2dRelu"<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "[ Conv2D Relu"<<ex.what()<<"]" << endl;
          }    
      } else if (layer_kind == DENSE) {
          try {
              Dense* dense = dynamic_cast<Dense*>(layer);
              if(!dense->hasInput()) dense->GenInputAsync(stream);
              dense->Compute(candidate_name, stream);
              //cout<<layer_idx<<"dense"<<intermediate_data<<endl;
          } catch(std::bad_cast const& ex) {
              cout << "Dense "<<ex.what()<<"]" << endl;
          } 
      } else if (layer_kind == DEPTHWISECONV) {
          try {
              DepthwiseConv* conv = dynamic_cast<DepthwiseConv*>(layer);
              if(!conv->hasInput()) conv->GenInputAsync(stream);
              conv->Compute(candidate_name, stream);
          } catch(std::bad_cast const& ex) {
              cout << "DepthwiseConv "<<ex.what()<<"]" << endl;
          }
      }
    }
    GetBackendHandle(_runtime)->StreamSync(stream);
    timer.Stop();
    auto latency = timer.SyncAndGetElapsedms();
    //cout<<" inference total latency: "<<latency<<"ms"<<endl;
    return latency;
  }



  void AddRequest(int request_number) {
    _request_queue.push_back(request_number);
  }

  void FinishRequest(int finish_request_number) {
    while(finish_request_number>0) {
      int cur_num = _request_queue.front();
      if (finish_request_number >= cur_num) {
        _request_queue.pop_front();
        finish_request_number -= cur_num;
      } else {
        cur_num -= finish_request_number;
        _request_queue.pop_front();
        _request_queue.push_front(cur_num);
        return;
      }
    }
  }

  bool RequestIsEmpty() {
    return _request_queue.empty();
  }

  int GetRequestSize() {
    int size = 0;
    for (int i=0; i<_request_queue.size(); i++) {
      size += _request_queue[i];
    }
    return size;
  }

  string GetModelName() {
    return _model_name;
  }

  string GetLayerName(int layer_idx) {
    return _super_model[layer_idx]->GetLayerName();
  }

  float GetModelComputation() {
    float g_computation = 0;
    for(auto layer : _super_model) {
      float g_flop = 1.0 * layer->GetComputation()/1000000000;
      g_computation += g_flop;
      //cout<<"GFlops is:"<<g_flop<<endl;
    }
    return g_computation;
  }

 unsigned int GetOutputSize() {
  return _output_size;
 }

 unsigned int GetInputSize() {
  return _input_size;
 }

 void SetOutPutSize(unsigned int output_size) {
  _output_size = output_size;
 }

 void SetInPutSize(unsigned int input_size) {
  _input_size = input_size;
 }

  RUNTIME GetModelRuntime() {
    return _runtime;
  }

  int GetLayerSize() {
    return _super_model.size();
  }

  int GetRightSize() {
    return _right_size;
  }

  void SetRightSize(int right_size) {
    _right_size = right_size;
  }

  Layer* GetLayer(int layer_idx) {
    return _super_model[layer_idx];
  }

  int GetLayerBlock(int layer_idx, int candidate_idx) {
    return _super_model[layer_idx]->_block_latency_pair[candidate_idx]->GetBlock();
  }

  int GetLayerPreBlock(int layer_idx, int candidate_idx) {
    if (candidate_idx > 1 && _super_model[layer_idx]->_block_latency_pair.size() > 1) {
      return _super_model[layer_idx]->_block_latency_pair[candidate_idx-1]->GetBlock();
    }
    return -1;
  }

  float GetLayerLatency(int layer_idx, int candidate_idx) {
    //cout<<"layer idx: "<<layer_idx<<" candidate idx: "<<candidate_idx<<endl;
    return _super_model[layer_idx]->_block_latency_pair[candidate_idx]->GetLatency();
  }

  int GetLayerPreLatency(int layer_idx, int candidate_idx) {
    if (candidate_idx > 1 && _super_model[layer_idx]->_block_latency_pair.size() > 1) {
      return _super_model[layer_idx]->_block_latency_pair[candidate_idx-1]->GetLatency();
    }
    return -1;
  }

  int GetSatisfiedCandidateIdx(int layer_idx, int occupy_block) {
    int candidate_cnt = _super_model[layer_idx]->_block_latency_pair.size();
    //cout<<"layer idx: "<<layer_idx<<" candidate count: "<<candidate_cnt<<endl;
    for(int cidx=candidate_cnt-1; cidx>0; cidx--) {
      if (_super_model[layer_idx]->_block_latency_pair[cidx]->GetBlock() + occupy_block <= 108) {
        return cidx;
      }
    }
    return 0;
  }

  int GetLimitCandidateIdx(int layer_idx, int limit_block) {
    int candidate_cnt = _super_model[layer_idx]->_block_latency_pair.size();
    //cout<<"layer idx: "<<layer_idx<<" candidate count: "<<candidate_cnt<<endl;
    for(int cidx=candidate_cnt-1; cidx>0; cidx--) {
      if (_super_model[layer_idx]->_block_latency_pair[cidx]->GetBlock() <= limit_block) {
        return cidx;
      }
    }
    return 0;
  }

  int GetBestCandidateIdx(int layer_idx) {
    int candidate_cnt = _super_model[layer_idx]->_block_latency_pair.size();
    int best_cidx = 0;
    float best_latency = -1;
    for(int cidx=0; cidx<candidate_cnt; cidx++) {
      auto tmp_latency = _super_model[layer_idx]->_block_latency_pair[cidx]->GetLatency();
      if (cidx == 0) best_latency = tmp_latency;
      else if (tmp_latency < best_latency) {
        best_cidx = cidx;
        best_latency = tmp_latency;
      }
    }
    return best_cidx;
  }

  


 protected:
  vector<Layer*> _super_model;
  vector<DLStream> _run_stream_list;
  RUNTIME _runtime;
  string _model_name;
  
  deque<int> _request_queue;

  int _kernel_idx = 1;
  int _layer_idx = 1;

  int _right_size;

 private:
  unsigned int _output_size = 1;
  unsigned int _input_size = 1;

};

#endif
