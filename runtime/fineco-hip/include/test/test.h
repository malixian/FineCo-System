#ifndef TEST_H_
#define TEST_H_

class  Test {
 public:
  virtual ~Test() {}

  void Serve(int request_num_1, int request_num_2, int repeat, int stream_num) {}

  void GenRequestStream(int request_num, vector<int>& request) {
    request.clear();
    for (int i=0; i<request_num; i++) {
        // rand batch size is [0, 9]
        //int rand_num = rand() % 9 +1;
        request.push_back(1);
    }
  }

  int GetRequestSize(vector<int>& requests) {
    int request_size = 0;
    for (int i=0; i<requests.size(); i++) {
        request_size += requests[i];
    }
    return request_size;
  }

  void GenRandomList(int request_size, vector<int>& random_list) {
    random_list.clear();
    //int random_us[6] = {0, 20, 30, 40, 50, 100};
    int random_us[6] = {0,0,0,0,0,0};
    for (int i=0; i<request_size; i++) {
        int rand_num = rand() % 6 +1;
        random_list.push_back(random_us[rand_num]);
    }
  }

  void GenRandomCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    for(int i=0; i<layer_count.size(); i++) {
        int cnt = layer_count[i];
        random_comb.push_back(rand() % cnt);
    }
  }

  void GenDefaultCombination(const vector<int>& layer_count, vector<int>& random_comb) {
    for(int i=0; i<layer_count.size(); i++) {
        int cnt = layer_count[i];
        // 默认候选集合最后一个是100%算力调优下的算子
        random_comb.push_back(cnt-1);

        //random_comb.push_back(0);
    }
  }

  void GenBestEffortCombination(const vector<int>& layer_count, vector<int>& comb) {
    for(int i=0; i<layer_count.size(); i++) {
        //int cnt = layer_count[i];
        comb.push_back(0);

    }
  }


  vector<vector<int>> GeneCombinations(vector<int>& layer_count) {
    int LayerNumber = layer_count.size();
    int list_number = 1;
    std::vector<std::vector<int>> pre_combs;
    for(int i=0; i<layer_count[0]; i++) {
        vector<int> tmp(1,i);
        pre_combs.push_back(tmp);
    }

    for(int i=1; i<LayerNumber; i++) {
        std::vector<std::vector<int>> tmp_combs;
        for(int j=0; j<layer_count[i]; j++) {
            for(int k=0; k<pre_combs.size(); k++) {
                vector<int> tmp(pre_combs[k]);
                tmp.push_back(j);
                tmp_combs.push_back(tmp);
            }
        }
        pre_combs = tmp_combs;
    }

    //print all combinations
    /*
    for(int i=0; i<pre_combs.size(); i++) {
        for(int j=0; j<pre_combs[i].size(); j++)
            std::cout<<pre_combs[i][j]<<" ";
        std::cout<<std::endl;
    }
    */
    return pre_combs;      
 }

};

#endif