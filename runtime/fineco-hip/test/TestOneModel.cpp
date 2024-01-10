
class TestOneModel : public Test {
 public:
    template<typename M>
    void IsolationRun(const string& model_name) {
        M model(model_name);
        model.InitModel(ROCM);

        vector<int> layer_count;
        model.GetAllLayerCandidateCnt(layer_count);
        vector<int> random_comb;
        for(int i=0; i<layer_count.size(); i++) {
            int cnt = layer_count[i];
            random_comb.push_back(0);
        }
    
        vector<int> request;
        int request_num = 1;
        int request_size = 0;


        for (int i=0; i<request_num; i++) {
        // rand batch size is [0, 9]
            int rand_num = 1;
            request.push_back(rand_num);
            request_size += rand_num;
        }

        vector<int> random_list;
        int random_us[10] = {0, 50, 100, 150, 200, 250, 300, 350, 400, 500};

        for (int i=0; i<request_size; i++) {
            int rand_num = rand() % 9 +1;
            random_list.push_back(random_us[rand_num]);
        }

        vector<float> ret_list;

        //warm up
        model.Run(random_comb, request, model_name, 1, random_list, ret_list, false);

        model.Run(random_comb, request, model_name, 1, random_list, ret_list, true);

        float computation = model.GetModelComputation();
        cout<<"Model Computation is: "<<computation<<endl;
    }    

};
