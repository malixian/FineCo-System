#ifndef CO_CONFIG_H_
#define CO_CONFIG_H_

class  CoConfig {
 public:
    virtual ~CoConfig() {}

    void UpdateSelectorConfig(vector<int>& selector1, vector<int>& selector2){
        ConstructOfflineResult();
        for(int i=0; i<_layer_idx_1.size(); i++) {
            int idx_1 = _layer_idx_1[i];
            int idx_2 = _layer_idx_2[i];
            selector1[idx_1] = _layer_1_candidates[i];
            selector2[idx_2] = _layer_2_candidates[i];
        }
    }

    virtual void ConstructOfflineResult() {}

 protected:
    vector<int> _layer_idx_1;
    vector<int> _layer_idx_2;
    vector<int> _layer_1_candidates;
    vector<int> _layer_2_candidates;

};

#endif