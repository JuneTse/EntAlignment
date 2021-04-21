#coding:utf-8

class ParamsDict(dict):
    def __init__(self, d=None):
        if d is not None:
            for k,v in d.items():
                self[k] = v
        return super().__init__()
    def __key(self, key):
        return "" if key is None else key.lower()
    def __str__(self):
        import json
        return json.dumps(self)
    def __setattr__(self, key, value):
        self[self.__key(key)] = value
    def __getattr__(self, key):
        return self.get(self.__key(key))
    def __getitem__(self, key):
        return super().get(self.__key(key))
    def __setitem__(self, key, value):
        return super().__setitem__(self.__key(key), value)

class Params(ParamsDict):
    def __init__(self,data_name):
        self.src_triple_path = "datasets/%s/s_triples"%data_name  # source triple
        self.tgt_triple_path = "datasets/%s/t_triples"%data_name  # target triple
        self.aligned_rel_path = "datasets/%s/rel_ILLs"%data_name  # pre-aligned relations
        self.aligned_ent_path = "datasets/%s/ent_ILLs"%data_name  # pre-aligned entities
        self.aligned_train_ent_path = "datasets/%s/train_ent"%data_name  # pre-aligned entities
        self.aligned_dev_ent_path = "datasets/%s/dev_ent"%data_name  # pre-aligned entities
        self.aligned_test_ent_path = "datasets/%s/test_ent"%data_name  # pre-aligned entities
        self.src_attr_path = "datasets/%s/training_attrs_1"%data_name  # source attribute
        self.tgt_attr_path = "datasets/%s/training_attrs_2"%data_name  # target attribute
        self.src_entity2id_path = "datasets/%s/ent_ids_s"%data_name
        self.src_relation2id_path = "datasets/%s/rel_ids_s"%data_name
        self.tgt_entity2id_path = "datasets/%s/ent_ids_t"%data_name
        self.tgt_relation2id_path = "datasets/%s/rel_ids_t"%data_name
        self.src_ent_emb_path="datasets/%s/ent_emb_s"%data_name
        self.tgt_ent_emb_path="datasets/%s/ent_emb_t"%data_name
        self.new_aligned_path="datasets/%s/ent_ILLs_new"%data_name
        self.data_dir=data_name
        #surface name
        self.src_label_path="datasets/%s/s_labels"%data_name
        self.tgt_label_path="datasets/%s/t_labels"%data_name

        #vocab
        self.src_ent_num=19661
        self.tgt_ent_num=19993
        self.src_rel_num=903
        self.tgt_rel_num=1208
        self.src_triple_num=105998
        self.tgt_triple_num=115722
        # embedding
        self.emb_dim=200
        #model
        self.hidden_dim=100
        self.num_attention_head=1
        self.leaky_relu_alph=0.2
        self.num_layer=2
        #loss
        self.margin=3.0
        self.neg_num=30
        self.l2_reg_lambda=0.001
        # train
        self.keep_prob=0.8
        self.batch_size=3000
        self.lr=50
        self.lr_decay=1.0
        self.lr_decay_step=5
        self.warm_up_step=5
        # log
        self.eval_step_num=100
        self.eval_epoch_num=100
        self.weight_path="./weights/"
        self.log_path="./weights/"

if __name__=="__main__":
    p={"a":1}
    param=Params(data_dir="kinship")
    param.update(p)
    print(param)



