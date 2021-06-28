#coding:utf-8
from data_helper import DataHelper
from models.models_IDGAT import IDGAT
from params import Params
from trainer import BaseTrainer
import tensorflow as tf
from data_reader import read_embedding
import json

import time


config={
    #data
    "src_triple_num":105998,
    "tgt_triple num": 115722,
    #model
    "num_layer":2,
    "emb_dim":200,
    "rel_dim":50,
    #train
    "optimizer":tf.train.GradientDescentOptimizer, 
    "lr":50,
    "iter":True
   }

zh_en={
    #data
    "src_triple_num":70414,
    "tgt_triple_num": 95142,
    "src_ent_num": 19388,
    "tgt_ent_num": 19572,
    "src_rel_num": 1701,
    "tgt_rel_num": 1323,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm": True,
    "output_norm": True,
    #train
    "optimizer":tf.train.AdamOptimizer,
    "lr":0.002,
    "iter":True
   }

fr_en={
    #data
    "src_triple_num":105998,
    "tgt_triple num": 115722,
    "src_ent_num": 19661,
    "tgt_ent_num": 19993,
    "src_rel_num": 903,
    "tgt_rel_num": 1208,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm":True,
    "output_norm":True,
    #train
    "optimizer":tf.train.AdamOptimizer,
    "lr":0.002,
    "iter":True
   }

ja_en={
    #data
    "src_triple_num":77214,
    "tgt_triple_num": 93484,
    "src_ent_num": 19814,
    "tgt_ent_num": 19780,
    "src_rel_num": 1299,
    "tgt_rel_num": 1153,
    #model
    "num_layer":2,
    "emb_dim":300,
    "rel_dim":20,
    "emb_norm":True,
    "output_norm":True,
    #train
    "optimizer":tf.train.AdamOptimizer,
    "lr":0.002,
    "iter":True
   }
def train(data_name="0.1/fr_en",data_param=fr_en):
    params=Params(data_name=data_name)
    params.update(data_param)

   #data
    data_helper=DataHelper(params)
    #embedding
    src_ent_embeding=read_embedding(params.src_ent_emb_path,vocab_size=params.src_ent_num,emb_dim=300)
    tgt_ent_embeding=read_embedding(params.tgt_ent_emb_path,vocab_size=params.tgt_ent_num,emb_dim=300)

    #model
    model=IDGAT(params,src_ent_embedding=src_ent_embeding,tgt_ent_embedding=tgt_ent_embeding)

    #trainer
    trainer=BaseTrainer(model,params)

    sess=tf.Session()
    sess.run(tf.global_variables_initializer())
    #train
    # trainer.restore_last_session(sess)
    start_time=time.time()
    trainer.train(sess,data_helper,iter_num=300)
    end_time=time.time()
    print("time:",end_time-start_time)

    #test
    results=trainer.evaluate(sess,data_helper)
    json.dump(results,open("results_IGAT_%s.json"%data_name,"w",encoding="utf-8"),ensure_ascii=False,indent=3)


    #new aligned
    new_aligned_entities=trainer.get_new_aligned_entities(sess,data_helper)
    print(len(new_aligned_entities))
    data_helper.save_new_aligned_entities(new_aligned_entities)

    sess.close()

if __name__=="__main__":
    data_params={"zh_en":zh_en,"ja_en":ja_en,"fr_en":fr_en}

    for dname in ["zh_en","ja_en","fr_en"]:
        data_name=dname
        data_param=data_params[dname]
        with tf.Graph().as_default():
            train(data_name,data_param)
