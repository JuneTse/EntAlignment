#coding:utf-8
import numpy as np
import tensorflow as tf
import logging
import os
import time
from evaluate import get_hits,get_new_aligned_entities

t=time.localtime()
mon=t[1]
date=t[2]
h=t[3]
m=t[4]        
def getLogger(path,data_dir,name="Logger",mode="a"):
    logger=logging.Logger(name)
    logger.setLevel(logging.INFO)
    # name="%s-%s-%s_%s"%(mon,date,h,name)
    name="%s-%s-%s-%s"%(mon,date,name,data_dir)
    filename=os.path.join(path,name)
    fh=logging.FileHandler(filename=filename,mode=mode)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.info("add logger")
    return logger


class BaseTrainer(object):
    def __init__(self, model, params):
        self.model = model
        self.iter=params.iter
        self.use_cosine=params.use_cosine
        self.init_params(params)
        data_dir=params.data_dir.replace("/","_")
        if params.iter:
            data_dir="%s_%s"%(data_dir,"iter")
        if params.num_layer:
            data_dir="%s_Layer%s"%(data_dir,params.num_layer)
        name = model.__class__.__name__
        # 参数保存路径
        self.ckpt_path = os.path.join(params.weight_path, "%s_%s" % (name, data_dir))
        if not os.path.exists(self.ckpt_path):
            os.mkdir(self.ckpt_path)
        self.log_path = os.path.join(params.log_path, model.__class__.__name__)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)
        self.log = getLogger(path=self.log_path, data_dir=data_dir,name=model.__class__.__name__)

    def init_params(self, params):
        self.batch_size = params.batch_size
        self.global_step = 0
        self.keep_prob = params.keep_prob
        # lr
        self.init_lr = params.lr
        self.lr=params.lr
        self.lr_decay = params.lr_decay
        self.lr_decay_step = params.lr_decay_step
        self.warm_up_epoch=params.warm_up_step

        self.num_neg = params.num_neg

        # self.data_helper=data_helper
        self.eval_step_num = params.eval_step_num or 100
        self.eval_epoch_num = params.eval_epoch_num or 10
        self.log_step_num = params.log_step_num or 100
        self.saver = None


    def get_feed_dict(self, batch_datas, mode="train"):
        feed_dict = {}
        feed_dict[self.model.features["src_triples"]] = batch_datas["src_triples"]
        feed_dict[self.model.features["tgt_triples"]] = batch_datas["tgt_triples"]
        feed_dict[self.model.features["all_src_ent"]] = batch_datas["all_src_ent"]
        feed_dict[self.model.features["all_tgt_ent"]]=batch_datas["all_tgt_ent"]
        feed_dict[self.model.features["src_ent"]] = batch_datas["src_ent"]
        feed_dict[self.model.features["tgt_ent"]]=batch_datas["tgt_ent"]
        feed_dict[self.model.features["neg_src_ent"]]=batch_datas["neg_src_ent"]
        feed_dict[self.model.features["neg_tgt_ent"]]=batch_datas["neg_tgt_ent"]
        feed_dict[self.model.features["ent_mask"]]=batch_datas["ent_mask"]
        #batch_triples
        feed_dict[self.model.features["batch_src_triples"]] = batch_datas["batch_src_triples"]
        feed_dict[self.model.features["batch_tgt_triples"]] = batch_datas["batch_tgt_triples"]
        feed_dict[self.model.features["batch_neg_src_tails"]]=batch_datas["batch_neg_src_tails"]
        feed_dict[self.model.features["batch_neg_tgt_tails"]]=batch_datas["batch_neg_tgt_tails"]
        return feed_dict

    def evaluate(self, sess, data_helper):
        print("evaluate model....")
        src_ent_embedding,tgt_ent_embedding=self.predict_embeddings(sess,data_helper)
        test_src_entities=data_helper.test_src_entities
        test_tgt_entities=data_helper.test_tgt_entities
        src_ent2degree=data_helper.src_ent2degree
        src_id2ent=data_helper.src_id2entities
        top_k,top_lr,top_rl,Lmrr,Rmrr,results=get_hits(src_ent_embedding,tgt_ent_embedding,test_src_entities,test_tgt_entities,
                                               top_k=[1,5,10,50,100],use_cosine=self.use_cosine,src_id2ent=src_id2ent)
        self.log.info('For each left:')
        self.log.info("Lmrr: %.4f"%Lmrr)
        for i in range(len(top_lr)):
                self.log.info('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(test_src_entities) * 100))
        self.log.info('For each right:')
        self.log.info("Rmrr: %.4f"%Rmrr)
        for i in range(len(top_rl)):
                self.log.info('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(test_src_entities) * 100))
        return results


    def get_new_aligned_entities(self,sess,data_helper):
        src_ent_embedding,tgt_ent_embedding=self.predict_embeddings(sess,data_helper)
        test_src_entities=data_helper.test_src_entities
        test_tgt_entities=data_helper.test_tgt_entities
        aligned_entities=get_new_aligned_entities(src_ent_embedding,tgt_ent_embedding,test_src_entities,test_tgt_entities)
        return aligned_entities
    def save_new_aligned_entities(self,sess,data_helper):
        new_aligned_entities=self.get_new_aligned_entities(sess,data_helper)
        print("new aligned_entities:", len(new_aligned_entities))
        data_helper.save_new_aligned_entities(new_aligned_entities)

    def predict(self, sess, data_helper):
        data_gen = data_helper.batch_generator(batch_size=self.batch_size)
        results = []
        start_time = time.time()
        for batch_datas in data_gen:
            output = self.predict_batch(sess, batch_datas)
            results.extend(output)
        end_time = time.time()
        print("time:%s" % (end_time - start_time))
        predicts = np.array(results)
        print("result shape:", predicts.shape)
        return predicts

    def predict_batch(self, sess, batch_datas):
        feed_dict = self.get_feed_dict(batch_datas, mode="predict")
        feed_dict[self.model.is_training] = False
        if "keep_prob" in self.model.features:
            feed_dict[self.model.features["keep_prob"]] = 1.0
        output = sess.run(self.model.scores, feed_dict)
        output = output[:, 0]
        return output

    def predict_embeddings(self, sess,data_helper):
        data_gen = data_helper.train_batch_generator(batch_size=self.batch_size, shuffle=False,is_training=False)
        for batch_datas in data_gen:
            # start_time = time.time()
            feed_dict = self.get_feed_dict(batch_datas)
            feed_dict[self.model.lr] = self.lr
            feed_dict[self.model.is_training] = True
            if "keep_prob" in self.model.features:
                feed_dict[self.model.features["keep_prob"]] = self.keep_prob
        src_entitiy_embedding,tgt_entity_embedding = \
            sess.run([self.model.src_entity_embedding,self.model.tgt_entity_embedding],feed_dict=feed_dict)
        return src_entitiy_embedding, tgt_entity_embedding

    def train(self, sess, data_helper, iter_num=50, shuffle=True):
        is_stop = False
        for epoch in range(iter_num):
            # self.log.info("epoch: %s" % epoch)
            data_gen = data_helper.train_batch_generator(batch_size=self.batch_size, shuffle=shuffle,cur_epoch=epoch)
            total_loss = 0
            epoch_start_time=time.time()
            for batch_datas in data_gen:

                feed_dict = self.get_feed_dict(batch_datas)
                feed_dict[self.model.lr] = self.lr
                feed_dict[self.model.is_training] = True
                if "keep_prob" in self.model.features:
                    feed_dict[self.model.features["keep_prob"]] = self.keep_prob
                # print(feed_dict)
                #train step
                start_time = time.time()
                _,loss=sess.run([self.model.train_op,self.model.loss], feed_dict=feed_dict)
                total_loss += loss
                end_time=time.time()
                self.global_step += 1
                # print("batch time: %s"%(end_time-start_time))
            if (epoch+1) % self.eval_epoch_num == 0:
                self.log.info("==============Epoch: %s=============="%epoch)
                self.evaluate(sess,data_helper)
                self.save_weights(sess, global_step=epoch)
                # if self.iter:
                #     self.save_new_aligned_entities(sess,data_helper)
            epoch_end_time = time.time()
            # if epoch%100==0:
            if epoch%50==0:
                print("epoch: %s, lr: %s ,total loss :%s, time: %s"%(epoch,self.lr,total_loss, epoch_end_time-epoch_start_time))
                self.log.info("total loss: %s" % (total_loss))
        self.save_weights(sess, global_step=epoch)
    def save_weights(self, sess, global_step=None, saver=None):
        if saver is None:
            if self.saver is None:
                self.saver = tf.train.Saver(max_to_keep=5)
            saver = self.saver
        saver.save(sess, save_path=os.path.join(self.ckpt_path, "weights.ckpt"), global_step=global_step)

    def restore_last_session(self, sess):
        saver = tf.train.Saver()
        # get checkpoint state
        ckpt = tf.train.get_checkpoint_state(self.ckpt_path)
        # restore session
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            print("restore params from %s" % ckpt.model_checkpoint_path)
        else:
            print("fail to restore..., ckpt:%s" %self.ckpt_path)
