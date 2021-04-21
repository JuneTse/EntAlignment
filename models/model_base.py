#coding:utf-8
import tensorflow as tf
import math

class BaseModel(object):
    def __init__(self,params,src_ent_embedding=None,src_rel_embedding=None,
                 tgt_ent_embedding=None,tgt_rel_embedding=None):
        self.optimizer=params.optimizer or tf.train.GradientDescentOptimizer

        self.hidden_dim=params.hidden_dim
        self.init_src_entity_embedding=src_ent_embedding
        self.init_tgt_entity_embedding=tgt_ent_embedding
        self.init_src_relation_embedding=src_rel_embedding
        self.init_tgt_relation_embedding=tgt_rel_embedding

        self.neg_num=params.neg_num #负样本个数
        #vocab
        self.src_ent_num=params.src_ent_num
        self.src_rel_num=params.src_rel_num
        self.tgt_ent_num=params.tgt_ent_num
        self.tgt_rel_num=params.tgt_rel_num
        self.src_triple_num=params.src_triple_num
        self.tgt_triple_num=params.tgt_triple_num
        #params
        # self.lr=params.lr
        self.num_attention_head=params.num_attention_head or 1
        self.emb_dim=params.emb_dim or 100
        self.margin=params.margin or 3
        self.num_layer=params.num_layer or 2
        self.emb_norm=params.emb_norm
        self.output_norm=params.output_norm
        #placeholders
        self.build_placeholders()
        #model
        self.build_model(self.features)

    def build_placeholders(self):
        features={}
        features["src_triples"]=tf.placeholder(dtype=tf.int32,shape=[None,3],name="src_triple")
        features["tgt_triples"]=tf.placeholder(dtype=tf.int32,shape=[None,3],name="tgt_triple")
        #batch triples
        features["batch_src_triples"]=tf.placeholder(dtype=tf.int32,shape=[None,3],name="batch_src_triple")
        features["batch_tgt_triples"]=tf.placeholder(dtype=tf.int32,shape=[None,3],name="batch_tgt_triple")
        features["batch_neg_src_tails"]=tf.placeholder(dtype=tf.int32,shape=[None,10],name="batch_neg_src_tails")
        features["batch_neg_tgt_tails"]=tf.placeholder(dtype=tf.int32,shape=[None,10],name="batch_neg_tgt_tails")
        #aligned relations
        features["src_rel"]=tf.placeholder(dtype=tf.int32,shape=[None],name="src_rel")
        features["tgt_rel"]=tf.placeholder(dtype=tf.int32,shape=[None],name="tgt_rel")
        features["neg_src_rel"]=tf.placeholder(dtype=tf.int32,shape=[None,self.neg_num],name="neg_src_rel")
        features["neg_tgt_rel"]=tf.placeholder(dtype=tf.int32,shape=[None,self.neg_num],name="neg_tgt_rel")

        #aligned entities
        features["all_src_ent"]=tf.placeholder(dtype=tf.int32,shape=[None,1],name="all_src_ent")
        features["all_tgt_ent"]=tf.placeholder(dtype=tf.int32,shape=[None,1],name="all_tgt_ent")
        features["src_ent"]=tf.placeholder(dtype=tf.int32,shape=[None,1],name="src_ent")
        features["tgt_ent"]=tf.placeholder(dtype=tf.int32,shape=[None,1],name="tgt_ent")
        #mask,遮挡掉一个batch的pre-aligned的实体
        features["ent_mask"]=tf.placeholder(dtype=tf.float32,shape=[None,1],name="ent_mask")
        #负样本
        features["neg_src_ent"]=tf.placeholder(dtype=tf.int32,shape=[None,self.neg_num],name="neg_src")
        features["neg_tgt_ent"]=tf.placeholder(dtype=tf.int32,shape=[None,self.neg_num],name="neg_tgt")
        #超参数
        self.keep_prob=features["keep_prob"]=tf.placeholder(dtype=tf.float32,name="kee_prob")
        self.lr=tf.placeholder(dtype=tf.float32,name="lr")
        self.is_training=tf.placeholder(dtype=tf.bool,name="is_training")
        self.features=features

    def build_model(self,features):
        src_triples=features["src_triples"]
        tgt_triples=features["tgt_triples"]
        src_heads,src_relatioins,src_tails=src_triples[:,0],src_triples[:,1],src_triples[:,2]
        tgt_heads,tgt_relations,tgt_tails=tgt_triples[:,0],tgt_triples[:,1],tgt_triples[:,2]
        #gat
        src_ent_embedding,src_rel_embedding=self.get_init_embeddings(
            self.init_src_entity_embedding,self.init_src_relation_embedding,self.src_ent_num,self.src_rel_num,name="src")
        tgt_ent_embedding,tgt_rel_embedding=self.get_init_embeddings(
            self.init_tgt_entity_embedding,self.init_tgt_relation_embedding,self.tgt_ent_num,self.tgt_rel_num,name="tgt")
        with tf.variable_scope("gat"):
            src_entity_embedding,src_rel_embedding=self.gat(src_heads,src_tails,src_relatioins,src_ent_embedding,src_rel_embedding)
        with tf.variable_scope("gat",reuse=True):
            tgt_entity_embedding,tgt_rel_embedding=self.gat(tgt_heads,tgt_tails,tgt_relations,tgt_ent_embedding,tgt_rel_embedding)

        self.src_entity_embedding=src_entity_embedding
        self.tgt_entity_embedding=tgt_entity_embedding

        # 计算对齐实体之间的距离
        src_ent=features["src_ent"]
        tgt_ent=features["tgt_ent"]
        neg_src_ent=features["neg_src_ent"]
        neg_tgt_ent=features["neg_tgt_ent"]
        self.loss=self.get_loss(src_entity_embedding,tgt_entity_embedding,
                                src_ent,tgt_ent,neg_src_ent,neg_tgt_ent)

        self.train_op=self.get_train_op(self.loss,learning_rate=self.lr)
        # self.build_summary_op()

    def get_loss(self,src_entity_embedding,tgt_entity_embedding,
                 src_ent,tgt_ent,neg_src_ent,neg_tgt_ent):
        src_ent_vecs=tf.nn.embedding_lookup(src_entity_embedding,src_ent)
        tgt_ent_vecs=tf.nn.embedding_lookup(tgt_entity_embedding,tgt_ent)
        neg_src_ent_vecs=tf.nn.embedding_lookup(src_entity_embedding,neg_src_ent)
        neg_tgt_ent_vecs=tf.nn.embedding_lookup(tgt_entity_embedding,neg_tgt_ent)

        pos_dist=tf.reduce_sum(tf.abs(src_ent_vecs-tgt_ent_vecs),axis=2,keepdims=False)
        neg_dist_tgt=tf.reduce_sum(tf.abs(src_ent_vecs-neg_tgt_ent_vecs),axis=2)
        neg_dist_src=tf.reduce_sum(tf.abs(tgt_ent_vecs-neg_src_ent_vecs),axis=2)
        #计算损失
        loss1=tf.maximum(0.,pos_dist-neg_dist_src+self.margin)
        loss2=tf.maximum(0.,pos_dist-neg_dist_tgt+self.margin)
        loss=tf.reduce_mean(loss1)+tf.reduce_mean(loss2)
        return loss

    def get_init_embeddings(self,init_entity_embedding,init_relation_embedding,
                            entity_vocab_size,rel_vocab_size,name="src"):
        with tf.variable_scope("%s_embdding"%name):
            if init_entity_embedding is not None:
                entity_embeddings=tf.get_variable(name="entity_embeddings", initializer=init_entity_embedding,
                                                                  dtype=tf.float32)
                # entity_embeddings=tf.nn.l2_normalize(entity_embeddings,axis=1)
            else:
                entity_embeddings=tf.get_variable(name="entity_embeddings", shape=[entity_vocab_size,self.emb_dim],
                                                  initializer=tf.keras.initializers.he_uniform(),dtype=tf.float32,)
                # entity_embeddings=tf.nn.l2_normalize(entity_embeddings,axis=1)
            if init_relation_embedding is not None:
                relation_embeddings = tf.get_variable(name="relation_embeddings", initializer=init_relation_embedding,
                                                      dtype=tf.float32)
            else:
                relation_embeddings= tf.get_variable(name="relation_embeddings",shape=[rel_vocab_size,self.emb_dim*self.num_layer],
                                                    initializer=tf.keras.initializers.he_uniform(),dtype=tf.float32)
                # relation_embeddings=tf.nn.l2_normalize(relation_embeddings,axis=1)
            if self.emb_norm:
                print("emb_norm: %s"%self.emb_norm)
                entity_embeddings = tf.nn.l2_normalize(entity_embeddings, axis=1)
                relation_embeddings = tf.nn.l2_normalize(relation_embeddings, axis=1)
            return entity_embeddings,relation_embeddings

    def gat(self,heads,tails,relations,entity_embeddings,relation_embeddings):
        relation_emb=relation_embeddings
        entity_emb=entity_embeddings
        att_heads=self.num_attention_head
        num_layer=self.num_layer
        entity_outputs=[tf.expand_dims(entity_embeddings,axis=1)]
        for i in range(num_layer):
            with tf.variable_scope("layer_%s"%i):
                entity_emb=self.multiHeadGAT(heads,relations,tails,in_dim=self.emb_dim,rel_dim=self.emb_dim,out_dim=self.emb_dim,
                                           entity_embeddings=entity_emb,relation_embeddings=relation_emb,
                                            num_att_heads=att_heads,use_layer_norm=False,use_resdual=True)

            entity_outputs.append(tf.expand_dims(entity_emb,axis=1))
        if self.output_norm:
            print("output_norm")
            entity_emb=tf.nn.l2_normalize(entity_emb,axis=1)
        return entity_emb,relation_emb

    def multiHeadGAT(self,heads,relations,tails,in_dim,rel_dim,
                     out_dim,entity_embeddings,relation_embeddings,
                     num_att_heads=2,use_layer_norm=False,use_resdual=True):
        num_ent=entity_embeddings.get_shape()[0]
        outputs=[]
        for i in range(num_att_heads):
            with tf.variable_scope("att_head_%s"%i):
                output=self.sparseBiGATLayer(heads,relations,tails,in_dim,rel_dim,out_dim,
                                             entity_embeddings,relation_embeddings)
                outputs.append(tf.expand_dims(output,1))
        outputs=tf.concat(outputs,axis=1)
        outputs=tf.reduce_mean(outputs,axis=1)
        outputs=tf.reshape(outputs,[num_ent,self.emb_dim])
        # outputs=tf.nn.relu(outputs)
        if use_resdual:
            outputs=outputs+entity_embeddings
        if use_layer_norm:
            outputs=tf.nn.l2_normalize(outputs)
        return outputs

    def sparseBiGATLayer(self,heads,relations,tails,in_dim,rel_dim,out_dim,
                         entity_embeddings,relation_embeddings):
        n_e=entity_embeddings.get_shape()[0]
        # n_r=self.relation_vocab_size
        #embedding
        head_emb=tf.nn.embedding_lookup(entity_embeddings,heads)
        relation_emb=tf.nn.embedding_lookup(relation_embeddings,relations)

        tail_emb=tf.nn.embedding_lookup(entity_embeddings,tails)
        triples=tf.concat([head_emb,relation_emb,tail_emb],axis=-1)
        triple_num=tf.shape(triples,out_type=tf.int32)[0]
        #[node_id,triple_id]
        indices_in = tf.concat([tf.expand_dims(tails, 1), tf.expand_dims(tf.range(0, triple_num), 1)], axis=-1)
        indices_out = tf.concat([tf.expand_dims(heads, 1), tf.expand_dims(tf.range(0, triple_num), 1)], axis=-1)
        with tf.variable_scope("in_direction_gat"):
            with tf.variable_scope("gat_triple_dense"):
                W=tf.get_variable(name="weights",shape=[in_dim*2+rel_dim,out_dim],initializer=tf.initializers.he_normal())
                tf.add_to_collection("l2",tf.nn.l2_loss(W))
            triples_in=tf.matmul(triples,W)
            #attention score
            with tf.variable_scope("gat_att"):
                W=tf.get_variable(name="weights",shape=[out_dim,1],initializer=tf.initializers.he_normal())
                # tf.add_to_collection("l2", tf.nn.l2_loss(W))
                scores=tf.matmul(triples_in,W)
                scores=tf.nn.leaky_relu(scores)
                scores=tf.exp(scores)
                # scores=scores*triple_masks
                scores=tf.nn.dropout(scores,keep_prob=self.keep_prob)
            #score Sparse Matrix
            indices=tf.cast(indices_in,tf.int64)
            scores=tf.squeeze(scores,1)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[n_e,triple_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_in=tf.sparse_tensor_dense_matmul(scores_matrix,triples_in)/(scores_sum+1e-8)
        with tf.variable_scope("out_direction_gat"):
            with tf.variable_scope("gat_triple_dense"):
                W=tf.get_variable(name="weights",shape=[in_dim*2+rel_dim,out_dim],initializer=tf.initializers.he_normal())
                tf.add_to_collection("l2", tf.nn.l2_loss(W))
            triples_out=tf.matmul(triples,W)
            #attention score
            with tf.variable_scope("gat_att"):
                W=tf.get_variable(name="weights",shape=[out_dim,1],initializer=tf.initializers.he_normal())
                scores=tf.matmul(triples_out,W)
                scores=tf.nn.leaky_relu(scores)
                scores=tf.exp(scores)
                scores=tf.nn.dropout(scores,keep_prob=self.keep_prob)
            #score Sparse Matrix
            indices=tf.cast(indices_out,tf.int64)
            scores=tf.squeeze(scores,1)
            scores_matrix=tf.SparseTensor(indices=indices,values=scores,dense_shape=[n_e,triple_num])
            scores_sum=tf.sparse_reduce_sum(scores_matrix,axis=1,keepdims=True)
            #triple sparse Matrix
            outputs_out=tf.sparse_tensor_dense_matmul(scores_matrix,triples_out)/(scores_sum+1e-8)
        #merge in and out direction
        with tf.variable_scope("outputs"):
            outputs=tf.concat([outputs_in,outputs_out],axis=-1)
            outputs=tf.nn.dropout(outputs,keep_prob=self.keep_prob)
            outputs=tf.nn.relu(outputs)
            W=tf.get_variable(name="weights",shape=[out_dim*2,out_dim],initializer=tf.initializers.he_normal())
            tf.add_to_collection("l2",tf.nn.l2_loss(W))
            outputs=tf.matmul(outputs,W)
        return outputs

    def get_trainable_variables(self):
        return tf.trainable_variables()

    def build_summary_op(self):
        tf.summary.scalar("loss",self.loss)
        self.summary_op=tf.summary.merge_all()

    def get_train_op(self,loss,learning_rate=0.001):
        optimizer=self.optimizer(learning_rate)
        train_op=optimizer.minimize(loss)
        return train_op

