#coding:utf-8
from collections import defaultdict
import numpy as np
from collections import Counter

def read_name2ids(path):
    name2ids={}
    id2names={}
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split("\t")
            # print(line)
            if len(line)==1:
                id=line[0]
                ent=""
            else:
                id,ent=line
            id=int(id)
            name2ids[ent]=id
            id2names[id]=ent
    return name2ids,id2names



def read_aligned_entities(path,sep="\t"):
    entities1=[]
    entities2=[]
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split(sep)
            # print(line)
            ent1,ent2=line
            entities1.append(ent1)
            entities2.append(ent2)
    return entities1,entities2

def read_triples(path,sep="\t"):
    triples=[]
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split(sep)
            h,r,t=line
            triples.append((h,r,t))
    return triples

def count_degree(triples):
    ent2degree=Counter()
    for h,r,t in triples:
        ent2degree.update([h,t])
    return ent2degree



def read_attributes(path):
    entity2attrs={}
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            line=line.strip().split()
            ent=line[0]
            attrs=line[1:]
            entity2attrs[ent]=attrs
    return entity2attrs

def convert_triple_to_ids(triples,entity2id,relatioin2id):
    triples_new=[]
    for h,r,t in triples:
        hid=entity2id[h]
        tid=entity2id[t]
        rid=relatioin2id[r]
        triples_new.append([hid,rid,tid])
    triples_new=np.array(triples_new)
    return triples_new

def convert_names_to_ids(names,name2id):
    names_new=[]
    for ent in names:
        eid=name2id[ent]
        names_new.append(eid)
    return names_new

def read_embedding(path,vocab_size,emb_dim=300):
    embedding=np.zeros(shape=[vocab_size,emb_dim])
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            id,vec=line.strip().split("\t")
            id=int(id)
            vec=[float(v) for v in vec.split()]
            embedding[id,:]=vec[:emb_dim]
    return embedding.astype("float32")


def read_entity2labels(path):
    ent2texts={}
    with open(path,encoding="utf-8") as fin:
        for line in fin:
            try:
                ent,name1,name2=line.strip().split("\t")
                name="%s==>%s"%(name1,name2)
            except:
                ent,name1=line.strip().split("\t")
                name=name1
            ent2texts[ent]=name
    return ent2texts