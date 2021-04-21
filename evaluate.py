#coding:utf-8
import time
import numpy as np
import scipy.spatial as spatial
from collections import defaultdict

def get_topk_results(scores,right_score,k=20):
    '''找最小的topk个结果
    时间复杂度n*k'''
    scores=list(enumerate(scores))
    topk=scores[:k]
    topk=sorted(topk,key=lambda x:x[1])
    rank_index=0
    for v in scores:
        if v[0]<right_score:
                rank_index+=1
        for i in range(0,k):
            if v[1]<topk[i][1]:
                topk.insert(i,v)
    return topk[:k],rank_index

def get_rank_index(scores,right_score):
    '''计算正确得分的排名
    时间复杂度n'''
    res=np.where(scores<right_score)
    rank_index=len(res[0])
    return rank_index

# a=np.array([1,2,3])
# print(np.where(a>1))

def get_topk_results(datahelper,src_name,src_embedding,tgt_embedding,k=5):
    src_ent2ids=datahelper.src_entity2ids
    src_id2ents=datahelper.src_id2entities
    tgt_ent2ids=datahelper.tgt_entity2ids
    tgt_id2ents=datahelper.tgt_id2entities

    src_id=src_ent2ids[src_name]
    src_emb=np.array([src_embedding[src_id]])
    tgt_emb=tgt_embedding
    sim = spatial.distance.cdist(src_emb, tgt_emb, metric='cityblock')
    #rank
    rank = sim[0, :].argsort()
    topk=rank[:k]
    topk=[tgt_id2ents[i] for i in topk]
    return topk

def get_errors(data_helper,src_ent_embeding,tgt_ent_embeding,error_path="errors.txt"):
        # get errors
        fout = open(error_path, "w", encoding="utf-8")
        rights=[]
        for src_id, tgt_id in zip(data_helper.test_src_entities, data_helper.test_tgt_entities):
                tgt_id = tgt_id[0]
                tgt_name = data_helper.tgt_id2entities[tgt_id]
                src_name = data_helper.src_id2entities[src_id[0]]
                results = get_topk_results(data_helper, src_name, src_ent_embeding, tgt_ent_embeding, k=10)
                res_id = data_helper.tgt_entity2ids[results[0]]
                if res_id != tgt_id:
                        # print(src_name,tgt_name)
                        # print(results)
                        src_label = data_helper.src_ent2labels.get(src_name, "")
                        fout.write("Error: %s @@@%s \t %s\n" % (src_name, src_label, tgt_name))
                        fout.write(" @@@\t@@@ ".join(results[:5]) + "\n")
                        fout.write("\n")
                else:
                        src_label = data_helper.src_ent2labels.get(src_name, "")
                        rights.append([src_name,src_label,tgt_name,results])
        for src_name,src_label,tgt_name,results in rights:
                fout.write("Right: %s @@@%s \t %s\n" % (src_name, src_label, tgt_name))
                fout.write(" @@@\t@@@ ".join(results[:5]) + "\n")
                fout.write("\n")

        fout.close()

def get_hits(src_embedding,tgt_embedding, src_test_ents,tgt_test_ents, top_k=(1, 5,10, 50, 100),use_cosine=False,src_id2ent={}):
        Lvec = np.array([src_embedding[e1[0]] for e1 in src_test_ents])
        Rvec = np.array([tgt_embedding[e2[0]] for e2 in tgt_test_ents])

        results={}

        # print("compute sims....")
        if use_cosine:
                print("compute cosine sims...")
                #cosine
                Lvec=Lvec/(np.sqrt(np.sum(Lvec*Lvec,axis=1,keepdims=True))+1e-8)
                Rvec=Rvec/(np.sqrt(np.sum(Rvec*Rvec,axis=1,keepdims=True))+1e-8)
                sim=-np.matmul(Lvec,Rvec.T)
        else:
                print("compute l1 sims...")
                sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')

        # print("rank....")
        top_lr = [0] * len(top_k)
        Lmrrs=[]

        for i in range(Lvec.shape[0]):
                rank = sim[i, :].argsort()
                ent=src_test_ents[i][0]
                ent_name=src_id2ent[ent]

                rank_index = np.where(rank == i)[0][0]
                results[ent_name] = [str(rank_index)]
                Lmrrs.append(1.0/(rank_index+1))
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_lr[j] += 1

        top_rl = [0] * len(top_k)
        Rmrrs=[]
        for i in range(Rvec.shape[0]):
                rank = sim[:, i].argsort()
                rank_index = np.where(rank == i)[0][0]
                Rmrrs.append(1/(rank_index+1))
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_rl[j] += 1
        Lmrr=np.mean(Lmrrs)
        Rmrr=np.mean(Rmrrs)
        print('For each left:')
        print("Lmrr: %.4f"%Lmrr)
        for i in range(len(top_lr)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(src_test_ents) * 100))
        print('For each right:')
        print("Rmrr: %.4f"%Rmrr)
        for i in range(len(top_rl)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(src_test_ents) * 100))

        return top_k,top_lr,top_rl,Lmrr,Rmrr,results


def get_hits_fast(src_embedding,tgt_embedding, src_test_ents,tgt_test_ents, top_k=(1, 5,10, 20,50,100),use_cosine=False):
        Lvec = np.array([src_embedding[e1[0]] for e1 in src_test_ents])
        Rvec = np.array([tgt_embedding[e2[0]] for e2 in tgt_test_ents])

        if use_cosine:
                print("compute cosine sims....")
                start_time=time.time()
                #cosine
                Lvec=Lvec/(np.sqrt(np.sum(Lvec*Lvec,axis=1,keepdims=True))+1e-8)
                Rvec=Rvec/(np.sqrt(np.sum(Rvec*Rvec,axis=1,keepdims=True))+1e-8)
                sim=-np.matmul(Lvec,Rvec.T)
                print("time:",time.time()-start_time)
        else:
                print("compute l1 sims...")
                start_time = time.time()
                sim = spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
                print("time:", time.time() - start_time)

        print("rank....")
        top_lr = [0] * len(top_k)
        Lmrr=[]
        for i in range(Lvec.shape[0]):
                right_score=sim[i][i]
                rank_index=get_rank_index(sim[i,:],right_score)
                Lmrr.append(1.0/(rank_index+1))
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_lr[j] += 1
        top_rl = [0] * len(top_k)
        Rmrr=[]
        for i in range(Rvec.shape[0]):
                right_score=sim[i][i]
                rank_index=get_rank_index(sim[:,i],right_score)
                Rmrr.append(1/(rank_index+1))
                for j in range(len(top_k)):
                        if rank_index < top_k[j]:
                                top_rl[j] += 1
        Lmrr=np.mean(Lmrr)
        Rmrr=np.mean(Rmrr)
        print('For each left:')
        print("Lmrr: %.4f"%Lmrr)
        for i in range(len(top_lr)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_lr[i] / len(src_test_ents) * 100))
        print('For each right:')
        print("Rmrr: %.4f"%Rmrr)
        for i in range(len(top_rl)):
                print('Hits@%d: %.2f%%' % (top_k[i], top_rl[i] / len(src_test_ents) * 100))

        return top_k,top_lr,top_rl,Lmrr,Rmrr


def get_new_aligned_entities(src_embedding,tgt_embedding, src_test_ents,tgt_test_ents):
        Lvec = np.array([src_embedding[e1[0]] for e1 in src_test_ents])
        Rvec = np.array([tgt_embedding[e2[0]] for e2 in tgt_test_ents])
        sim =spatial.distance.cdist(Lvec, Rvec, metric='cityblock')
        src2tgt={}
        for i in range(Lvec.shape[0]):
                rank = sim[i, :].argsort()
                tgt_id=rank[0]
                src_ent=src_test_ents[i][0]
                tgt_ent=tgt_test_ents[tgt_id][0]
                src2tgt[src_ent]=tgt_ent
        tgt2src={}
        for i in range(Rvec.shape[0]):
                rank = sim[:, i].argsort()
                src_id=rank[0]
                src_ent=src_test_ents[src_id][0]
                tgt_ent=tgt_test_ents[i][0]
                tgt2src[tgt_ent]=src_ent
        new_aligned_entities=[]
        for src_ent in src_test_ents:
                src_ent=src_ent[0]
                tgt_ent=src2tgt[src_ent]
                tgt_src_ent=tgt2src[tgt_ent]
                if src_ent==tgt_src_ent:
                        new_aligned_entities.append([src_ent,tgt_ent])
        return new_aligned_entities
