#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The standard way to train a model. After training, also computes validation
and test error.

The user must provide a model (with ``--model``) and a task (with ``--task`` or
``--pytorch-teacher-task``).

Examples
--------

.. code-block:: shell

  python -m parlai.scripts.train -m ir_baseline -t dialog_babi:Task:1 -mf /tmp/model
  python -m parlai.scripts.train -m seq2seq -t babi:Task10k:1 -mf '/tmp/model' -bs 32 -lr 0.5 -hs 128
  python -m parlai.scripts.train -m drqa -t babi:Task10k:1 -mf /tmp/model -bs 10

"""  # noqa: E501

# TODO List:
# * More logging (e.g. to files), make things prettier.

import numpy as np
from tqdm import tqdm
from math import exp
import os
os.environ['CUDA_VISIBLE_DEVICES']='3'
import signal
import json
import argparse
import pickle as pkl
from dataset import dataset,CRSdataset
from model import CrossModel
import torch.nn as nn
from torch import optim
import torch
try:
    import torch.version
    import torch.distributed as dist
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
from nltk.translate.bleu_score import sentence_bleu

TOTAL_NOVEL_MOVIES = [190468, 131077, 167942, 96274, 86045, 176159, 194600, 143404, 127029, 120886, 84028, 194621, 147538, 178258, 116824, 139354, 129114, 112735, 157810, 168050, 129144, 77946, 139391, 92288, 112779, 139411, 151700, 176293, 108725, 157878, 153792, 188619, 198861, 88275, 129239, 162022, 149737, 178413, 137465, 78076, 80131, 172293, 96519, 141576, 127243, 184595, 119062, 178456, 92441, 174394, 100667, 151866, 137533, 88393, 194891, 115021, 143696, 102747, 180575, 102761, 201070, 117103,133489, 141688, 162174, 180620, 160163, 194987, 137649, 151987, 174517, 201143, 108986, 119227, 111041, 94659, 117188, 168388, 137669, 104901, 184782, 135634, 100821, 90582, 174560, 190948, 76279, 137726, 145926, 139786, 160266, 127508, 150037, 127514, 92704, 203301, 170540, 76336, 125499, 100924, 135743, 152127, 82497, 197186, 117315, 125515, 170576, 84571, 189020, 141917, 125540, 78436, 82536, 121453, 168568, 182905, 88703, 137861, 121484, 90768, 178835, 201368, 178852, 203437, 135858, 150197, 127674, 139964, 195261, 123583, 201415, 137927, 115404, 96973, 154329, 185051, 131811, 137955, 185065, 92911, 94963, 144116, 146166, 187147, 127759, 203543, 146199, 133916, 174881, 135973, 174889, 80683, 144178, 164666, 138042, 125767, 76620, 93005, 78673, 179035, 90975, 174953, 123754, 95083, 86889, 189294, 191342, 191347, 129907, 105334, 82808, 109435, 78719, 172934, 95110, 144263, 187280, 127901, 89005, 76724, 97209, 105403, 82876, 123837, 125893, 179153, 91089, 168913, 105428, 115669, 107477, 91095, 101335, 95205, 103400, 134132, 171000, 183293, 109568, 87048, 85003, 105486, 160791, 87064, 107548, 132127, 205858, 181284, 128044, 156724, 162873, 169027, 195667, 95321, 85082, 154716, 78943, 101479, 101483, 173169, 169090, 201860, 76935, 142484, 193710, 191672, 185529, 195773, 206020, 206021, 206022, 181447, 206026, 206029, 179407, 206032, 206035, 173267, 85204, 160981, 206044, 206045, 206056, 199917, 206062, 206064, 201971, 199930, 206076, 206079, 206080, 156931, 206087, 206092, 138525, 105764, 148780, 85298, 152883, 177468, 159039, 171338, 144716, 109911, 154968, 191834, 169313, 109931, 99694, 163184, 114034, 200078, 157074, 97683, 152993, 200103, 91563, 105916, 91583, 105923, 144842, 95692, 142800, 148947, 126421, 120281, 105952, 128487, 120297, 83436, 87541, 157174, 196085, 77306, 105979, 140796, 146948, 124433, 165396, 140823, 130589, 124449, 95780, 128548, 196132, 165416, 124457, 95785, 126520, 81469, 81474, 89669, 114256, 181859, 165499, 151167, 179845, 175751, 85648, 186002, 157333, 79512, 151198, 81568, 102067, 169662, 181960, 136912, 77521, 106197, 194285, 102147, 190214, 145161, 198411, 110352, 114453, 169750, 114458, 194336, 186145, 151341, 112434, 141107, 165683, 104248, 139076, 79689, 173899, 110419, 128862, 188260, 161638, 153450, 81775, 116593, 188293, 141191, 178056, 145291, 118669, 200590, 180131, 106404, 110510, 139199, 92098, 96197, 98259, 161756, 124895, 159718, 114669]

def is_distributed():
    """
    Returns True if we are in distributed mode.
    """
    return TORCH_AVAILABLE and dist.is_available() and dist.is_initialized()

def setup_args():
    train = argparse.ArgumentParser()
    train.add_argument("-max_c_length","--max_c_length",type=int,default=256)
    train.add_argument("-max_r_length","--max_r_length",type=int,default=30)
    train.add_argument("-beam","--beam",type=int,default=1)
    # train.add_argument("-max_r_length","--max_r_length",type=int,default=256)
    train.add_argument("-batch_size","--batch_size",type=int,default=32)
    train.add_argument("-max_count","--max_count",type=int,default=5)
    train.add_argument("-use_cuda","--use_cuda",type=bool,default=True)
    train.add_argument("-is_template","--is_template",type=bool,default=True)
    train.add_argument("-infomax_pretrain","--infomax_pretrain",type=bool,default=False)
    train.add_argument("-load_dict","--load_dict",type=str,default=None)
    train.add_argument("-learningrate","--learningrate",type=float,default=1e-3)
    train.add_argument("-optimizer","--optimizer",type=str,default='adam')
    train.add_argument("-momentum","--momentum",type=float,default=0)
    train.add_argument("-is_finetune","--is_finetune",type=bool,default=False)
    train.add_argument("-embedding_type","--embedding_type",type=str,default='random')
    train.add_argument("-save_exp_name","--save_exp_name",type=str,default='saved_model/new_model')
    # train.add_argument("-saved_hypo_txt","--saved_hypo_txt",type=str,default='case_file/output_hypo_latest.txt')
    train.add_argument("-saved_hypo_txt","--saved_hypo_txt",type=str,default=None)
    train.add_argument("-load_model_pth","--load_model_pth",type=str,default='saved_model/net_parameter1.pkl')
    train.add_argument("-epoch","--epoch",type=int,default=30)
    train.add_argument("-gpu","--gpu",type=str,default='2')
    train.add_argument("-gradient_clip","--gradient_clip",type=float,default=0.1)
    train.add_argument("-gen_loss_weight","--gen_loss_weight",type=float,default=5)
    train.add_argument("-embedding_size","--embedding_size",type=int,default=300)

    train.add_argument("-n_heads","--n_heads",type=int,default=2)
    train.add_argument("-n_layers","--n_layers",type=int,default=2)
    train.add_argument("-ffn_size","--ffn_size",type=int,default=300)

    train.add_argument("-dropout","--dropout",type=float,default=0.1)
    train.add_argument("-attention_dropout","--attention_dropout",type=float,default=0.0)
    train.add_argument("-relu_dropout","--relu_dropout",type=float,default=0.1)

    train.add_argument("-learn_positional_embeddings","--learn_positional_embeddings",type=bool,default=False)
    train.add_argument("-embeddings_scale","--embeddings_scale",type=bool,default=True)

    train.add_argument("-n_movies","--n_movies",type=int,default=6924)
    train.add_argument("-n_entity","--n_entity",type=int,default=64368)
    train.add_argument("-n_relation","--n_relation",type=int,default=214)
    train.add_argument("-n_concept","--n_concept",type=int,default=29308)
    train.add_argument("-n_con_relation","--n_con_relation",type=int,default=48)
    train.add_argument("-dim","--dim",type=int,default=128)
    train.add_argument("-n_hop","--n_hop",type=int,default=2)
    train.add_argument("-kge_weight","--kge_weight",type=float,default=1)
    train.add_argument("-l2_weight","--l2_weight",type=float,default=2.5e-6)
    train.add_argument("-n_memory","--n_memory",type=float,default=32)
    train.add_argument("-item_update_mode","--item_update_mode",type=str,default='0,1')
    train.add_argument("-using_all_hops","--using_all_hops",type=bool,default=True)
    train.add_argument("-num_bases", "--num_bases", type=int, default=8)



    

    return train

class TrainLoop_fusion_rec():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.train_dataset=dataset('data/full_data.jsonl',opt)
        # self.train_dataset=dataset('data/train_data.jsonl',opt)

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}

        self.movieID2selection_label=pkl.load(open('movieID2selection_label.pkl','rb'))
        self.selection_label2movieID={self.movieID2selection_label[key]:key for key in self.movieID2selection_label}

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']
        self.infomax_pretrain=self.opt['infomax_pretrain']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}

        self.build_model(is_finetune)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self,is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        #self.model.load_model()
        losses=[]
        best_val_rec=0
        rec_stop=False
        if self.infomax_pretrain:
            for i in range(3):
                train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
                train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                                batch_size=self.batch_size,
                                                                shuffle=False)
                num=0
                for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,movies_gth,movie_nums in tqdm(train_dataset_loader):
                    seed_sets = []
                    batch_size = context.shape[0]
                    for b in range(batch_size):
                        seed_set = entity[b].nonzero().view(-1).tolist()
                        seed_sets.append(seed_set)
                    self.model.train()
                    self.zero_grad()

                    scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _, selection_loss, matching_pred=self.model(context.cuda(), response.cuda(), mask_response.cuda(),
                                                                                                                                concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec, movies_gth.cuda(),movie_nums,test=False)

                    joint_loss=info_db_loss#+info_con_loss

                    losses.append([info_db_loss])
                    self.backward(joint_loss)
                    self.update_params()
                    if num%50==0:
                        print('info db loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                        #print('info con loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                        losses=[]
                    num+=1

            print("masked loss pre-trained")
            losses=[]

        for i in range(self.epoch):
            train_set=CRSdataset(self.train_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,movies_gth,movie_nums in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, _, selection_loss, matching_pred=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie,concept_vec, db_vec, entity_vector.cuda(), rec, movies_gth.cuda(),movie_nums,test=False)

                joint_loss=rec_loss+0.025*info_db_loss#+0.0*info_con_loss#+mask_loss*0.05

                losses.append([rec_loss,info_db_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%50==0:
                    print('rec loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('info db loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

            output_metrics_rec = self.val()

            if best_val_rec > output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]:
                rec_stop=True
            else:
                best_val_rec = output_metrics_rec["recall@50"]+output_metrics_rec["recall@1"]
                self.model.save_model(model_name= self.opt['save_exp_name'] + '_best_recom_model.pkl')
                print("recommendation model saved once------------------------------------------------")

            if rec_stop==True:
                break

        _=self.val(is_test=True)

    def metrics_cal_rec(self,rec_loss,scores,labels):
        batch_size = len(labels.view(-1).tolist())
        self.metrics_rec["loss"] += rec_loss
        outputs = scores.cpu()
        outputs = outputs[:, torch.LongTensor(self.movie_ids)]
        _, pred_idx = torch.topk(outputs, k=100, dim=1)
        for b in range(batch_size):
            if labels[b].item()==0:
                continue
            target_idx = self.movie_ids.index(labels[b].item())
            self.metrics_rec["recall@1"] += int(target_idx in pred_idx[b][:1].tolist())
            self.metrics_rec["recall@10"] += int(target_idx in pred_idx[b][:10].tolist())
            self.metrics_rec["recall@50"] += int(target_idx in pred_idx[b][:50].tolist())
            self.metrics_rec["count"] += 1

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        recs=[]
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec,movies_gth,movie_nums in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss, selection_loss, matching_pred = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec,movies_gth.cuda(),movie_nums, test=True, maxlen=20, bsz=batch_size)

            recs.extend(rec.cpu())
            #print(losses)
            #exit()
            self.metrics_cal_rec(rec_loss, rec_scores, movie)

        output_dict_rec={key: self.metrics_rec[key] / self.metrics_rec['count'] for key in self.metrics_rec}
        output_dict_rec['count'] = self.metrics_rec['count']
        print(output_dict_rec)

        return output_dict_rec

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

class TrainLoop_fusion_gen():
    def __init__(self, opt, is_finetune):
        self.opt=opt
        self.train_dataset=dataset('data/train_data.jsonl',opt)

        self.dict=self.train_dataset.word2index
        self.index2word={self.dict[key]:key for key in self.dict}

        self.movieID2selection_label=pkl.load(open('movieID2selection_label.pkl','rb'))
        self.selection_label2movieID={self.movieID2selection_label[key]:key for key in self.movieID2selection_label}
        self.id2entity=pkl.load(open('data/id2entity.pkl','rb'))

        self.total_novel_movies = TOTAL_NOVEL_MOVIES

        self.batch_size=self.opt['batch_size']
        self.epoch=self.opt['epoch']

        self.use_cuda=opt['use_cuda']
        if opt['load_dict']!=None:
            self.load_data=True
        else:
            self.load_data=False
        self.is_finetune=False

        self.is_template = opt['is_template']

        self.movie_ids = pkl.load(open("data/movie_ids.pkl", "rb"))
        # Note: we cannot change the type of metrics ahead of time, so you
        # should correctly initialize to floats or ints here

        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"count":0}
        self.metrics_gen={"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0, "true_recall_movie_count":0, "res_movie_recall":0.0,"recall@1":0,"recall@10":0,"recall@50":0}

        self.build_model(is_finetune=True)

        if opt['load_dict'] is not None:
            # load model parameters if available
            print('[ Loading existing model params from {} ]'
                  ''.format(opt['load_dict']))
            states = self.model.load(opt['load_dict'])
        else:
            states = {}

        self.init_optim(
            [p for p in self.model.parameters() if p.requires_grad],
            optim_states=states.get('optimizer'),
            saved_optim_type=states.get('optimizer_type')
        )

    def build_model(self,is_finetune):
        self.model = CrossModel(self.opt, self.dict, is_finetune)
        if self.opt['embedding_type'] != 'random':
            pass
        if self.use_cuda:
            self.model.cuda()

    def train(self):
        # self.model.load_model()
        losses=[]
        best_val_gen=0
        best_val_rec=0
        gen_stop=False
        for i in range(self.epoch*3):
            train_set=CRSdataset(self.train_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
            train_dataset_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                            batch_size=self.batch_size,
                                                            shuffle=False)
            num=0
            for context,c_lengths,response,r_length,mask_response,mask_r_length,entity,entity_vector,movie,concept_mask,dbpedia_mask,concept_vec, db_vec,rec,movies_gth,movie_nums in tqdm(train_dataset_loader):
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)
                self.model.train()
                self.zero_grad()

                scores, preds, rec_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss, selection_loss, matching_pred,matching_scores=self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec,movies_gth.cuda(),movie_nums, test=False)

                gen_loss = self.opt['gen_loss_weight'] * gen_loss
                joint_loss= gen_loss + selection_loss

                losses.append([gen_loss, selection_loss])
                self.backward(joint_loss)
                self.update_params()
                if num%20==0:
                    print('gen loss is %f'%(sum([l[0] for l in losses])/len(losses)))
                    print('selection_loss is %f'%(sum([l[1] for l in losses])/len(losses)))
                    losses=[]
                num+=1

            output_metrics_gen = self.val(True)
            if best_val_gen > output_metrics_gen["dist4"]:
                pass
            else:
                best_val_gen = output_metrics_gen["dist4"]
                self.model.save_model(model_name= self.opt['save_exp_name'] + '_best_dist4.pkl')
                print("Best Dist4 generator model saved once------------------------------------------------")
            print("best dist4 is :", best_val_gen)

            if best_val_rec > output_metrics_gen["recall@50"] + output_metrics_gen["recall@1"]:
                pass
            else:
                best_val_rec = output_metrics_gen["recall@50"] + output_metrics_gen["recall@1"]
                self.model.save_model(model_name= self.opt['save_exp_name'] + '_best_Rec.pkl')
                print("Best Recall generator model saved once------------------------------------------------")
            print("best res_movie_R@1 is :", output_metrics_gen["recall@1"])
            print("best res_movie_R@10 is :", output_metrics_gen["recall@10"])
            print("best res_movie_R@50 is :", output_metrics_gen["recall@50"])
            print('cur selection_loss is %f'%(sum([l[1] for l in losses])/len(losses)))
            print('cur Epoch is : ', i)
            
            # if i % 5 ==0: # save each 5 epoch
            #     model_name = self.opt['save_exp_name'] + '_' + str(i) + '.pkl'
            #     self.model.save_model(model_name=model_name)
            #     print("generator model saved once------------------------------------------------")
            #     print('cur selection_loss is %f'%(sum([l[1] for l in losses])/len(losses)))

        _=self.val(is_test=True)

    def val(self,is_test=False):
        self.metrics_gen={"ppl":0,"dist1":0,"dist2":0,"dist3":0,"dist4":0,"bleu1":0,"bleu2":0,"bleu3":0,"bleu4":0,"count":0,"true_recall_movie_count":0, "res_movie_recall":0.0,"recall@1":0,"recall@10":0,"recall@50":0}
        self.metrics_rec={"recall@1":0,"recall@10":0,"recall@50":0,"loss":0,"gate":0,"count":0,'gate_count':0}
        self.model.eval()
        if is_test:
            val_dataset = dataset('data/test_data.jsonl', self.opt)
        else:
            val_dataset = dataset('data/valid_data.jsonl', self.opt)
        val_set=CRSdataset(val_dataset.data_process(True),self.opt['n_entity'],self.opt['n_concept'])
        val_dataset_loader = torch.utils.data.DataLoader(dataset=val_set,
                                                           batch_size=self.batch_size,
                                                           shuffle=False)
        inference_sum=[]
        golden_sum=[]
        context_sum=[]
        losses=[]
        recs=[]
        for context, c_lengths, response, r_length, mask_response, mask_r_length, entity, entity_vector, movie, concept_mask, dbpedia_mask, concept_vec, db_vec, rec,movies_gth,movie_nums in tqdm(val_dataset_loader):
            with torch.no_grad():
                seed_sets = []
                batch_size = context.shape[0]
                for b in range(batch_size):
                    seed_set = entity[b].nonzero().view(-1).tolist()
                    seed_sets.append(seed_set)

                #-----dump , run the first time only to get the gen_loss, could be optimized here ------By Jokie 2021/04/15
                _, _, _, _, gen_loss, mask_loss, info_db_loss, info_con_loss, selection_loss, _, _ = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec,movies_gth.cuda(),movie_nums, test=False)
                scores, preds, rec_scores, rec_loss, _, mask_loss, info_db_loss, info_con_loss, selection_loss, matching_pred, matching_scores = self.model(context.cuda(), response.cuda(), mask_response.cuda(), concept_mask, dbpedia_mask, seed_sets, movie, concept_vec, db_vec, entity_vector.cuda(), rec,movies_gth.cuda(),movie_nums,test=True, maxlen=20, bsz=batch_size)

            # golden_sum.extend(self.vector2sentence(response.cpu()))
            # inference_sum.extend(self.vector2sentence(preds.cpu()))
            # context_sum.extend(self.vector2sentence(context.cpu()))
                
                self.all_response_movie_recall_cal(preds.cpu(), matching_scores.cpu(),movies_gth.cpu())

            #-----------template pro-process gth response and prediction--------------------
            if self.is_template:
                golden_sum.extend(self.template_vector2sentence(response.cpu(), movies_gth.cpu()))
                if matching_pred is not None:
                    inference_sum.extend(self.template_vector2sentence(preds.cpu(), matching_pred.cpu()))
                else:
                    inference_sum.extend(self.template_vector2sentence(preds.cpu(), None))

            else:
                golden_sum.extend(self.vector2sentence(response.cpu()))
                inference_sum.extend(self.vector2sentence(preds.cpu()))
            context_sum.extend(self.vector2sentence(context.cpu()))


            recs.extend(rec.cpu())
            losses.append(torch.mean(gen_loss))
            #print(losses)
            #exit()

        self.metrics_cal_gen(losses,inference_sum,golden_sum,recs, beam=self.opt['beam'])

        output_dict_gen={}
        for key in self.metrics_gen:
            if 'bleu' in key:
                output_dict_gen[key]=self.metrics_gen[key]/self.metrics_gen['count']
            else:
                output_dict_gen[key]=self.metrics_gen[key]
        print(output_dict_gen)

        # f=open('context_test.txt','w',encoding='utf-8')
        # f.writelines([' '.join(sen)+'\n' for sen in context_sum])
        # f.close()

        # f=open('output_self_attn_no_decode_first_filled_template_test.txt','w',encoding='utf-8')
        # f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
        # f.close()
        
        if self.opt['saved_hypo_txt'] is not None:
            f=open(self.opt['saved_hypo_txt'],'w',encoding='utf-8')
            f.writelines([' '.join(sen)+'\n' for sen in inference_sum])
            f.close()

        # f=open('golden_test.txt','w',encoding='utf-8')
        # f.writelines([' '.join(sen)+'\n' for sen in golden_sum])
        # f.close()

        # f=open('case_visualize.txt','w',encoding='utf-8')
        # for cont, hypo, gold in zip(context_sum, inference_sum, golden_sum):
        #     f.writelines('context: '+' '.join(cont)+'\n')
        #     f.writelines('hypo: '+' '.join(hypo)+'\n')
        #     f.writelines('gold: '+' '.join(gold)+'\n')
        #     f.writelines('\n')
        # f.close()

        return output_dict_gen

    def all_response_movie_recall_cal(self,decode_preds, matching_scores,labels):

        # matching_scores is non-mask version [bsz, seq_len, matching_vocab]
        # decode_preds [bsz, seq_len]
        # labels [bsz, movie_length_with_padding]
        # print('decode_preds shape', decode_preds.shape)
        # print('matching_scores shape', matching_scores.shape)
        # print('labels shape', labels.shape)
        decode_preds = decode_preds[:, 1:] # removing the start index

        labels = labels * (labels!=-1) # removing the padding token

        batch_size, seq_len = decode_preds.shape[0], decode_preds.shape[1]
        for cur_b in range(batch_size):
            for cur_seq_len in range(seq_len):
                if decode_preds[cur_b][cur_seq_len] ==6: # word id is 6
                    _, pred_idx = torch.topk(matching_scores[cur_b][cur_seq_len], k=100, dim=-1)
                    targets = labels[cur_b]
                    for target in targets:
                        self.metrics_gen["recall@1"] += int(target in pred_idx[:1].tolist())
                        self.metrics_gen["recall@10"] += int(target in pred_idx[:10].tolist())
                        self.metrics_gen["recall@50"] += int(target in pred_idx[:50].tolist())

    def metrics_cal_gen(self,rec_loss,preds,responses,recs, beam=1):
        def bleu_cal(sen1, tar1):
            bleu1 = sentence_bleu([tar1], sen1, weights=(1, 0, 0, 0))
            bleu2 = sentence_bleu([tar1], sen1, weights=(0, 1, 0, 0))
            bleu3 = sentence_bleu([tar1], sen1, weights=(0, 0, 1, 0))
            bleu4 = sentence_bleu([tar1], sen1, weights=(0, 0, 0, 1))
            return bleu1, bleu2, bleu3, bleu4
        
        def response_movie_recall_cal(sen1, tar1):
            for word in sen1:
                if '@' in word: # if is movie
                    if word in tar1: # if in gth
                        return int(1)
                    else:
                        return int(0)
            return int(0)


        def distinct_metrics(outs):
            # outputs is a list which contains several sentences, each sentence contains several words
            unigram_count = 0
            bigram_count = 0
            trigram_count=0
            quagram_count=0
            unigram_set = set()
            bigram_set = set()
            trigram_set=set()
            quagram_set=set()
            for sen in outs:
                for word in sen:
                    unigram_count += 1
                    unigram_set.add(word)
                for start in range(len(sen) - 1):
                    bg = str(sen[start]) + ' ' + str(sen[start + 1])
                    bigram_count += 1
                    bigram_set.add(bg)
                for start in range(len(sen)-2):
                    trg=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2])
                    trigram_count+=1
                    trigram_set.add(trg)
                for start in range(len(sen)-3):
                    quag=str(sen[start]) + ' ' + str(sen[start + 1]) + ' ' + str(sen[start + 2]) + ' ' + str(sen[start + 3])
                    quagram_count+=1
                    quagram_set.add(quag)
            dis1 = len(unigram_set) / len(outs)#unigram_count
            dis2 = len(bigram_set) / len(outs)#bigram_count
            dis3 = len(trigram_set)/len(outs)#trigram_count
            dis4 = len(quagram_set)/len(outs)#quagram_count
            return dis1, dis2, dis3, dis4

        predict_s=preds
        golden_s=responses
        #print(rec_loss[0])
        self.metrics_gen["ppl"]+=sum([exp(ppl) for ppl in rec_loss])/len(rec_loss)
        generated=[]
        total_movie_gth_response_cnt = 0
        have_movie_res_cnt = 0
        loop = 0
        total_item_response_cnt=0
        total_hypo_word_count=0
        novel_pred_movies =[]
        non_novel_pred_movies =[]
        # for out, tar, rec in zip(predict_s, golden_s, recs):
        for out in predict_s:
            tar = golden_s[loop // beam]
            loop = loop+1
            bleu1, bleu2, bleu3, bleu4=bleu_cal(out, tar)
            generated.append(out)
            self.metrics_gen['bleu1']+=bleu1
            self.metrics_gen['bleu2']+=bleu2
            self.metrics_gen['bleu3']+=bleu3
            self.metrics_gen['bleu4']+=bleu4
            self.metrics_gen['count']+=1
            self.metrics_gen['true_recall_movie_count']+=response_movie_recall_cal(out, tar)
            for word in out:
                total_hypo_word_count +=1
                if '@' in word:
                    total_item_response_cnt+=1
                    try:
                        int_movie_id = int(word[1:])
                        if int_movie_id in set(self.total_novel_movies):
                            novel_pred_movies.append(int_movie_id)
                        else:
                            non_novel_pred_movies.append(int_movie_id)
                    except:
                        non_novel_pred_movies.append(word[1:])
                        pass
            
        total_target_word_count = 0
        for tar in golden_s:
            for word in tar:
                total_target_word_count +=1
                if '@' in word:
                    total_movie_gth_response_cnt+=1
            for word in tar:
                if '@' in word:
                    have_movie_res_cnt+=1
                    break

        dis1, dis2, dis3, dis4=distinct_metrics(generated)
        self.metrics_gen['dist1']=dis1
        self.metrics_gen['dist2']=dis2
        self.metrics_gen['dist3']=dis3
        self.metrics_gen['dist4']=dis4

        self.metrics_gen['res_movie_recall'] = self.metrics_gen['true_recall_movie_count'] / have_movie_res_cnt
        self.metrics_gen["recall@1"] = self.metrics_gen["recall@1"] / have_movie_res_cnt
        self.metrics_gen["recall@10"] = self.metrics_gen["recall@10"] / have_movie_res_cnt
        self.metrics_gen["recall@50"] = self.metrics_gen["recall@50"] / have_movie_res_cnt
        print('----------'*10)
        print('total_movie_gth_response_cnt: ', total_movie_gth_response_cnt)
        print('total_gth_response_cnt: ', len(golden_s))
        print('total_hypo_response_cnt: ', len(predict_s))
        print('hypo item ratio: ', total_item_response_cnt / len(predict_s))
        print('target item ratio: ', total_movie_gth_response_cnt / len(golden_s))
        print('have_movie_res_cnt: ', have_movie_res_cnt)
        print('len(novel_pred_movies): ', len(novel_pred_movies))
        print('novel_pred_movies: ', novel_pred_movies)
        print('len(non_novel_pred_movies): ', len(non_novel_pred_movies))
        print('num of different predicted movies: ', len(set(non_novel_pred_movies)))
        print('non_novel_pred_movies: ', set(non_novel_pred_movies))
        print('----------'*10)
        
    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.index2word[word])
                    # if word==6: #if MOVIE token
                    #     sentence.append(self.selection_label2movieID[selection_label])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences
    
    def template_vector2sentence(self,batch_sen, batch_selection_pred):
        sentences=[]
        all_movie_labels = []
        if batch_selection_pred is not None:
            batch_selection_pred = batch_selection_pred * (batch_selection_pred!=-1)
            batch_selection_pred = torch.masked_select(batch_selection_pred, (batch_selection_pred!=0))
            for movie in batch_selection_pred.numpy().tolist():
                all_movie_labels.append(movie)

        # print('all_movie_labels:', all_movie_labels)
        curr_movie_token = 0
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    if word==6: #if MOVIE token
                        # print('all_movie_labels[curr_movie_token]',all_movie_labels[curr_movie_token])
                        # print('selection_label2movieID',self.selection_label2movieID[all_movie_labels[curr_movie_token]])

                        # WAY1: original method
                        sentence.append('@' + str(self.selection_label2movieID[all_movie_labels[curr_movie_token]]))


                        # WAY2: print out the movie name, but should comment when calculating the gen metrics
                        # if self.id2entity[self.selection_label2movieID[all_movie_labels[curr_movie_token]]] is not None:
                        #     sentence.append(self.id2entity[self.selection_label2movieID[all_movie_labels[curr_movie_token]]].split('/')[-1])
                        # else:
                        #     sentence.append('@' + str(self.selection_label2movieID[all_movie_labels[curr_movie_token]]))


                        curr_movie_token +=1
                    else:
                        sentence.append(self.index2word[word])
                    
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)

            # print('[DEBUG]sentence : ')
            # print(u' '.join(sentence).encode('utf-8').strip())

        assert curr_movie_token == len(all_movie_labels)
        return sentences

    @classmethod
    def optim_opts(self):
        """
        Fetch optimizer selection.

        By default, collects everything in torch.optim, as well as importing:
        - qhm / qhmadam if installed from github.com/facebookresearch/qhoptim

        Override this (and probably call super()) to add your own optimizers.
        """
        # first pull torch.optim in
        optims = {k.lower(): v for k, v in optim.__dict__.items()
                  if not k.startswith('__') and k[0].isupper()}
        try:
            import apex.optimizers.fused_adam as fused_adam
            optims['fused_adam'] = fused_adam.FusedAdam
        except ImportError:
            pass

        try:
            # https://openreview.net/pdf?id=S1fUpoR5FQ
            from qhoptim.pyt import QHM, QHAdam
            optims['qhm'] = QHM
            optims['qhadam'] = QHAdam
        except ImportError:
            # no QHM installed
            pass

        return optims

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        """
        Initialize optimizer with model parameters.

        :param params:
            parameters from the model

        :param optim_states:
            optional argument providing states of optimizer to load

        :param saved_optim_type:
            type of optimizer being loaded, if changed will skip loading
            optimizer states
        """

        opt = self.opt

        # set up optimizer args
        lr = opt['learningrate']
        kwargs = {'lr': lr}
        kwargs['amsgrad'] = True
        kwargs['betas'] = (0.9, 0.999)

        optim_class = self.optim_opts()[opt['optimizer']]
        self.optimizer = optim_class(params, **kwargs)

    def backward(self, loss):
        """
        Perform a backward pass. It is recommended you use this instead of
        loss.backward(), for integration with distributed training and FP16
        training.
        """
        loss.backward()

    def update_params(self):
        """
        Perform step of optimization, clipping gradients and adjusting LR
        schedule if needed. Gradient accumulation is also performed if agent
        is called with --update-freq.

        It is recommended (but not forced) that you call this in train_step.
        """
        update_freq = 1
        if update_freq > 1:
            # we're doing gradient accumulation, so we don't only want to step
            # every N updates instead
            self._number_grad_accum = (self._number_grad_accum + 1) % update_freq
            if self._number_grad_accum != 0:
                return

        if self.opt['gradient_clip'] > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.opt['gradient_clip']
            )

        self.optimizer.step()

    def zero_grad(self):
        """
        Zero out optimizer.

        It is recommended you call this in train_step. It automatically handles
        gradient accumulation if agent is called with --update-freq.
        """
        self.optimizer.zero_grad()

if __name__ == '__main__':
    args=setup_args().parse_args()
    # import os
    # os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    print('CUDA_VISIBLE_DEVICES:', os.environ['CUDA_VISIBLE_DEVICES'])
    print(vars(args))
    if args.is_finetune==False:
        loop=TrainLoop_fusion_rec(vars(args),is_finetune=False)
        # loop.model.load_model('saved_model/net_parameter1_bu.pkl')
        loop.train()
    else:
        loop=TrainLoop_fusion_gen(vars(args),is_finetune=True)
        #Tips: should at least load one of the model By Jokie

        #if validation 
        #WAY1:
        # loop.model.load_model('saved_model/matching_linear_model/generation_model_best.pkl')

        #WAY2:
        # loop.model.load_model('saved_model/sattn_dialog_model_best.pkl')
        # loop.model.load_model('saved_model/generation_model_best.pkl')
        # loop.model.load_model('saved_model/generation_model.pkl')
        # loop.model.load_model('saved_model/self_attn_generation_model_22.pkl')

        #WAY3: insert
        # loop.model.load_model()
        loop.model.load_model(args.load_model_pth)

        loop.train()
    