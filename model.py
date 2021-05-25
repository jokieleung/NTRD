from models.transformer import TorchGeneratorModel,_build_encoder,_build_decoder,_build_encoder_mask, _build_encoder4kg, _build_decoder4kg, _build_decoder_selection
from models.utils import _create_embeddings,_create_entity_embeddings
from models.graph import SelfAttentionLayer,SelfAttentionLayer_batch
from torch_geometric.nn.conv.rgcn_conv import RGCNConv
from torch_geometric.nn.conv.gcn_conv import GCNConv
import pickle as pkl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from collections import defaultdict
import numpy as np
import json

def _load_kg_embeddings(entity2entityId, dim, embedding_path):
    kg_embeddings = torch.zeros(len(entity2entityId), dim)
    with open(embedding_path, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            entity = line[0]
            if entity not in entity2entityId:
                continue
            entityId = entity2entityId[entity]
            embedding = torch.Tensor(list(map(float, line[1:])))
            kg_embeddings[entityId] = embedding
    return kg_embeddings

EDGE_TYPES = [58, 172]
def _edge_list(kg, n_entity, hop):
    edge_list = []
    for h in range(hop):
        for entity in range(n_entity):
            # add self loop
            # edge_list.append((entity, entity))
            # self_loop id = 185
            edge_list.append((entity, entity, 185))
            if entity not in kg:
                continue
            for tail_and_relation in kg[entity]:
                if entity != tail_and_relation[1] and tail_and_relation[0] != 185 :# and tail_and_relation[0] in EDGE_TYPES:
                    edge_list.append((entity, tail_and_relation[1], tail_and_relation[0]))
                    edge_list.append((tail_and_relation[1], entity, tail_and_relation[0]))

    relation_cnt = defaultdict(int)
    relation_idx = {}
    for h, t, r in edge_list:
        relation_cnt[r] += 1
    for h, t, r in edge_list:
        if relation_cnt[r] > 1000 and r not in relation_idx:
            relation_idx[r] = len(relation_idx)

    return [(h, t, relation_idx[r]) for h, t, r in edge_list if relation_cnt[r] > 1000], len(relation_idx)

def concept_edge_list4GCN():
    node2index=json.load(open('key2index_3rd.json',encoding='utf-8'))
    f=open('conceptnet_edges2nd.txt',encoding='utf-8')
    edges=set()
    stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
    for line in f:
        lines=line.strip().split('\t')
        entity0=node2index[lines[1].split('/')[0]]
        entity1=node2index[lines[2].split('/')[0]]
        if lines[1].split('/')[0] in stopwords or lines[2].split('/')[0] in stopwords:
            continue
        edges.add((entity0,entity1))
        edges.add((entity1,entity0))
    edge_set=[[co[0] for co in list(edges)],[co[1] for co in list(edges)]]
    return torch.LongTensor(edge_set).cuda()

class CrossModel(nn.Module):
    def __init__(self, opt, dictionary, is_finetune=False, padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        # self.pad_idx = dictionary[dictionary.null_token]
        # self.start_idx = dictionary[dictionary.start_token]
        # self.end_idx = dictionary[dictionary.end_token]
        super().__init__()  # self.pad_idx, self.start_idx, self.end_idx)
        self.batch_size = opt['batch_size']
        self.max_r_length = opt['max_r_length']
        self.beam = opt['beam']
        self.is_finetune = is_finetune

        self.index2word={dictionary[key]:key for key in dictionary}

        self.movieID2selection_label=pkl.load(open('movieID2selection_label.pkl','rb'))

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        self.pad_idx = padding_idx
        self.embeddings = _create_embeddings(
            dictionary, opt['embedding_size'], self.pad_idx
        )

        self.concept_embeddings=_create_entity_embeddings(
            opt['n_concept']+1, opt['dim'], 0)
        self.concept_padding=0

        self.kg = pkl.load(
            open("data/subkg.pkl", "rb")
        )

        if opt.get('n_positions'):
            # if the number of positions is explicitly provided, use that
            n_positions = opt['n_positions']
        else:
            # else, use the worst case from truncate
            n_positions = max(
                opt.get('truncate') or 0,
                opt.get('text_truncate') or 0,
                opt.get('label_truncate') or 0
            )
            if n_positions == 0:
                # default to 1024
                n_positions = 1024

        if n_positions < 0:
            raise ValueError('n_positions must be positive')

        self.encoder = _build_encoder(
            opt, dictionary, self.embeddings, self.pad_idx, reduction=False,
            n_positions=n_positions,
        )
        self.decoder = _build_decoder4kg(
            opt, dictionary, self.embeddings, self.pad_idx,
            n_positions=n_positions,
        )

        self.selection_cross_attn_decoder = _build_decoder_selection(
            opt, len(self.movieID2selection_label), self.pad_idx,
            n_positions=n_positions,
        )
        self.db_norm = nn.Linear(opt['dim'], opt['embedding_size'])
        self.kg_norm = nn.Linear(opt['dim'], opt['embedding_size'])

        self.db_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])
        self.kg_attn_norm=nn.Linear(opt['dim'],opt['embedding_size'])

        self.enti_gcn_linear2_emb=nn.Linear(opt['dim'],opt['embedding_size'])

        self.criterion = nn.CrossEntropyLoss(reduce=False)

        self.self_attn = SelfAttentionLayer_batch(opt['dim'], opt['dim'])

        self.self_attn_db = SelfAttentionLayer(opt['dim'], opt['dim'])

        self.user_norm = nn.Linear(opt['dim']*2, opt['dim'])
        self.gate_norm = nn.Linear(opt['dim'], 1)
        self.copy_norm = nn.Linear(opt['embedding_size']*2+opt['embedding_size'], opt['embedding_size'])
        self.representation_bias = nn.Linear(opt['embedding_size'], len(dictionary) + 4)

        self.info_con_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_db_norm = nn.Linear(opt['dim'], opt['dim'])
        self.info_output_db = nn.Linear(opt['dim'], opt['n_entity'])
        self.info_output_con = nn.Linear(opt['dim'], opt['n_concept']+1)
        self.info_con_loss = nn.MSELoss(size_average=False,reduce=False)
        self.info_db_loss = nn.MSELoss(size_average=False,reduce=False)

        self.user_representation_to_bias_1 = nn.Linear(opt['dim'], 512)
        self.user_representation_to_bias_2 = nn.Linear(512, len(dictionary) + 4)

        self.output_en = nn.Linear(opt['dim'], opt['n_entity'])

        self.matching_linear = nn.Linear(opt['embedding_size'], opt['n_movies'])

        self.embedding_size=opt['embedding_size']
        self.dim=opt['dim']

        edge_list, self.n_relation = _edge_list(self.kg, opt['n_entity'], hop=2)
        edge_list = list(set(edge_list))
        print(len(edge_list), self.n_relation)
        self.dbpedia_edge_sets=torch.LongTensor(edge_list).cuda()
        self.db_edge_idx = self.dbpedia_edge_sets[:, :2].t()
        self.db_edge_type = self.dbpedia_edge_sets[:, 2]

        self.dbpedia_RGCN=RGCNConv(opt['n_entity'], self.dim, self.n_relation, num_bases=opt['num_bases'])
        #self.concept_RGCN=RGCNConv(opt['n_concept']+1, self.dim, self.n_con_relation, num_bases=opt['num_bases'])
        self.concept_edge_sets=concept_edge_list4GCN()
        self.concept_GCN=GCNConv(self.dim, self.dim)

        #self.concept_GCN4gen=GCNConv(self.dim, opt['embedding_size'])

        w2i=json.load(open('word2index_redial.json',encoding='utf-8'))
        self.i2w={w2i[word]:word for word in w2i}

        #---------------------------- still a hack ----------------------------2020/4/22 By Jokie
        self.mask4key=torch.Tensor(np.load('mask4key.npy')).cuda()
        self.mask4movie=torch.Tensor(np.load('mask4movie.npy')).cuda()
        # Original mask
        # self.mask4=self.mask4key+self.mask4movie

        # tmp hack for runable By Jokie 2021/04/12 for template generation task
        self.mask4=torch.ones(len(dictionary) + 4).cuda()

        if is_finetune:
            params = [self.dbpedia_RGCN.parameters(), self.concept_GCN.parameters(),
                      self.concept_embeddings.parameters(),
                      self.self_attn.parameters(), self.self_attn_db.parameters(), self.user_norm.parameters(),
                      self.gate_norm.parameters(), self.output_en.parameters()]
            for param in params:
                for pa in param:
                    pa.requires_grad = False

    def vector2sentence(self,batch_sen):
        sentences=[]
        for sen in batch_sen.numpy().tolist():
            sentence=[]
            for word in sen:
                if word>3:
                    sentence.append(self.index2word[word])
                elif word==3:
                    sentence.append('_UNK_')
            sentences.append(sentence)
        return sentences

    def _starts(self, bsz):
        """Return bsz start tokens."""
        return self.START.detach().expand(bsz, 1)

    def decode_greedy(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, bsz, maxlen):
        """
        Greedy search

        :param int bsz:
            Batch size. Because encoder_states is model-specific, it cannot
            infer this automatically.

        :param encoder_states:
            Output of the encoder model.

        :type encoder_states:
            Model specific

        :param int maxlen:
            Maximum decoding length

        :return:
            pair (logits, choices) of the greedy decode

        :rtype:
            (FloatTensor[bsz, maxlen, vocab], LongTensor[bsz, maxlen])
        """
        xs = self._starts(bsz)
        incr_state = None
        logits = []
        latents = []
        for i in range(maxlen):
            # todo, break early if all beams saw EOS
            scores, incr_state = self.decoder(xs, encoder_states, encoder_states_kg, encoder_states_db, incr_state)
            #batch*1*hidden
            scores = scores[:, -1:, :]
            #scores = self.output(scores)
            kg_attn_norm = self.kg_attn_norm(attention_kg)
            
            db_attn_norm = self.db_attn_norm(attention_db)

            copy_latent = self.copy_norm(torch.cat([kg_attn_norm.unsqueeze(1), db_attn_norm.unsqueeze(1), scores], -1))

            latents.append(scores) # WAY2:self attn matching model
            # latents.append(copy_latent) # WAY1:original matching linear model

            # logits = self.output(latent)
            con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
            voc_logits = F.linear(scores, self.embeddings.weight)
            # print(logits.size())
            # print(mem_logits.size())
            #gate = F.sigmoid(self.gen_gate_norm(scores))

            sum_logits = voc_logits + con_logits #* (1 - gate)
            _, preds = sum_logits.max(dim=-1)
            
            #scores = F.linear(scores, self.embeddings.weight)

            #print(attention_map)
            #print(db_attention_map)
            #print(preds.size())
            #print(con_logits.size())
            #exit()
            #print(con_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            #print(voc_logits.squeeze(0).squeeze(0)[preds.squeeze(0).squeeze(0)])
            
            #print(torch.topk(voc_logits.squeeze(0).squeeze(0),k=50)[1])

            #sum_logits = scores
            # print(sum_logits.size())

            #_, preds = sum_logits.max(dim=-1)
            logits.append(sum_logits)
            xs = torch.cat([xs, preds], dim=1)
            # check if everyone has generated an end token
            all_finished = ((xs == self.END_IDX).sum(dim=1) > 0).sum().item() == bsz
            if all_finished:
                break
        logits = torch.cat(logits, 1)
        latents = torch.cat(latents, 1)
        # return logits, xs
        return logits, xs, latents

    def decode_beam_search_with_kg(self, token_encoding, encoder_states_kg, encoder_states_db, attention_kg, attention_db, maxlen=None, beam=4):
        entity_reps, entity_mask = encoder_states_db
        word_reps, word_mask = encoder_states_kg
        entity_emb_attn = attention_db
        word_emb_attn = attention_kg
        batch_size = token_encoding[0].shape[0]
        
        inputs = self._starts(batch_size).long().reshape(1, batch_size, -1)
        incr_state = None

        sequences = [[[list(), list(), 1.0]]] * batch_size
        all_latents = []
        # for i in range(self.response_truncate):
        for i in range(maxlen):
            if i == 1:
                token_encoding = (token_encoding[0].repeat(beam, 1, 1),
                                  token_encoding[1].repeat(beam, 1, 1))
                entity_reps = entity_reps.repeat(beam, 1, 1)
                entity_emb_attn = entity_emb_attn.repeat(beam, 1)
                entity_mask = entity_mask.repeat(beam, 1)
                word_reps = word_reps.repeat(beam, 1, 1)
                word_emb_attn = word_emb_attn.repeat(beam, 1)
                word_mask = word_mask.repeat(beam, 1)

                encoder_states_kg = word_reps, word_mask
                encoder_states_db = entity_reps, entity_mask

            # at beginning there is 1 candidate, when i!=0 there are 4 candidates
            if i != 0:
                inputs = []
                for d in range(len(sequences[0])):
                    for j in range(batch_size):
                        text = sequences[j][d][0]
                        inputs.append(text)
                inputs = torch.stack(inputs).reshape(beam, batch_size, -1)  # (beam, batch_size, _)

            with torch.no_grad():
                
                dialog_latent, incr_state = self.decoder(inputs.reshape(len(sequences[0]) * batch_size, -1), token_encoding, encoder_states_kg, encoder_states_db, incr_state)
                # dialog_latent, incr_state = self.conv_decoder(
                #     inputs.reshape(len(sequences[0]) * batch_size, -1),
                #     token_encoding, word_reps, word_mask,
                #     entity_reps, entity_mask, incr_state
                # )
                dialog_latent = dialog_latent[:, -1:, :]  # (bs, 1, dim)
                
                concept_latent = self.kg_attn_norm(word_emb_attn).unsqueeze(1)
                db_latent = self.db_attn_norm(entity_emb_attn).unsqueeze(1)

                # print('concept_latent shape', concept_latent.shape)
                # print('db_latent shape', db_latent.shape)
                # print('dialog_latent shape', dialog_latent.shape)

                copy_latent = self.copy_norm(torch.cat((db_latent, concept_latent, dialog_latent), dim=-1))

                all_latents.append(copy_latent)

                # copy_logits = self.copy_output(copy_latent) * self.copy_mask.unsqueeze(0).unsqueeze(0)
                copy_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)
                gen_logits = F.linear(dialog_latent, self.embeddings.weight)
                sum_logits = copy_logits + gen_logits

            logits = sum_logits.reshape(len(sequences[0]), batch_size, 1, -1)
            # turn into probabilities,in case of negative numbers
            probs, preds = torch.nn.functional.softmax(logits).topk(beam, dim=-1)

            # (candeidate, bs, 1 , beam) during first loop, candidate=1, otherwise candidate=beam

            for j in range(batch_size):
                all_candidates = []
                for n in range(len(sequences[j])):
                    for k in range(beam):
                        prob = sequences[j][n][2]
                        logit = sequences[j][n][1]
                        if logit == []:
                            logit_tmp = logits[n][j][0].unsqueeze(0)
                        else:
                            logit_tmp = torch.cat((logit, logits[n][j][0].unsqueeze(0)), dim=0)
                        seq_tmp = torch.cat((inputs[n][j].reshape(-1), preds[n][j][0][k].reshape(-1)))
                        candidate = [seq_tmp, logit_tmp, prob * probs[n][j][0][k]]
                        all_candidates.append(candidate)
                ordered = sorted(all_candidates, key=lambda tup: tup[2], reverse=True)
                sequences[j] = ordered[:beam]

            # check if everyone has generated an end token
            all_finished = ((inputs == self.END_IDX).sum(dim=1) > 0).sum().item() == batch_size
            if all_finished:
                break
        
        # original solution
        # logits = torch.stack([seq[0][1] for seq in sequences])
        # inputs = torch.stack([seq[0][0] for seq in sequences])

        out_logits = []
        out_preds = []
        for beam_num in range(beam):
            cur_out_logits = torch.stack([seq[beam_num][1] for seq in sequences])
            curout_preds = torch.stack([seq[beam_num][0] for seq in sequences])
            out_logits.append(cur_out_logits)
            out_preds.append(curout_preds)

        logits = torch.cat([x for x in out_logits], dim=0)
        inputs = torch.cat([x for x in out_preds], dim=0)
        all_latents =  torch.cat(all_latents, 1)

        return logits, inputs, all_latents

    def decode_forced(self, encoder_states, encoder_states_kg, encoder_states_db, attention_kg, attention_db, ys):
        """
        Decode with a fixed, true sequence, computing loss. Useful for
        training, or ranking fixed candidates.

        :param ys:
            the prediction targets. Contains both the start and end tokens.

        :type ys:
            LongTensor[bsz, time]

        :param encoder_states:
            Output of the encoder. Model specific types.

        :type encoder_states:
            model specific

        :return:
            pair (logits, choices) containing the logits and MLE predictions

        :rtype:
            (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)
        inputs = ys.narrow(1, 0, seqlen - 1)
        inputs = torch.cat([self._starts(bsz), inputs], 1)
        latent, _ = self.decoder(inputs, encoder_states, encoder_states_kg, encoder_states_db) #batch*r_l*hidden

        kg_attention_latent=self.kg_attn_norm(attention_kg)

        #map=torch.bmm(latent,torch.transpose(kg_embs_norm,2,1))
        #map_mask=((1-encoder_states_kg[1].float())*(-1e30)).unsqueeze(1)
        #attention_map=F.softmax(map*map_mask,dim=-1)
        #attention_latent=torch.bmm(attention_map,encoder_states_kg[0])

        db_attention_latent=self.db_attn_norm(attention_db)

        #db_map=torch.bmm(latent,torch.transpose(db_embs_norm,2,1))
        #db_map_mask=((1-encoder_states_db[1].float())*(-1e30)).unsqueeze(1)
        #db_attention_map=F.softmax(db_map*db_map_mask,dim=-1)
        #db_attention_latent=torch.bmm(db_attention_map,encoder_states_db[0])

        copy_latent=self.copy_norm(torch.cat([kg_attention_latent.unsqueeze(1).repeat(1,seqlen,1), db_attention_latent.unsqueeze(1).repeat(1,seqlen,1), latent],-1))

        #logits = self.output(latent)
        con_logits = self.representation_bias(copy_latent)*self.mask4.unsqueeze(0).unsqueeze(0)#F.linear(copy_latent, self.embeddings.weight)
        logits = F.linear(latent, self.embeddings.weight)
        # print('logit size', logits.size())
        # print(mem_logits.size())
        #gate=F.sigmoid(self.gen_gate_norm(latent))

        sum_logits = logits+con_logits#*(1-gate)
        _, preds = sum_logits.max(dim=2)
        
        # return logits, preds, copy_latent
        return logits, preds, latent

    def infomax_loss(self, con_nodes_features, db_nodes_features, con_user_emb, db_user_emb, con_label, db_label, mask):
        #batch*dim
        #node_count*dim
        con_emb=self.info_con_norm(con_user_emb)
        db_emb=self.info_db_norm(db_user_emb)
        con_scores = F.linear(db_emb, con_nodes_features, self.info_output_con.bias)
        db_scores = F.linear(con_emb, db_nodes_features, self.info_output_db.bias)

        info_db_loss=torch.sum(self.info_db_loss(db_scores,db_label.cuda().float()),dim=-1)*mask.cuda()
        info_con_loss=torch.sum(self.info_con_loss(con_scores,con_label.cuda().float()),dim=-1)*mask.cuda()

        return torch.mean(info_db_loss), torch.mean(info_con_loss)

    def forward(self, xs, ys, mask_ys, concept_mask, db_mask, seed_sets, labels, con_label, db_label, entity_vector, rec,movies_gth=None, movie_nums=None, test=True, cand_params=None, prev_enc=None, maxlen=None,
                bsz=None):
        """
        Get output predictions from the model.

        :param xs:
            input to the encoder
        :type xs:
            LongTensor[bsz, seqlen]
        :param ys:
            Expected output from the decoder. Used
            for teacher forcing to calculate loss.
        :type ys:
            LongTensor[bsz, outlen]
        :param prev_enc:
            if you know you'll pass in the same xs multiple times, you can pass
            in the encoder output from the last forward pass to skip
            recalcuating the same encoder output.
        :param maxlen:
            max number of tokens to decode. if not set, will use the length of
            the longest label this model has seen. ignored when ys is not None.
        :param bsz:
            if ys is not provided, then you must specify the bsz for greedy
            decoding.

        :return:
            (scores, candidate_scores, encoder_states) tuple

            - scores contains the model's predicted token scores.
              (FloatTensor[bsz, seqlen, num_features])
            - candidate_scores are the score the model assigned to each candidate.
              (FloatTensor[bsz, num_cands])
            - encoder_states are the output of model.encoder. Model specific types.
              Feed this back in to skip encoding on the next call.
        """
        if test == False:
            # TODO: get rid of longest_label
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        # use cached encoding if available
        #xxs = self.embeddings(xs)
        #mask=xs == self.pad_idx
        # encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)

        # graph network
        db_nodes_features = self.dbpedia_RGCN(None, self.db_edge_idx, self.db_edge_type)
        con_nodes_features=self.concept_GCN(self.concept_embeddings.weight,self.concept_edge_sets)

        user_representation_list = []
        db_con_mask=[]
        for i, seed_set in enumerate(seed_sets):
            if seed_set == []:
                user_representation_list.append(torch.zeros(self.dim).cuda())
                db_con_mask.append(torch.zeros([1]))
                continue
            user_representation = db_nodes_features[seed_set]  # torch can reflect
            user_representation = self.self_attn_db(user_representation)
            user_representation_list.append(user_representation)
            db_con_mask.append(torch.ones([1]))

        db_user_emb=torch.stack(user_representation_list)
        db_con_mask=torch.stack(db_con_mask)

        graph_con_emb=con_nodes_features[concept_mask]
        con_emb_mask=concept_mask==self.concept_padding

        con_user_emb=graph_con_emb
        con_user_emb,attention=self.self_attn(con_user_emb,con_emb_mask.cuda())
        user_emb=self.user_norm(torch.cat([con_user_emb,db_user_emb],dim=-1))
        uc_gate = F.sigmoid(self.gate_norm(user_emb))
        user_emb = uc_gate * db_user_emb + (1 - uc_gate) * con_user_emb
        entity_scores = F.linear(user_emb, db_nodes_features, self.output_en.bias)
        #entity_scores = scores_db * gate + scores_con * (1 - gate)
        #entity_scores=(scores_db+scores_con)/2

        # select the topk entity for selection module TODO: By JOkie
        # topk_ent_probs, topk_ent_ind = torch.topk(entity_scores,k=50, dim=-1) # entity_scores [bsz, n_entities], topk_ent_ind [bsz, n_topk]
        # movie_embed = self.enti_gcn_linear2_emb(db_nodes_features[topk_ent_ind]) # db_nodes_features[n_entities, enti_dim], movie_embed [bsz, n_topk, embedding_size]


        #mask loss
        #m_emb=db_nodes_features[labels.cuda()]
        #mask_mask=concept_mask!=self.concept_padding
        mask_loss=0#self.mask_predict_loss(m_emb, attention, xs, mask_mask.cuda(),rec.float())

        info_db_loss, info_con_loss=self.infomax_loss(con_nodes_features,db_nodes_features,con_user_emb,db_user_emb,con_label,db_label,db_con_mask)

        #entity_scores = F.softmax(entity_scores.cuda(), dim=-1).cuda()

        rec_loss=self.criterion(entity_scores.squeeze(1).squeeze(1).float(), labels.cuda())
        #rec_loss=self.klloss(entity_scores.squeeze(1).squeeze(1).float(), labels.float().cuda())
        rec_loss = torch.sum(rec_loss*rec.float().cuda())

        self.user_rep=user_emb

        if self.is_finetune:
            #generation---------------------------------------------------------------------------------------------------
            encoder_states = prev_enc if prev_enc is not None else self.encoder(xs)
            con_nodes_features4gen=con_nodes_features#self.concept_GCN4gen(con_nodes_features,self.concept_edge_sets)
            con_emb4gen = con_nodes_features4gen[concept_mask]
            con_mask4gen = concept_mask != self.concept_padding
            #kg_encoding=self.kg_encoder(con_emb4gen.cuda(),con_mask4gen.cuda())
            kg_encoding=(self.kg_norm(con_emb4gen),con_mask4gen.cuda())

            db_emb4gen=db_nodes_features[entity_vector] #batch*50*dim
            db_mask4gen=entity_vector!=0
            #db_encoding=self.db_encoder(db_emb4gen.cuda(),db_mask4gen.cuda())
            db_encoding=(self.db_norm(db_emb4gen),db_mask4gen.cuda())

            if test == False:
                # use teacher forcing  scores, pred: (FloatTensor[bsz, ys, vocab], LongTensor[bsz, ys])
                # print('shape of entity_scores', entity_scores.shape)
                # print('shape of rec label', labels.shape)
                # print('rec label', labels)
                # print('shape of movie label', movies_gth.shape)
                movies_gth = movies_gth * (movies_gth!=-1)

                # print('shape of movies_gth', torch.sum(movies_gth!=0, dim=(0,1)))
                # print('shape of gth masked hole num', torch.sum((mask_ys == 6), dim=(0,1)))
                # print('movie_nums', movie_nums)
                # print('__MOVIE__ position ', torch.sum((mask_ys == 6), dim=(1)))
                assert torch.sum(movies_gth!=0, dim=(0,1)) == torch.sum((mask_ys == 6), dim=(0,1))

                # cant run case : case 1 : [-15] case2: [-8] By Jokie tmp 2021/4/14
                # print(movies_gth[-15])
                # print(mask_ys[-15])
                # print(self.vector2sentence(mask_ys.cpu())[-15])

                # print('shape of encoder_states,kg_encoding,db_encoding,con_user_emb, db_user_emb, mask_ys', encoder_states[0].shape,kg_encoding[0].shape,db_encoding[0].shape,con_user_emb.shape, db_user_emb.shape, mask_ys.shape)
                scores, preds, latent = self.decode_forced(encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb, mask_ys)
                # print('shape of scores,preds, mask_ys, latent', scores.shape,preds.shape,mask_ys.shape,latent.shape)
                gen_loss = torch.mean(self.compute_loss(scores, mask_ys))

                #-------------------------------- stage2 movie selection loss-------------- by Jokie
                masked_for_selection_token = (mask_ys == 6)

                #---------------------------WAY1: simply linear-------------------------------
                # original implementation
                # selected_token_latent = torch.masked_select(latent, masked_for_selection_token.unsqueeze(-1).expand_as(latent)).view(-1, latent.shape[-1])
                # # matching_logits = self.matching_linear(selected_token_latent)

                #change for inference the R@10 R@50
                # matching_logits_ = self.matching_linear(latent)
                # matching_logits = torch.masked_select(matching_logits_, masked_for_selection_token.unsqueeze(-1).expand_as(matching_logits_)).view(-1, matching_logits_.shape[-1])

                #---------------------WAY2: self attn----------------------------------------------------
                matching_tensor, _ = self.selection_cross_attn_decoder(latent, encoder_states, db_encoding)
                matching_logits_ = self.matching_linear(matching_tensor)
                matching_logits = torch.masked_select(matching_logits_, masked_for_selection_token.unsqueeze(-1).expand_as(matching_logits_)).view(-1, matching_logits_.shape[-1])

                # ---------------------------------------------

                #W1: greedy
                _, matching_pred = matching_logits.max(dim=-1) # [bsz * dynamic_movie_nums] 
                #W2: sample
                # matching_pred = torch.multinomial(F.softmax(matching_logits, dim=-1), num_samples=1, replacement=True)
                movies_gth = torch.masked_select(movies_gth, (movies_gth!=0))
                selection_loss = torch.mean(self.compute_loss(matching_logits, movies_gth)) # movies_gth.squeeze(0):[bsz * dynamic_movie_nums]
                

                
            else:
                #---------------------------------------------Beam Search decode----------------------------------------
                # scores, preds, latent = self.decode_beam_search_with_kg(
                #     encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,
                #     maxlen, self.beam)
                # # #pred here is soft template prediction
                # # # --------------post process the prediction to full sentence
                # # #-------------------------------- stage2 movie selection loss-------------- by Jokie
                # preds_for_selection = preds[:, 1:] # skip the start_ind
                # # preds_for_selection = preds[:, 2:] # skip the start_ind
                # masked_for_selection_token = (preds_for_selection == 6)

                # # print('latent shape', latent.shape)
                # # print('preds_for_selection: ', preds_for_selection)
                # # print('masked_for_selection_token shape', masked_for_selection_token.shape)

                # selected_token_latent = torch.masked_select(latent, masked_for_selection_token.unsqueeze(-1).expand_as(latent)).view(-1, latent.shape[-1])
                # print('selected_token_latent shape: ' , selected_token_latent)
                # matching_logits = self.matching_linear(selected_token_latent)

                # _, matching_pred = matching_logits.max(dim=-1) # [bsz * dynamic_movie_nums]
                # # print('matching_pred', matching_pred.shape)


                #---------------------------------------------Greedy decode-------------------------------------------
                scores, preds, latent = self.decode_greedy(
                    encoder_states, kg_encoding, db_encoding, con_user_emb, db_user_emb,
                    bsz,
                    maxlen or self.longest_label
                )

                # #pred here is soft template prediction
                # # --------------post process the prediction to full sentence
                # #-------------------------------- stage2 movie selection loss-------------- by Jokie
                preds_for_selection = preds[:, 1:] # skip the start_ind
                masked_for_selection_token = (preds_for_selection == 6)

                
                #---------------------------WAY1: simply linear-------------------------------
                # # original implementation
                # selected_token_latent = torch.masked_select(latent, masked_for_selection_token.unsqueeze(-1).expand_as(latent)).view(-1, latent.shape[-1])
                # matching_logits = self.matching_linear(selected_token_latent)

                # #change for inference the R@10 R@50
                # matching_logits_ = self.matching_linear(latent)
                # matching_logits = torch.masked_select(matching_logits_, masked_for_selection_token.unsqueeze(-1).expand_as(matching_logits_)).view(-1, matching_logits_.shape[-1])

                #---------------------WAY2: self attn----------------------------------------------------
                matching_tensor, _ = self.selection_cross_attn_decoder(latent, encoder_states, db_encoding) 
                matching_logits_ = self.matching_linear(matching_tensor)    
                matching_logits = torch.masked_select(matching_logits_, masked_for_selection_token.unsqueeze(-1).expand_as(matching_logits_)).view(-1, matching_logits_.shape[-1])

                if matching_logits.shape[0] is not 0:
                    #W1: greedy
                    _, matching_pred = matching_logits.max(dim=-1) # [bsz * dynamic_movie_nums] 
                    #W2: sample
                    # matching_pred = torch.multinomial(F.softmax(matching_logits,dim=-1), num_samples=1, replacement=True)
                else:
                    matching_pred = None
                # print('matching_pred', matching_pred.shape)
                #---------------------------------------------Greedy decode(end)-------------------------------------------

                gen_loss = None
                selection_loss = None

            return scores, preds, entity_scores, rec_loss, gen_loss, mask_loss, info_db_loss, info_con_loss, selection_loss, matching_pred, matching_logits_

        else:
            return None, None, entity_scores, rec_loss, None, None, info_db_loss, info_con_loss, None, None

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.

        This is an abstract method, and *must* be implemented by the user.

        Its purpose is to provide beam search with a model-agnostic interface for
        beam search. For example, this method is used to sort hypotheses,
        expand beams, etc.

        For example, assume that encoder_states is an bsz x 1 tensor of values

        .. code-block:: python

            indices = [0, 2, 2]
            encoder_states = [[0.1]
                              [0.2]
                              [0.3]]

        then the output will be

        .. code-block:: python

            output = [[0.1]
                      [0.3]
                      [0.3]]

        :param encoder_states:
            output from encoder. type is model specific.

        :type encoder_states:
            model specific

        :param indices:
            the indices to select over. The user must support non-tensor
            inputs.

        :type indices: list[int]

        :return:
            The re-ordered encoder states. It should be of the same type as
            encoder states, and it must be a valid input to the decoder.

        :rtype:
            model specific
        """
        enc, mask = encoder_states
        if not torch.is_tensor(indices):
            indices = torch.LongTensor(indices).to(enc.device)
        enc = torch.index_select(enc, 0, indices)
        mask = torch.index_select(mask, 0, indices)
        return enc, mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        """
        Reorder incremental state for the decoder.

        Used to expand selected beams in beam_search. Unlike reorder_encoder_states,
        implementing this method is optional. However, without incremental decoding,
        decoding a single beam becomes O(n^2) instead of O(n), which can make
        beam search impractically slow.

        In order to fall back to non-incremental decoding, just return None from this
        method.

        :param incremental_state:
            second output of model.decoder
        :type incremental_state:
            model specific
        :param inds:
            indices to select and reorder over.
        :type inds:
            LongTensor[n]

        :return:
            The re-ordered decoder incremental states. It should be the same
            type as incremental_state, and usable as an input to the decoder.
            This method should return None if the model does not support
            incremental decoding.

        :rtype:
            model specific
        """
        # no support for incremental decoding at this time
        return None

    def compute_loss(self, output, scores):
        score_view = scores.view(-1)
        output_view = output.view(-1, output.size(-1))
        loss = self.criterion(output_view.cuda(), score_view.cuda())
        return loss

    def save_model(self,model_name='saved_model/net_parameter1.pkl'):
        torch.save(self.state_dict(), model_name)

    def load_model(self,model_name='saved_model/net_parameter1.pkl'):
        # self.load_state_dict(torch.load('saved_model/net_parameter1.pkl'))
        self.load_state_dict(torch.load(model_name), strict= False)

    def output(self, tensor):
        # project back to vocabulary
        output = F.linear(tensor, self.embeddings.weight)
        up_bias = self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_rep)))
        # up_bias = self.user_representation_to_bias_3(F.relu(self.user_representation_to_bias_2(F.relu(self.user_representation_to_bias_1(self.user_representation)))))
        # Expand to the whole sequence
        up_bias = up_bias.unsqueeze(dim=1)
        output += up_bias
        return output
