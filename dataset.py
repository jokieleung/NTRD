import numpy as np
from tqdm import tqdm
import pickle as pkl
import json
from nltk import word_tokenize
import re
from torch.utils.data.dataset import Dataset
import numpy as np
from copy import deepcopy

# import sys
# # from importlib import reload
# # reload(sys)
# # sys.setdefaultencoding('utf-8')
# import importlib
# importlib.reload(sys)

MOVIE_TOKEN = '__MOVIE__'

class dataset(object):
    def __init__(self,filename,opt):
        self.entity2entityId=pkl.load(open('data/entity2entityId.pkl','rb'))
        self.movieID2selection_label=pkl.load(open('movieID2selection_label.pkl','rb'))
        self.entity_max=len(self.entity2entityId)

        self.id2entity=pkl.load(open('data/id2entity.pkl','rb'))
        self.subkg=pkl.load(open('data/subkg.pkl','rb'))    #need not back process
        self.text_dict=pkl.load(open('data/text_dict.pkl','rb'))

        self.batch_size=opt['batch_size']
        self.max_c_length=opt['max_c_length']
        self.max_r_length=opt['max_r_length']
        self.max_count=opt['max_count']
        self.entity_num=opt['n_entity']
        #self.word2index=json.load(open('word2index.json',encoding='utf-8'))

        f=open(filename,encoding='utf-8')
        self.data=[]
        self.corpus=[]
        for line in tqdm(f):
            lines=json.loads(line.strip())
            seekerid=lines["initiatorWorkerId"]
            recommenderid=lines["respondentWorkerId"]
            contexts=lines['messages']
            movies=lines['movieMentions']
            altitude=lines['respondentQuestions']
            initial_altitude=lines['initiatorQuestions']
            cases=self._context_reformulate(contexts,movies,altitude,initial_altitude,seekerid,recommenderid)
            self.data.extend(cases)

        #if 'train' in filename:

        # self.prepare_word2vec()
        self.word2index = json.load(open('word2index_redial.json', encoding='utf-8'))
        self.key2index=json.load(open('key2index_3rd.json',encoding='utf-8'))

        self.stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])

        #self.co_occurance_ext(self.data)
        #exit()

    def prepare_word2vec(self):
        import gensim
        # model=gensim.models.word2vec.Word2Vec(self.corpus,vector_size=300,min_count=1)
        model=gensim.models.word2vec.Word2Vec(self.corpus,size=300,min_count=1)
        model.save('word2vec_redial')
        # word2index = {word: i + 4 for i, word in enumerate(model.wv.index_to_key)}
        word2index = {word: i + 4 for i, word in enumerate(model.wv.index2word)}
        #word2index['_split_']=len(word2index)+4
        #json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)
        word2embedding = [[0] * 300] * 4 + [model[word] for word in word2index]+[[0]*300]
        import numpy as np
        
        word2index['_split_']=len(word2index)+4
        print('---Saving word2vec data... ----')
        json.dump(word2index, open('word2index_redial.json', 'w', encoding='utf-8'), ensure_ascii=False)

        print(np.shape(word2embedding))
        np.save('word2vec_redial.npy', word2embedding)

    def padding_w2v(self,sentence,max_length,transformer=True,is_response=False,movies_gth=None,pad=0,end=2,unk=3):
        vector=[]
        concept_mask=[]
        dbpedia_mask=[]
        for word in sentence:
            vector.append(self.word2index.get(word,unk))
            #if word.lower() not in self.stopwords:
            # key2index file may have problem due to the vocab size, TODO by Jokie 2021/04/12
            concept_mask.append(self.key2index.get(word.lower(),0))
            #else:
            #    concept_mask.append(0)
            
            # if '@' in word:
            #     try:
            #         entity = self.id2entity[int(word[1:])]
            #         id=self.entity2entityId[entity]
            #     except:
            #         id=self.entity_max
            #     dbpedia_mask.append(id)
            # else:
            #     dbpedia_mask.append(self.entity_max)
            
            if MOVIE_TOKEN in word:
            # if is_response and MOVIE_TOKEN in word:
                dbpedia_mask.append(1)
            else:
                dbpedia_mask.append(0)
        vector.append(end)
        concept_mask.append(0)
        # dbpedia_mask.append(self.entity_max)
        dbpedia_mask.append(0)

        if len(vector)>max_length:
            if transformer:
                movie_nums = sum(dbpedia_mask[-max_length:])
                if movie_nums!=0:
                    movies_gth = movies_gth[-movie_nums:]
                else:
                    movies_gth = []
                return vector[-max_length:],max_length,concept_mask[-max_length:],dbpedia_mask[-max_length:],movies_gth
            else:
                movie_nums = sum(dbpedia_mask[:max_length])
                movies_gth = movies_gth[:movie_nums]
                return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length],movies_gth
        else:
            length=len(vector)
            return vector+(max_length-len(vector))*[pad],length,\
                   concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[0],movies_gth
            # return vector+(max_length-len(vector))*[pad],length,\
            #        concept_mask+(max_length-len(vector))*[0],dbpedia_mask+(max_length-len(vector))*[self.entity_max]

    def padding_all_movies(self,movies,max_length,transformer=True,pad=-1):
        

        if len(movies)>max_length:
            # if transformer:
            return movies[-max_length:]
            # else:
            #     return vector[:max_length],max_length,concept_mask[:max_length],dbpedia_mask[:max_length]
        else:
            length=len(movies)
            return movies+(max_length-len(movies))*[pad], length

    def padding_context(self,contexts,pad=0,transformer=True):
        vectors=[]
        vec_lengths=[]
        #--------------------Useless ---------------------
        if transformer==False:
            if len(contexts)>self.max_count:
                for sen in contexts[-self.max_count:]:
                    vec,v_l=self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors,vec_lengths,self.max_count
            else:
                length=len(contexts)
                for sen in contexts:
                    vec, v_l = self.padding_w2v(sen,self.max_r_length,transformer)
                    vectors.append(vec)
                    vec_lengths.append(v_l)
                return vectors+(self.max_count-length)*[[pad]*self.max_c_length],vec_lengths+[0]*(self.max_count-length),length
        #--------------------Useless(end) ---------------------
        else:
            contexts_com=[]
            for sen in contexts[-self.max_count:-1]:
                contexts_com.extend(sen)
                contexts_com.append('_split_')
            contexts_com.extend(contexts[-1])
            vec,v_l,concept_mask,dbpedia_mask,_=self.padding_w2v(contexts_com,self.max_c_length,transformer)
            return vec,v_l,concept_mask,dbpedia_mask,0

    def response_delibration(self,response,unk='MASKED_WORD'):
        new_response=[]
        for word in response:
            # if word in self.key2index:
            if '@' in word:
                # print(word)
                new_response.append(unk)
            else:
                new_response.append(word)
        return new_response

    def data_process(self,is_finetune=False):
        data_set = []
        context_before = []
        for line in self.data:
            #if len(line['contexts'])>2:
            #    continue
            if is_finetune and line['contexts'] == context_before:
                continue
            else:
                context_before = line['contexts']

            

            # context,c_lengths,concept_mask,dbpedia_mask,_=self.padding_context(line['contexts'])
            context,c_lengths,concept_mask,dbpedia_mask_context,_=self.padding_context(line['contexts'])
            response,r_length,_,dbpedia_mask,movies_gth=self.padding_w2v(line['response'],self.max_r_length,transformer=True,is_response=True,movies_gth=line['all_movies'])
            
            #padding all_movies 
            movies_gth, movies_num = self.padding_all_movies(movies_gth,self.max_r_length)
            # movies_gth, movies_num = self.padding_all_movies(line['all_movies'],self.max_r_length)
            
            if False:
                mask_response,mask_r_length,_,_,_=self.padding_w2v(self.response_delibration(line['response']),self.max_r_length)
            else:
                mask_response, mask_r_length=response,r_length
            assert len(context)==self.max_c_length
            assert len(concept_mask)==self.max_c_length
            # assert len(dbpedia_mask)==self.max_c_length
            assert len(dbpedia_mask)==self.max_c_length or len(dbpedia_mask)==self.max_r_length

            data_set.append([np.array(context),c_lengths,np.array(response),r_length,np.array(mask_response),mask_r_length,line['entity'],
                             line['movie'],concept_mask,dbpedia_mask,line['rec'], np.array(movies_gth), movies_num])
        return data_set

    def co_occurance_ext(self,data):
        stopwords=set([word.strip() for word in open('stopwords.txt',encoding='utf-8')])
        keyword_sets=set(self.key2index.keys())-stopwords
        movie_wordset=set()
        for line in data:
            movie_words=[]
            if line['rec']==1:
                for word in line['response']:
                    if '@' in word:
                        try:
                            num=self.entity2entityId[self.id2entity[int(word[1:])]]
                            movie_words.append(word)
                            movie_wordset.add(word)
                        except:
                            pass
            line['movie_words']=movie_words
        new_edges=set()
        for line in data:
            if len(line['movie_words'])>0:
                before_set=set()
                after_set=set()
                co_set=set()
                for sen in line['contexts']:
                    for word in sen:
                        if word in keyword_sets:
                            before_set.add(word)
                        if word in movie_wordset:
                            after_set.add(word)
                for word in line['response']:
                    if word in keyword_sets:
                        co_set.add(word)

                for movie in line['movie_words']:
                    for word in list(before_set):
                        new_edges.add('co_before'+'\t'+movie+'\t'+word+'\n')
                    for word in list(co_set):
                        new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in line['movie_words']:
                        if word!=movie:
                            new_edges.add('co_occurance' + '\t' + movie + '\t' + word + '\n')
                    for word in list(after_set):
                        new_edges.add('co_after'+'\t'+word+'\t'+movie+'\n')
                        for word_a in list(co_set):
                            new_edges.add('co_after'+'\t'+word+'\t'+word_a+'\n')
        f=open('co_occurance.txt','w',encoding='utf-8')
        f.writelines(list(new_edges))
        f.close()
        json.dump(list(movie_wordset),open('movie_word.json','w',encoding='utf-8'),ensure_ascii=False)
        print(len(new_edges))
        print(len(movie_wordset))

    def entities2ids(self,entities):
        return [self.entity2entityId[word] for word in entities]

    def detect_movie(self,sentence,movies):
        token_text = word_tokenize(sentence)
        num=0
        token_text_com=[]
        masked_movie_by = MOVIE_TOKEN
        masked_movie_num = 0
        while num<len(token_text):
            if token_text[num]=='@' and num+1<len(token_text):
                # token_text_com.append(token_text[num]+token_text[num+1])
                movie_token = token_text[num]+token_text[num+1]
                token_text_com.append(movie_token)
                num+=2
            else:
                token_text_com.append(token_text[num])
                num+=1
        movie_rec = []
        all_movie_selection_label = []
        for i in range(len(token_text_com)):
            if token_text_com[i][1:] in movies:
                movie_rec.append(token_text_com[i][1:])
                all_movie_selection_label.append(self.movieID2selection_label[int(token_text_com[i][1:])])
                token_text_com[i] = masked_movie_by
                masked_movie_num+=1
        movie_rec_trans=[]
        for movie in movie_rec:
            entity = self.id2entity[int(movie)]
            try:
                movie_rec_trans.append(self.entity2entityId[entity])
            except:
                pass
        assert masked_movie_num == len(all_movie_selection_label)
        return token_text_com,movie_rec_trans,all_movie_selection_label,masked_movie_num

    def _context_reformulate(self,context,movies,altitude,ini_altitude,s_id,re_id):
        last_id=None
        #perserve the list of dialogue
        context_list=[]
        for message in context:
            entities=[]
            try:
                for entity in self.text_dict[message['text']]:
                    try:
                        entities.append(self.entity2entityId[entity])
                    except:
                        pass
            except:
                pass

            token_text,movie_rec,all_movie_selection_label,masked_movie_num=self.detect_movie(message['text'],movies)
            # if message['senderWorkerId']==re_id  and len(context_list)>0:
            #     token_text,movie_rec,all_movie_selection_label,masked_movie_num=self.detect_movie(message['text'],movies)
                
            #     # print('processed context', u' '.join(token_text).encode('utf-8').strip())
            #     # print('movie rec:', movie_rec)
            #     # print('all_movie_selection_label :', all_movie_selection_label)
            # else:
            #     token_text,movie_rec,_,_=self.detect_movie(message['text'],movies)
            #     all_movie_selection_label=[]
            
            if len(context_list)==0:
                # context_dict={'text':token_text,'entity':entities+movie_rec,'user':message['senderWorkerId'],'movie':movie_rec}
                context_dict={'text':token_text,'entity':entities+movie_rec,'user':message['senderWorkerId'],'movie':movie_rec,'movies_selection_labels': all_movie_selection_label}
                context_list.append(context_dict)
                last_id=message['senderWorkerId']
                continue
            if message['senderWorkerId']==last_id:
                context_list[-1]['text']+=token_text
                context_list[-1]['entity']+=entities+movie_rec
                context_list[-1]['movie']+=movie_rec
                # if message['senderWorkerId']==re_id and len(context_list)>0:
                #     context_list[-1]['movies_selection_labels']+=all_movie_selection_label
                context_list[-1]['movies_selection_labels']+=all_movie_selection_label
            else:
                # context_dict = {'text': token_text, 'entity': entities+movie_rec,
                #            'user': message['senderWorkerId'], 'movie':movie_rec, 'movies_selection_labels': all_movie_selection_label}
                context_dict = {'text': token_text, 'entity': entities+movie_rec,
                           'user': message['senderWorkerId'], 'movie':movie_rec}
                # if message['senderWorkerId']==re_id and len(context_list)>0:
                #     context_dict['movies_selection_labels']=all_movie_selection_label
                context_dict['movies_selection_labels']=all_movie_selection_label
                context_list.append(context_dict)
                last_id = message['senderWorkerId']

            # if message['senderWorkerId']==re_id and len(context_list)>0:
            movie_num=context_list[-1]['text'].count(MOVIE_TOKEN)
            len_labels = len(context_list[-1]['movies_selection_labels'])
            assert movie_num == len_labels

            # if 'zombie' in ' '.join(context_list[-1]['text']) and 'focuses' in ' '.join(context_list[-1]['text']) and 'survive' in ' '.join(context_list[-1]['text']) and 'individual' in ' '.join(context_list[-1]['text']):
            #     print(u' '.join(context_list[-1]['text']).encode('utf-8').strip())
            #     print('movie_num', movie_num)
            #     # print('len(movies_selection_labels', len_labels)
            #     # print(u' '.join(context_list[-1]['text']).encode('utf-8').strip())
            #     print(context_list[-1]['movies_selection_labels'])
            #     print(context_list[-1]['movie'])
            #     # print('-----------------------------------')
            
            

        cases=[]
        contexts=[]
        entities_set=set()
        entities=[]
        for context_dict in context_list:
            self.corpus.append(context_dict['text'])
            if context_dict['user']==re_id and len(contexts)>0:
                response=context_dict['text']

                #entity_vec=np.zeros(self.entity_num)
                #for en in list(entities):
                #    entity_vec[en]=1
                #movie_vec=np.zeros(self.entity_num+1,dtype=np.float)
                if len(context_dict['movie'])!=0:
                    for movie in context_dict['movie']:
                        #if movie not in entities_set:
                        cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': movie, 'rec':1, 'all_movies':context_dict['movies_selection_labels']})
                        # cases[-1]['all_movies']=context_dict['movie']
                else:
                    cases.append({'contexts': deepcopy(contexts), 'response': response, 'entity': deepcopy(entities), 'movie': 0, 'rec':0,'all_movies':context_dict['movies_selection_labels']})

                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
            else:
                contexts.append(context_dict['text'])
                for word in context_dict['entity']:
                    if word not in entities_set:
                        entities.append(word)
                        entities_set.add(word)
        return cases

class CRSdataset(Dataset):
    def __init__(self, dataset, entity_num, concept_num):
        self.data=dataset
        self.entity_num = entity_num
        self.concept_num = concept_num+1

    def __getitem__(self, index):
        '''
        movie_vec = np.zeros(self.entity_num, dtype=np.float)
        context, c_lengths, response, r_length, entity, movie, concept_mask, dbpedia_mask, rec = self.data[index]
        for en in movie:
            movie_vec[en] = 1 / len(movie)
        return context, c_lengths, response, r_length, entity, movie_vec, concept_mask, dbpedia_mask, rec
        '''
        context, c_lengths, response, r_length, mask_response, mask_r_length, entity, movie, concept_mask, dbpedia_mask, rec, movies_gth, movies_num= self.data[index]
        entity_vec = np.zeros(self.entity_num)
        entity_vector=np.zeros(50,dtype=np.int)
        point=0
        for en in entity:
            entity_vec[en]=1
            entity_vector[point]=en
            point+=1

        concept_vec=np.zeros(self.concept_num)
        for con in concept_mask:
            if con!=0:
                concept_vec[con]=1

        db_vec=np.zeros(self.entity_num)
        for db in dbpedia_mask:
            if db!=0:
                db_vec[db]=1

        return context, c_lengths, response, r_length, mask_response, mask_r_length, entity_vec, entity_vector, movie, np.array(concept_mask), np.array(dbpedia_mask), concept_vec, db_vec, rec, movies_gth, movies_num

    def __len__(self):
        return len(self.data)

if __name__=='__main__':
    ds=dataset('data/train_data.jsonl')
    print()
