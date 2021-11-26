# DialogBERT
# Copyright 2021-present NAVER Corp.
# BSD 3-clause

import os
import random
from copy import deepcopy
import numpy as np
import tables
import json
import itertools
from tqdm import tqdm
import torch
import torch.utils.data as data
import logging
logger = logging.getLogger(__name__)

class DialogTransformerDataset(data.Dataset):
    """
    A base class for Transformer dataset
    """
    def __init__(self, file_path, tokenizer, 
                 min_num_utts=1, max_num_utts=7, max_utt_len=30, 
                 block_size=256, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False):
        # 1. Initialize file path or list of file names.
        """read training sentences(list of int array) from a hdf5 file"""
        self.tokenizer = tokenizer
        self.min_num_utts = min_num_utts #if not context_shuf and not context_masklm else 3
        self.max_num_utts = max_num_utts
        self.max_utt_len =max_utt_len
        self.block_size = block_size # segment size to train BERT. when set -1 by default, use indivicual sentences(responses) as BERT inputs.
                            # Otherwise, clip a block from the context.
        
        self.utt_masklm = utt_masklm
        self.utt_sop =utt_sop
        self.context_shuf =context_shuf
        self.context_masklm =context_masklm
        
        self.rand_utt = [tokenizer.mask_token_id]*(max_utt_len-1) + [tokenizer.sep_token_id] # update during loading
        
        # a cache to store context and response that are longer than min_num_utts
        self.cache = [[tokenizer.mask_token_id]*max_utt_len]*max_num_utts, [tokenizer.mask_token_id]*max_utt_len
        
        self.perm_list = [list(itertools.permutations(range(L))) for L in range(1, max_num_utts+1)]
        print("loading data...")
        table = tables.open_file(file_path)
        self.contexts = table.get_node('/sentences')[:].astype(np.long)
        #self.knowlege = table.get_node('/knowledge')[:].astype(np.long)
        self.index = table.get_node('/indices')[:]
        self.data_len = self.index.shape[0]
        print("{} entries".format(self.data_len))

    def __getitem__(self, offset):
        index = self.index[offset]
        pos_utt, ctx_len, res_len,  = index['pos_utt'], index['ctx_len'], index['res_len']
        #pos_knowl, knowl_len = index['pos_knowl'], index['knowl_len']
        
        ctx_len = min(ctx_len, self.block_size) if self.block_size>-1 else ctx_len# trunck too long context
        
        ctx_arr=self.contexts[pos_utt-ctx_len:pos_utt].tolist()
        res_arr=self.contexts[pos_utt:pos_utt+res_len].tolist()
        #knowl_arr = self.knowledge[pos_knowl:pos_knowl+knowl_len].tolist()
        
        ## split context array into utterances        
        context = []
        tmp_utt = []
        for i, tok in enumerate(ctx_arr):
            tmp_utt.append(ctx_arr[i])
            if tok == self.tokenizer.sep_token_id:
                floor = tmp_utt[0]
                tmp_utt = tmp_utt[1:] 
                utt_len = min(len(tmp_utt), self.max_utt_len) # floor is not counted in the utt length
                utt = tmp_utt[:utt_len]            
                context.append(utt)  # append utt to context          
                tmp_utt=[]  # reset tmp utt
        response = res_arr[1:] # ignore cls token at the begining            
        res_len = min(len(response),self.max_utt_len)
        response = response[:res_len-1] + [self.tokenizer.sep_token_id] 
        
        '''
        knowledge = knowl_arr[:]              
        knowl_len = min(len(knowledge),self.max_utt_len)
        knowledge = knowledge[:knowl_len-1] + [self.tokenizer.sep_token_id] 
        '''
        
        # balancing by removing short contexts
 #       if len(context)< self.min_num_utts:
 #           context, response = self.cache
 #       else: 
 #           self.cache = deepcopy(context), deepcopy(response)
        # end balancing
        
        num_utts = min(len(context), self.max_num_utts)
        context = context[-num_utts:]
        
        return context, response #, knowlege
    
    def list2array(self, L, d1_len, d2_len=0, d3_len=0, dtype=np.long, pad_idx=0):
        '''  convert a list to an array or matrix  '''            
        def list_dim(a):
            if type(a)!=list: return 0
            elif len(a)==0: return 1
            else: return list_dim(a[0])+1
        
        if type(L) is not list:
            print("requires a (nested) list as input")
            return None
        
        if list_dim(L)==0: return L
        elif list_dim(L) == 1:
            arr = np.zeros(d1_len, dtype=dtype)+pad_idx
            for i, v in enumerate(L): arr[i] = v
            return arr
        elif list_dim(L) == 2:
            arr = np.zeros((d2_len, d1_len), dtype=dtype)+pad_idx
            for i, row in enumerate(L):
                for j, v in enumerate(row):
                    arr[i][j] = v
            return arr
        elif list_dim(L) == 3:
            arr = np.zeros((d3_len, d2_len, d1_len), dtype=dtype)+pad_idx
            for k, group in enumerate(L):
                for i, row in enumerate(group):
                    for j, v in enumerate(row):
                        arr[k][i][j] = v
            return arr
        else:
            print('error: the list to be converted cannot have a dimenson exceeding 3')
    
    def mask_words(self, utt):
        output_label = []
        tokens = [tok for tok in utt]
        for i, token in enumerate(utt):
            prob = random.random()
            if prob < 0.15 and not token in [self.tokenizer.pad_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]:
                prob /= 0.15                
                if prob < 0.8: 
                    tokens[i] = self.tokenizer.mask_token_id   # 80% randomly change token to mask token                
                elif prob < 0.9: 
                    tokens[i] = random.randint(5, len(self.tokenizer)-5)# 10% randomly change token to random token            
                output_label.append(token)
            else:
                output_label.append(-100)
        return tokens, output_label
               
    
    def swap_utt(self, utt):
        utt_sop_label = 0 if random.random()>0.6 or len(utt)<5 else 1
        tokens = [tok for tok in utt]
        utt_len = len(tokens)
        if utt_len == self.max_utt_len: # if utt has reached the maximum length, then remove the last token because we will add a new sep token
            tokens = tokens[:-2]+ [self.tokenizer.sep_token_id]
            utt_len-=1
        sep_pos = random.randrange(2, utt_len-1) # seperate position where tokens to the right are random or coherent contexts

        # new utt
        L_utt, R_utt = tokens[1:sep_pos]+[self.tokenizer.sep_token_id], tokens[sep_pos:]
        swaped_utt = L_utt + R_utt if utt_sop_label ==0 else R_utt + L_utt
        swaped_utt = [self.tokenizer.cls_token_id] + swaped_utt
        utt_attn_mask = [1]*len(swaped_utt)
        # segment_ids                                                 
        utt_segment_ids = [0]*(sep_pos+1)+[1]*(utt_len-sep_pos) if utt_sop_label == 0 else [0]*(utt_len-sep_pos+1)+[1]*(sep_pos)       
        
        return swaped_utt, utt_attn_mask, utt_segment_ids, utt_sop_label
                    
    def mask_context(self, context):
        def is_special_utt(utt):
            return len(utt)==3 and utt[1] in [self.tokenizer.mask_token_id, self.tokenizer.sep_token_id, self.tokenizer.cls_token_id]
        
        utts = [utt for utt in context]
        lm_label = [[-100]*len(utt) for utt in context] 
        context_len = len(context)
        assert context_len>1, 'a context to be masked should have at least 2 utterances'

        mlm_probs = [0.0, 0.1, 0.4, 0.7, 0.8, 0.9, 1.0, 1.0, 1.0, 1.0]
        mlm_prob = mlm_probs[context_len-1]
        
        prob = random.random()
        if prob < mlm_prob:
            i = random.randrange(context_len)
            while is_special_utt(utts[i]): 
                i = random.randrange(context_len)
            utt = utts[i]
            prob = prob/mlm_prob
            if prob < 0.8: # 80% randomly change utt to mask utt
                utts[i] = [self.tokenizer.cls_token_id, self.tokenizer.mask_token_id, self.tokenizer.sep_token_id] 
            elif prob < 0.9: # 10% randomly change utt to a random utt  
                utts[i] = deepcopy(self.rand_utt)
            lm_label[i]= deepcopy(utt)
            #assert len(utts[i]) == len(lm_label[i]), "the size of the lm label is different to that of the masked utterance"
            self.rand_utt = deepcopy(utt) # update random utt
        return utts, lm_label
        
    def shuf_ctx(self, context):    
        perm_label = 0
        num_utts = len(context)
        if num_utts==1: 
            return context, perm_label, [0]
        for i in range(num_utts-1): perm_label += len(self.perm_list[i])
        perm_id = int(random.random()*len(self.perm_list[num_utts-1]))
        perm_label += perm_id
        ctx_position_ids = self.perm_list[num_utts-1][perm_id]
        # new context
        shuf_context = [context[i] for i in ctx_position_ids]
        return shuf_context, perm_label, ctx_position_ids

    def __len__(self):
        return self.data_len    
    

class HBertMseEuopDataset(DialogTransformerDataset):
    """
    A hierarchical Bert data loader where the context is masked with ground truth utterances and to be trained with MSE matching.
    The context is shuffled for a novel energy-based order prediction approach (EUOP)
    """
    def __init__(self, file_path, tokenizer,
                 min_num_utts=1, max_num_utts=9, max_utt_len=30, 
                 block_size=-1, utt_masklm=False, utt_sop=False, 
                 context_shuf=False, context_masklm=False):
        
        super(HBertMseEuopDataset, self).__init__(
            file_path, tokenizer, min_num_utts, max_num_utts, max_utt_len, block_size, utt_masklm, utt_sop, context_shuf, context_masklm)
        
        self.cls_utt = [tokenizer.cls_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id]
        self.sep_utt = [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.sep_token_id]


    def __getitem__(self, offset):
        context, response = super().__getitem__(offset)
                
        context_len= min(len(context), self.max_num_utts-2)
        context = [self.cls_utt] + context[-context_len:] + [self.sep_utt]
        context_len+=2
        context_attn_mask = [1]*context_len
        context_mlm_target = [[-100]*len(utt) for utt in context]
        context_position_perm_id = -100
        context_position_ids = list(range(context_len))   #               
            
        if self.context_shuf and random.random()<0.4 and len(context)>2:
            context_, context_position_perm_id, context_position_ids_ = self.shuf_ctx(context[1:-1])
            context = [self.cls_utt] + context_ + [self.sep_utt]
            context_position_ids = [0] + [p+1 for p in context_position_ids_] + [context_len-1]
            context_mlm_target = [[-100]*len(utt) for utt in context]
            
        if self.context_masklm and context_position_perm_id<2 and len(context)>4:
            context, context_mlm_target = self.mask_context(context)
        
        context_utts_attn_mask = [[1]*len(utt) for utt in context]
        
        context = self.list2array(context, self.max_utt_len, self.max_num_utts, pad_idx=self.tokenizer.pad_token_id) 
        context_utts_attn_mask = self.list2array(context_utts_attn_mask, self.max_utt_len, self.max_num_utts)
        context_attn_mask = self.list2array(context_attn_mask, self.max_num_utts)
        context_mlm_target = self.list2array(context_mlm_target, self.max_utt_len, self.max_num_utts, pad_idx=-100)
        context_position_ids = self.list2array(context_position_ids, self.max_num_utts)

        response = self.list2array(response, self.max_utt_len, pad_idx=self.tokenizer.pad_token_id) # for decoder training
        
        return context, context_utts_attn_mask, context_attn_mask, \
              context_mlm_target, context_position_perm_id, context_position_ids, response    
    
    
def load_dict(filename):
    return json.loads(open(filename, "r").readline())

def load_vecs(fin):         
    """read vectors (2D numpy array) from a hdf5 file"""
    h5f = tables.open_file(fin)
    h5vecs= h5f.root.vecs
    
    vecs=np.zeros(shape=h5vecs.shape,dtype=h5vecs.dtype)
    vecs[:]=h5vecs[:]
    h5f.close()
    return vecs

def save_vecs(vecs, fout):
    fvec = tables.open_file(fout, 'w')
    atom = tables.Atom.from_dtype(vecs.dtype)
    filters = tables.Filters(complib='blosc', complevel=5)
    ds = fvec.create_carray(fvec.root,'vecs', atom, vecs.shape,filters=filters)
    ds[:] = vecs
    print('done')
    fvec.close()
    

if __name__ == '__main__':
    
    input_dir='./data/reddit/'
    VALID_FILE=input_dir+'train.h5'
    task = 'test_ctx'#'test_utt' # 'test_ctx'
    
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    if task == 'test_utt':
        dataset=DialogTransformerDataset(VALID_FILE, tokenizer, utt_masklm=True, utt_sop=True)
    elif task == 'test_ctx':
        dataset=DialogTransformerDataset(VALID_FILE, tokenizer, context_shuf=True, context_masklm=False)
    else:
        dataset=DialogTransformerDataset(VALID_FILE, tokenizer)
    data_loader=torch.utils.data.DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
        
    if task == 'test_utt':
        k=0
        for batch in data_loader:
            response, res_bert_input, res_attn_mask, res_segment_ids, res_lm_labels, res_sop_label = batch[9:]
            k+=1
            if k>3: break
            print('response:', tokenizer.decode(response[0].numpy().tolist()))
            print(f'response:\n {response[0]}')
            print('res_bert_input:', tokenizer.decode(res_bert_input[0].numpy().tolist()))
            print(f'res_bert_input\n {res_bert_input[0]}')
            print(f'attn_mask:\n {res_attn_mask[0]}')
            print(f'segment_ids:\n {res_segment_ids[0]}')
            print(f'lm_labels:\n {res_lm_labels[0]}')
            print(f'sop_label:\n {res_sop_label[0]}')
            
    elif task == 'test_ctx':
        k=0
        for batch in data_loader:
            context, context_attn_mask, context_seg_ids, \
            context_mlm_labels, context_position_perm_id, response = batch
            
            k+=1
            if k>10: break

  #          print(f'context:\n {context}')
  #          print('context_str:', tokenizer.decode(context[0].numpy().tolist()))
  #          print(f'context_attn_mask:\n {context_attn_mask}')
  #          print(f'context_segment_ids:\n {context_seg_ids}')
  #          print(f'context_lm_labels:\n {context_mlm_labels}')
  #          print(f'context_position_perm_id:\n {context_position_perm_id}')
            #print(f'utts_segment_ids:\n {utts_segment_ids}')
            #print(f'utts_lm_labels:\n {utts_lm_labels}')
            #print(f'utts_sop_labels:\n {utts_sop_labels}')
   #         print('response:', tokenizer.decode(response[0].numpy().tolist()))
