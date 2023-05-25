import os, sys
import os.path
import torch
import logging
import ast
import glob
import copy
import random
import argparse
import re
import pathlib
import math
import datetime
import time
import ortools
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import ortools.graph.pywrapgraph as pywrapgraph
from sklearn import metrics
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
from torch import optim
from transformers import AutoModel, AutoConfig, AutoTokenizer 
from sklearn.metrics import f1_score
from accelerate.logging import get_logger


class DialogueLineEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        self.first_speaker_embeddings=nn.Embedding(2, 50)
        self.turn_embeddings=nn.Embedding(2, 50)
        self.speaker_embeddings=nn.Embedding(2, 50)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 50*3
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, f_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs

class DialogueLineEncoderA(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 10)
        self.speaker_embeddings=nn.Embedding(2, 10)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 10*3
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, f_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs


class DialogueLineEncoderB(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 10)
        self.speaker_embeddings=nn.Embedding(2, 10)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 10*2
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs

class DialogueLineEncoderC(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        self.first_speaker_embeddings=nn.Embedding(2, 150)
        self.turn_embeddings=nn.Embedding(2, 150)
        self.speaker_embeddings=nn.Embedding(2, 150)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 150*3
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, f_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs

class DialogueLineEncoderD(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 150)
        self.speaker_embeddings=nn.Embedding(2, 150)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 150*2
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs

class DialogueLineEncoderE(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 200)
        self.speaker_embeddings=nn.Embedding(2, 200)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 200*2
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs

class DialogueLineEncoderF(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 250)
        self.speaker_embeddings=nn.Embedding(2, 250)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 250*2
        # dim2=round(full_dim*0.6)
        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.seen={}
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        out=self.fc(out)
        out=self.tanh(out)

        logits=self.fc2(out).squeeze()
        labels=batch['label']

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        

        return outputs        

################################################################################
#
class MultitaskDialogueEncoder(nn.Module): # new 1228
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 250)
        self.speaker_embeddings=nn.Embedding(2, 250)

        self.full_dim=self.BERT_HIDDEN_DIM*5 + self.args["distance_embedding_size"] + 250*2
        # dim2=round(full_dim*0.6)
        # self.fc=nn.Linear(self.full_dim, self.full_dim)
        # self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc_reply=nn.Linear(self.full_dim, 1)
        self.fc_thread=nn.Linear(self.full_dim, 1)
    
    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        e_d=self.distance_embeddings(batch['utterances_distance'])

        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        
        out=self.fc(out)
        out=self.tanh(out)

        reply_logits=self.fc_reply(out).squeeze()

        if (1 in batch['mode']) or (2 in batch['mode']):
            outputs={
                    "filename_id": batch['filename_id'],
                    "candidate_line_id": batch['candidate_line_id'],
                    "utterance_of_interest_id": batch['utterance_of_interest_id'],
                    "mode": batch['mode'],
                    "logits": reply_logits,
                    "label": batch['label']
                    }        
        else:
            thread_logits=self.fc_thread(out).squeeze()
            outputs={
                    "filename_id": batch['filename_id'],
                    "candidate_line_id": batch['candidate_line_id'],
                    "utterance_of_interest_id": batch['utterance_of_interest_id'],
                    "mode": batch['mode'],
                    "logits": reply_logits,
                    "label": batch['label'],
                    "thread_logits": thread_logits,
                    "thread_label": batch['same_thread']
                    }        

        return outputs


class MultitaskDialogueEncoderWithPointer(nn.Module): # new 1228
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 250)
        self.speaker_embeddings=nn.Embedding(2, 250)

        self.full_dim=self.BERT_HIDDEN_DIM + 3072*2 + self.args["distance_embedding_size"] + 250*2
        # dim2=round(full_dim*0.6)
        # self.fc=nn.Linear(self.full_dim, self.full_dim)
        # self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc_reply=nn.Linear(self.full_dim, 1)
        self.fc_thread=nn.Linear(self.full_dim, 1)
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention=torch.matmul(x1, x2.transpose(-1, -2))
        mask1=mask1.float().masked_fill_(torch.logical_not(mask1), -np.inf)
        mask2=mask2.float().masked_fill_(torch.logical_not(mask2), -np.inf)
        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention , dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(-1, -2) , dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align
    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)
    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        q1_align, q2_align=self.soft_attention_align(lj_cls, li_cls, inputs_lj['attention_mask'], inputs_li['attention_mask'])
        q1_combined=torch.cat([lj_cls, q1_align, self.submul(lj_cls, q1_align)], 1)
        q2_combined=torch.cat([li_cls, q2_align, self.submul(li_cls, q2_align)], 1)

        e_d=self.distance_embeddings(batch['utterances_distance'])
        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        # out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        emb=torch.cat([cj_cls, q1_combined, q2_combined, e_d, t_d, k_d], 1)
        
        out=self.fc(emb)
        out=self.tanh(out)

        reply_logits=self.fc_reply(out).squeeze()

        if (1 in batch['mode']) or (2 in batch['mode']):
            outputs={
                    "filename_id": batch['filename_id'],
                    "candidate_line_id": batch['candidate_line_id'],
                    "utterance_of_interest_id": batch['utterance_of_interest_id'],
                    "mode": batch['mode'],
                    "logits": reply_logits,
                    "label": batch['label']
                    }        
        else:
            thread_logits=self.fc_thread(out).squeeze()
            outputs={
                    "filename_id": batch['filename_id'],
                    "candidate_line_id": batch['candidate_line_id'],
                    "utterance_of_interest_id": batch['utterance_of_interest_id'],
                    "mode": batch['mode'],
                    "logits": reply_logits,
                    "label": batch['label'],
                    "thread_logits": thread_logits,
                    "thread_label": batch['same_thread']
                    }        

        return outputs


################################################################################

class DialogueEncoderWithPointer(nn.Module): # new 1228
    def __init__(self, args):
        super().__init__()

        self.args=args
        self.model_name=self.args['model_name']
    
        ### Utterance Encoder
        self.utterance_encoder=AutoModel.from_pretrained(self.model_name)
        self.utterance_encoder_config=AutoConfig.from_pretrained(self.model_name)
        self.utterance_encoder_tokenizer=AutoTokenizer.from_pretrained(self.model_name, do_lower_case=False, do_basic_tokenize=False, local_files_only=True) 

        LINE_TOKEN='[LINE]'
        SELF_TOKEN='[SELF]'

        self.utterance_encoder_tokenizer.add_tokens([LINE_TOKEN, SELF_TOKEN], special_tokens=True)
        self.utterance_encoder.resize_token_embeddings(len(self.utterance_encoder_tokenizer))

        self.BERT_HIDDEN_DIM=self.utterance_encoder_config.hidden_size
        self.SEQUENCE_MAX_LEN=self.utterance_encoder_tokenizer.model_max_length
    
        self.fix_encoder=self.args["fix_encoder"]
        
        if self.fix_encoder:
            for p in self.utterance_encoder.parameters():
                p.requires_grad=False

        self.distance_embeddings=nn.Embedding(args["distance_embedding_dim"], self.args["distance_embedding_size"]) #self.BERT_HIDDEN_DIM

        # self.first_speaker_embeddings=nn.Embedding(2, 10)
        self.turn_embeddings=nn.Embedding(2, 250)
        self.speaker_embeddings=nn.Embedding(2, 250)

        self.full_dim=self.BERT_HIDDEN_DIM + 3072*2 + self.args["distance_embedding_size"] + 250*2
        # dim2=round(full_dim*0.6)
        # self.fc=nn.Linear(self.full_dim, self.full_dim)
        # self.fc2=nn.Linear(self.full_dim, 1)
        self.tanh=nn.Tanh()

        self.fc=nn.Linear(self.full_dim, self.full_dim)
        self.fc_2=nn.Linear(self.full_dim, 1)
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention=torch.matmul(x1, x2.transpose(-1, -2))
        mask1=mask1.float().masked_fill_(torch.logical_not(mask1), -np.inf)
        mask2=mask2.float().masked_fill_(torch.logical_not(mask2), -np.inf)
        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention , dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(-1, -2) , dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size
        return x1_align, x2_align
    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)
    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, batch):        
        inputs_cj={"input_ids": batch["context"],
                   "attention_mask": (batch["context"] > 0).long()}
        inputs_lj={"input_ids": batch["parent_utterance"],
                   "attention_mask": (batch["parent_utterance"] > 0).long()}
        inputs_li={"input_ids": batch["utterance_of_interest"],
                   "attention_mask": (batch["utterance_of_interest"] > 0).long()}        

        cj_cls=self.utterance_encoder(**inputs_cj)['last_hidden_state'][:, 0, :]
        lj_cls=self.utterance_encoder(**inputs_lj)['last_hidden_state'][:, 0, :]
        li_cls=self.utterance_encoder(**inputs_li)['last_hidden_state'][:, 0, :]

        q1_align, q2_align=self.soft_attention_align(lj_cls, li_cls, inputs_lj['attention_mask'], inputs_li['attention_mask'])
        q1_combined=torch.cat([lj_cls, q1_align, self.submul(lj_cls, q1_align)], 1)
        q2_combined=torch.cat([li_cls, q2_align, self.submul(li_cls, q2_align)], 1)

        e_d=self.distance_embeddings(batch['utterances_distance'])
        t_d=self.turn_embeddings(batch['same_turn'])
        # f_d=self.first_speaker_embeddings(batch['first_spoke'])
        k_d=self.speaker_embeddings(batch['same_speaker'])

        # out=torch.cat([cj_cls, lj_cls, li_cls, li_cls-lj_cls, li_cls*lj_cls, e_d, t_d, k_d], 1)
        emb=torch.cat([cj_cls, q1_combined, q2_combined, e_d, t_d, k_d], 1)
        
        out=self.fc(emb)
        out=self.tanh(out)        

        logits=self.fc_2(out).squeeze()

        outputs={"filename_id": batch['filename_id'],
                 "logits": logits,
                 "candidate_line_id": batch['candidate_line_id'],
                 "utterance_of_interest_id": batch['utterance_of_interest_id'],
                 'mode': batch['mode'],
                 "label": batch['label']}        


        return outputs

################################################################################

class LogisticRegression(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
#
#         self.input_size=input_size
#         self.label_size=label_size
        self.linear=torch.nn.Linear(8, 1)
    def forward(self, x):
        logits=self.linear(x[:, 3:-1])#
#         print(x[:, 0])
        
        outputs={
            "filename_id": x[:, 0].long(),
             "candidate_line_id": x[:, 1].long(),
             "utterance_of_interest_id": x[:, 2].long(),
             "mode": x[:, -1].long(),
             "logits": logits.squeeze()}       
#         print(outputs)
        return outputs 
