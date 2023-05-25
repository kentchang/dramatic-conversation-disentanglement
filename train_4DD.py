########
# Contains code from the original implementation https://github.com/xbmxb/StructureCharacterization4DD
########
from __future__ import absolute_import, division, print_function
import os.path
import ast
import copy
import ortools
import ortools.graph.pywrapgraph as pywrapgraph
from sklearn import metrics
from tqdm import tqdm
from models import *
from eval import *
import re
import os
import sys
import csv
import json
import glob
import math
import time
import pickle
import random
import logging
import pathlib
import datasets
import datetime
import argparse
import transformers
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score
from pprint import pprint, pformat
from tqdm import tqdm, trange
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from transformers import (BertConfig, BertTokenizer, BertModel, BertPreTrainedModel,
                          ElectraConfig, ElectraTokenizer, ElectraModel, ElectraPreTrainedModel,
                          RobertaConfig, RobertaTokenizer, RobertaModel,
                          AdamW, WEIGHTS_NAME, CONFIG_NAME)

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger

BertLayerNorm = torch.nn.LayerNorm

class MHA(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3) # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        
        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return (layernormed_context_layer, attention_probs) if output_attentions else (layernormed_context_layer,)

class MyLSTM(nn.Module):
    def __init__(self, input_sz, hidden_sz, g_sz):
        super(MyLSTM, self).__init__()
        self.input_sz = input_sz
        self.hidden_sz = hidden_sz
        self.g_sz = g_sz
        self.all1 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1),  self.hidden_sz)
        self.all2 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all3 = nn.Linear((self.hidden_sz * 1 + self.input_sz  +self.g_sz), self.hidden_sz)
        self.all4 = nn.Linear((self.hidden_sz * 1 + self.input_sz  * 1), self.hidden_sz)

        self.all11 = nn.Linear((self.hidden_sz * 1 + self.g_sz),  self.hidden_sz)
        self.all44 = nn.Linear((self.hidden_sz * 1 + self.g_sz), self.hidden_sz)

        self.init_weights()
        self.drop = nn.Dropout(0.5)
    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_sz)
        for weight in self.parameters():
            nn.init.uniform_(weight, -stdv, stdv)

    def node_forward(self, xt, ht, Ct_x, mt, Ct_m):

        # # # new standard lstm
        hx_concat = torch.cat((ht, xt), dim=1)
        hm_concat = torch.cat((ht, mt), dim=1)
        hxm_concat = torch.cat((ht, xt, mt), dim=1)


        i = self.all1(hx_concat)
        o = self.all2(hxm_concat)
        f = self.all3(hxm_concat)
        u = self.all4(hx_concat)
        ii = self.all11(hm_concat)
        uu = self.all44(hm_concat)

        i, f, o, u = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(u)
        ii,uu = torch.sigmoid(ii), torch.tanh(uu)
        Ct_x = i * u + ii * uu + f * Ct_x
        ht = o * torch.tanh(Ct_x) 

        return ht, Ct_x, Ct_m 

    def forward(self, x, m, init_stat=None):
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        cell_seq = []
        if init_stat is None:
            ht = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_x = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
            Ct_m = torch.zeros((batch_sz, self.hidden_sz)).to(x.device)
        else:
            ht, Ct = init_stat
        for t in range(seq_sz):  # iterate over the time steps
            xt = x[:, t, :]
            mt = m[:, t, :]
            ht, Ct_x, Ct_m= self.node_forward(xt, ht, Ct_x, mt, Ct_m)
            hidden_seq.append(ht)
            cell_seq.append(Ct_x)
            if t == 0:
                mht = ht
                mct = Ct_x
            else:
                mht = torch.max(torch.stack(hidden_seq), dim=0)[0]
                mct = torch.max(torch.stack(cell_seq), dim=0)[0]
        hidden_seq = torch.stack(hidden_seq).permute(1, 0, 2) ##batch_size x max_len x hidden
        return hidden_seq


class Bert_v7(BertPreTrainedModel):
    def __init__(self, config, lstm_hidden_size=128, lstm_num_layers=2, gcn_layer=1, mylstm_hidden_size=128, num_decoupling=1):
        super().__init__(config)
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_decoupling = num_decoupling
        # self.graph_dim = graph_dim
        self.gcn_layer = gcn_layer
        self.mylstm_hidden_size = mylstm_hidden_size

        self.bert = BertModel(config)
        self.BiLSTM = nn.LSTM(config.hidden_size, self.lstm_hidden_size, self.lstm_num_layers, bias=True, batch_first=True, bidirectional=True)
        self.graph_dim = config.hidden_size
        self.lstm_f = MyLSTM(config.hidden_size, self.mylstm_hidden_size, self.graph_dim) 
        self.lstm_b = MyLSTM(config.hidden_size, self.mylstm_hidden_size, self.graph_dim)
        self.drop_lstm = nn.Dropout(config.hidden_dropout_prob)

        self.pooler = nn.Linear(8*self.mylstm_hidden_size, 4*self.mylstm_hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(4*self.mylstm_hidden_size, 1)
        # self.classifier2 = nn.Linear(config.hidden_size, 2)
        self.SASelfMHA = nn.ModuleList([MHA(config) for _ in range(num_decoupling)])
        self.linear = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.W = nn.ModuleList()
        for layer in range(self.gcn_layer):
            self.W.append(nn.Linear(self.graph_dim, self.graph_dim))

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        attention_mask=None,
        sep_pos=None,
        position_ids=None,
        turn_ids = None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        adj_matrix_speaker=None,
        adj_matrix_scene=None,
        filename_ids=None,
        utterance_of_interest_ids=None,
        candidate_ids_nested=None,
        true_parent_ids=None
    ):
        num_labels = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None 
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        orig_attention_mask = attention_mask
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None        
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )
        outputs = self.bert(
            input_ids,
            attention_mask=orig_attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = outputs[0] # (batch_size * num_choice, seq_len, hidden_size)
        cls_rep = sequence_output[:,0,:] #(batch_size*num_chioce, hidden_size)
        hidden_size = sequence_output.size(-1)
        cls_rep = cls_rep.view(-1, num_labels, hidden_size) #(batch_size, num_chioce, hidden_size)
        adj_matrix_speaker = adj_matrix_speaker.unsqueeze(1)
        sa_self_mask = (1.0 - adj_matrix_speaker) * -10000.0
        sa_self_ = self.SASelfMHA[0](cls_rep, cls_rep, attention_mask = sa_self_mask)[0]
        for t in range(1, self.num_decoupling):
            sa_self_ = self.SASelfMHA[t](sa_self_, sa_self_, attention_mask = sa_self_mask)[0]
        with_sa_self = self.linear(torch.cat((cls_rep,sa_self_),2))#(batch_size, num_chioce, hidden_size)

        adj_matrix_scene = adj_matrix_scene.unsqueeze(1)
        sa_self_mask = (1.0 - adj_matrix_scene) * -10000.0
        for t in range(1, self.num_decoupling):
            sa_self_ = self.SASelfMHA[t](sa_self_, sa_self_, attention_mask = sa_self_mask)[0]
        with_sa_self = self.linear(torch.cat((cls_rep,sa_self_),2))
        batch_size, sent_len, input_dim = cls_rep.size()#(batch_size, num_chioce, 2*lstm_hidden_size)
        graph_input = with_sa_self[:, :, :self.graph_dim]#(batch_size, num_chioce, 2*lstm_hidden_size)
        # forward LSTM
        lstm_out_f = self.lstm_f(cls_rep, graph_input)#(batch_size, num_chioce, mylstm_hidden_size)
        # backward LSTM
        cls_rep_b = torch.flip(cls_rep, [1])
        graph_input_b = torch.flip(graph_input, [1])
        lstm_out_b = self.lstm_b(cls_rep_b, graph_input_b)#(batch_size, num_chioce, mylstm_hidden_size)
        lstm_out_b = torch.flip(lstm_out_b, [1])

        lstm_output = torch.cat((lstm_out_f, lstm_out_b), dim=2)#(batch_size, num_chioce, 2*mylstm_hidden_size)
        lstm_output = self.drop_lstm(lstm_output)

        target = lstm_output[:,0,:] #(batch_size, 2*mylstm_hidden_size)
        final_lstm_output = lstm_output.repeat(1,1,4) #(batch_size, num_chioce, 2*mylstm_hidden_size *4)
        for i in range(final_lstm_output.size(0)):
            for j in range(final_lstm_output.size(1)):
                a = lstm_output[i,j,:]
                b = target[i,:]
                c = a * b
                d = a - b
                final_lstm_output[i,j,:] = torch.cat((a,b,c,d)) #(2*mylstm_hidden_size *4)

        pooled_output = self.pooler_activation(self.pooler(final_lstm_output)) #(batch_size, num_chioce, 4*mylstm_hidden_size )
        pooled_output = self.dropout(pooled_output)
        
        # if num_labels > 2:
        logits = self.classifier(pooled_output) #(batch_size, num_chioce, 1)
        reshaped_logits = logits.squeeze(2) #(batch_size, num_chioce)
        logits = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        outputs={
            "filename_ids": filename_ids,
            "logits": logits[0],
            "utterance_of_interest_ids": utterance_of_interest_ids,
            "candidate_ids_nested": candidate_ids_nested,
            "true_parent_ids": true_parent_ids,
            "labels": labels
        }
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            loss=loss_fct(reshaped_logits, labels)        
            outputs["loss"]=loss
            
        return outputs

def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    os.environ["PYTHONHASHSEED"]=str(seed)
    main_log(f"Random seed set as {seed}")

def to_cuda(x):
    if torch.cuda.is_available(): x=x.cuda()
    return x

def main_log(msg):
    global logger
    return logger.info(msg, main_process_only=True)

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, 
                 guid, 
                 utterance_id,                 
                 text_a, 
                 candidate_ids,
                 text_b=None,
                 true_parent_id=None,
                 label=None, adj_matrix_speaker=None, adj_matrix_scene=None):

        self.guid=guid # filename id 
        self.text_a=text_a
        self.text_b=text_b
        self.utterance_id=utterance_id
        self.true_parent_id=true_parent_id
        self.candidate_ids=candidate_ids
        self.label=label
        self.adj_matrix_speaker=adj_matrix_speaker
        self.adj_matrix_scene=adj_matrix_scene


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, example_id, choices_features, utterance_id, candidate_ids, true_parent_id, label, adj_matrix_speaker, adj_matrix_scene):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'sep_pos': sep_pos,
                'turn_ids': turn_ids
            }
            for input_ids, input_mask, segment_ids, sep_pos, turn_ids in choices_features
        ]
        self.utterance_id=utterance_id
        self.true_parent_id=true_parent_id
        self.candidate_ids=candidate_ids
        self.label=label
        self.adj_matrix_speaker=adj_matrix_speaker
        self.adj_matrix_scene=adj_matrix_scene

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""
    def get_examples(self, data_dir, mode):
        """Gets a collection of `InputExample`s for the train/dev/test set.""" 
        raise NotImplementedError()

    def get_labels(self, max_previous_utterance):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

class DCDProcessor(DataProcessor):

    def _get_examples(self, mode, lines_dict):
        
        examples={}
        
        for key, line_data in lines_dict[mode].items():   
            """
            use dictionary
            find all dialogue line ids in a file (don't just concate plain texts)
            find the true parent dialogue line ID (can be itself) 
                -> iterate possible candidate ids -> if matches set idx to label
            """

            filename_id, utterance_id=key
            if filename_id not in examples:
                examples[filename_id]=[]
             
            if line_data['reply_to_id'].startswith('T'):
                true_parent_utterance_id=utterance_id
            if line_data['reply_to_id'].startswith('D'):
                true_parent_utterance_id=line_data['reply_to_id']
                
            examples[filename_id].append((true_parent_utterance_id, utterance_id))
            
        return examples
    
    def get_examples(self, tokenizer, mode, 
                     reversed_filename_to_filename_id, line_id2line_text, line_id2speaker_n, scene_id2line_ids, line_id2scene_id, 
                     max_previous_utterance, lines_dict):        
        
        start_token=tokenizer.cls_token  
        sep_token=tokenizer.sep_token
        PAD_UTTERANCE_ID='D99999'
    
        filenames=[]
        reshaped_examples=[]
        
        examples=self._get_examples(mode, lines_dict)
    
        for filename_id, info_tuples in examples.items():
            filename=reversed_filename_to_filename_id[filename_id]

            if filename not in filenames:
                filenames.append(filename)
                        
            for info_tuple_idx, info_tuple in enumerate(info_tuples):
                seen_cands=[]
                true_parent_utterance_id, utterance_id=info_tuple
                
                text_a=[]
                text_b=[]
                candidate_ids=[]
                adj_matrix_speaker=[[0]*max_previous_utterance]*max_previous_utterance
                adj_matrix_scene=[[0]*max_previous_utterance]*max_previous_utterance
                
                true_parent_id, uoi_id=info_tuple#[1]
                uoi_text=line_id2line_text[mode][filename_id][uoi_id]
                uoi_scene=line_id2scene_id[mode][filename_id][uoi_id]
                uoi_speaker=line_id2speaker_n[mode][filename_id][uoi_id]
    
                for j in range(0, max_previous_utterance):
                    i=info_tuple_idx % max_previous_utterance
                    diff=info_tuple_idx - j

                    if diff > len(info_tuples):
                        text_b.append('')
                        text_a.append(uoi_text)
                        # artificial candidate doesn't get added to seen_cands so won't interfere with true D0's
                        candidate_ids.append(PAD_UTTERANCE_ID) 

                        adj_matrix_speaker[i][j]=0
                        adj_matrix_speaker[j][i]=0                        
                        adj_matrix_scene[i][j]=0
                        adj_matrix_scene[j][i]=0     
                        continue

                    if diff < 0:
                        text_b.append('')
                        text_a.append(uoi_text)
                        candidate_ids.append(PAD_UTTERANCE_ID)

                        adj_matrix_speaker[i][j]=0
                        adj_matrix_speaker[j][i]=0                        
                        adj_matrix_scene[i][j]=0
                        adj_matrix_scene[j][i]=0     
                        continue                            

                    text_b_utterance_id=info_tuples[diff][1]
                    candidate_ids.append(text_b_utterance_id)

                    if text_b_utterance_id in seen_cands:
                        # an uoi should not have the same candidate (we're operating on the candidate level now)
                        continue

                    text_b.append(line_id2line_text[mode][filename_id][text_b_utterance_id])
                    text_a.append(uoi_text)
                    seen_cands.append(text_b_utterance_id)

                    if true_parent_utterance_id == text_b_utterance_id:
                        label=j

                    scene_b=line_id2scene_id[mode][filename_id][text_b_utterance_id]
                    speaker_b=line_id2speaker_n[mode][filename_id][text_b_utterance_id]                        

                    if uoi_speaker == speaker_b:
                        adj_matrix_speaker[i][j]=1
                        adj_matrix_speaker[j][i]=1                        

                    if uoi_scene == scene_b:
                        adj_matrix_scene[i][j]=1
                        adj_matrix_scene[j][i]=1         
                        
                reshaped_examples.append(
                    InputExample(
                        guid=filename_id,#id,
                        text_a=text_a,
                        text_b=text_b,
                        utterance_id=int(utterance_id[1:]),
                        candidate_ids=[int(candidate_id[1:]) for candidate_id in candidate_ids],
                        true_parent_id=int(true_parent_id[1:]),                        
                        label=label,
                        adj_matrix_speaker=adj_matrix_speaker,
                        adj_matrix_scene=adj_matrix_scene
                    ) 
                )
        return reshaped_examples, filenames
    
    def get_labels(self, max_previous_utterance):
        return [i for i in range(max_previous_utterance)]

def convert_examples_to_features(examples, label_list, max_seq_length, max_utterance_num,
                                 tokenizer):

    label_map={label: i for i, label in enumerate(label_list)}
    cls_token=tokenizer.cls_token  
    sep_token=tokenizer.sep_token
        
    features=[]
    
    for (ex_index, example) in enumerate(examples):
        choices_features=[]
        all_tokens=[]

        for ending_idx, (text_a, text_b) in enumerate(zip(example.text_a, example.text_b)):
            tokens_a=tokenizer.tokenize(text_a)
            tokens_b=tokenizer.tokenize(text_b)
            tokens_b=tokens_b + [sep_token]
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 2)

            tokens = [cls_token]
            turn_ids = [0]

            context_len = []
            sep_pos = []

            tokens_b_raw = " ".join(tokens_b)
            tokens_b = []
            current_pos = 0
            
            for toks in tokens_b_raw.split(sep_token)[-max_utterance_num - 1:-1]:
                context_len.append(len(toks.split()) + 1)
                tokens_b.extend(toks.split())
                tokens_b.extend([sep_token])
                current_pos += context_len[-1]
                turn_ids += [len(sep_pos)] * context_len[-1]
                sep_pos.append(current_pos)
                
            tokens += tokens_b #cls b sep a sep

            segment_ids = [0] * (len(tokens))

            tokens_a += [sep_token]
            tokens += tokens_a            
            segment_ids += [1] * (len(tokens_a))
            
            turn_ids += [len(sep_pos)] * len(tokens_a) 
            sep_pos.append(len(tokens) - 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            input_mask = [1] * len(input_ids)

            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding
            turn_ids += padding

            context_len += [-1] * (max_utterance_num - len(context_len))
            sep_pos += [0] * (max_utterance_num + 1 - len(sep_pos))

            assert len(sep_pos) == max_utterance_num + 1
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(context_len) == max_utterance_num 
            assert len(turn_ids) == max_seq_length 

            choices_features.append((input_ids, input_mask, segment_ids, sep_pos, turn_ids))
            all_tokens.append(tokens)

        if example.label!=None:
            label_id=label_map[example.label]
        else:
            label_id=None

        features.append(
            InputFeatures(
                example_id = example.guid, 
                choices_features = choices_features,
                utterance_id=example.utterance_id, 
                candidate_ids=example.candidate_ids, 
                true_parent_id=example.true_parent_id,
                label=label_id,
                adj_matrix_speaker=example.adj_matrix_speaker,
                adj_matrix_scene=example.adj_matrix_scene
                )
        )

    return features
            
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop(0)

def prep_tensor_data(features):
    all_input_ids=torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask=torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids=torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_adj_speaker=torch.tensor([f.adj_matrix_speaker for f in features], dtype=torch.long)
    all_adj_matrix_scene=torch.tensor([f.adj_matrix_scene for f in features], dtype=torch.long)
    all_guid=torch.tensor([f.example_id for f in features], dtype=torch.long)
    all_utterance_ids=torch.tensor([f.utterance_id for f in features], dtype=torch.long)
    all_candidate_ids_nested=torch.tensor([f.candidate_ids for f in features], dtype=torch.long)

    try:
        all_true_parent_ids=torch.tensor([f.true_parent_id for f in features], dtype=torch.long)
        all_label_ids=torch.tensor([f.label for f in features], dtype=torch.long)
        return all_input_ids, all_input_mask, all_segment_ids, all_adj_speaker, all_adj_matrix_scene, all_guid, all_utterance_ids, all_candidate_ids_nested, all_true_parent_ids, all_label_ids
    except:
        return all_input_ids, all_input_mask, all_segment_ids, all_adj_speaker, all_adj_matrix_scene, all_guid, all_utterance_ids, all_candidate_ids_nested

if __name__=='__main__':

    arg_parser=argparse.ArgumentParser()
    arg_parser.add_argument('--encoder_name', help='specify encoder_name')
    arg_parser.add_argument('--epochs', help='specify epochs', default=8)    
    arg_parser.add_argument('--train_file', help='specify train_file')
    arg_parser.add_argument('--dev_file', help='specify dev_file')
    arg_parser.add_argument('--model_name', help='specify model_name')
    arg_parser.add_argument("--max_previous_utterance",
                        default=50,
                        type=int,
                        help="The maximum of previous utterances considerated.")
    arg_parser.add_argument('--model_output', help='specify model_output')
    arg_parser.add_argument('--log_output', help='specify log_output')

    arg_parser.add_argument("--use_tqdm",
                        default=False,
                        type=bool,
                        help="Use tqdm?")

    arg_parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="specific batch_size.")


    args=vars(arg_parser.parse_args())

    ROOT=pathlib.Path('/global/scratch/users/kentkchang/dramatic-bert')
    MODEL_PATH=ROOT / 'model'
    DATA_PATH=ROOT / 'model' / 'data'
    OUTPUT_PATH=MODEL_PATH / args['model_output']
    OUTPUT_PATH.mkdir(exist_ok=True)
    LOG_PATH=MODEL_PATH / args['log_output']
    LOG_PATH.mkdir(exist_ok=True)

    args['epochs']=int(args['epochs'])
    args["n_gpu"]=torch.cuda.device_count()
    args["learning_rate"]=5e-6
    args['grad_clip']=1
    args["fix_encoder"]=0
    args["output_dir"]=OUTPUT_PATH
    args["logs_dir"]=str(LOG_PATH)

    use_tqdm=args['use_tqdm']
    BATCH_SIZE=args['batch_size']

    ######

    ddp_kwargs=DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator=Accelerator(kwargs_handlers=[ddp_kwargs])

    ######
    timestamp=datetime.datetime.now().strftime("%m%d%Y-%H%M%S")    
    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=os.path.join(args["logs_dir"], f"train_{timestamp}.log"),
                    filemode='w')
    console=logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter=logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()

    logger=get_logger(__name__)

    set_seed(2022)#int(datetime.datetime.now().strftime("%Y%m%d")))
    accelerator.wait_for_everyone()    
    ##
    main_log('v5-4DD')

    RAW_TRAIN_FILE=(DATA_PATH).joinpath(args['train_file'])#
    main_log(args['train_file'])
    RAW_DEV_FILE=(DATA_PATH).joinpath(args['dev_file'])#

    main_log('Analyzing files ...')
    train_titles=[]
    with open(RAW_TRAIN_FILE, 'r') as a:
        for line in a:
            corpus, slug, title, line_idx, line_type, scene_id, turn_id, line_id, speaker_raw, speaker_label, scene_speaker_id, anno, line_text=line.split('\t')
            if anno.startswith('T'):
                if title not in train_titles:
                    train_titles.append(title)

    dev_titles=[]                
    with open(RAW_DEV_FILE, 'r') as a:
        for line in a:
            corpus, slug, title, line_idx, line_type, scene_id, turn_id, line_id, speaker_raw, speaker_label, scene_speaker_id, anno, line_text=line.split('\t')
            if anno.startswith('T'):
                if title not in dev_titles:
                    dev_titles.append(title)

                    
    files=[RAW_TRAIN_FILE, RAW_DEV_FILE]

    lines={'train': {}, 'dev': {}, 'test': {}}
    counts={'train': 0, 'dev': 0, 'test': 0}
    scene_id2line_ids={'train': {}, 'dev': {}, 'test': {}}
    line_id2scene_id={'train': {}, 'dev': {}, 'test': {}}
    line_id2line_text={'train': {}, 'dev': {}, 'test': {}}
    line_id2turn_n={'train': {}, 'dev': {}, 'test': {}}
    line_id2speaker_n={'train': {}, 'dev': {}, 'test': {}}
    filename_to_filename_id={}
    last_scene_id=''
    last_turn_line_no=''
    scene_line_ids=[]

    for file in files:
        with open(file, 'r') as f:
            next(f)
            for line in f:
                line=line.replace('\n', '')
                try:
                    category, filename, title, file_line_no, turn_line_no, scene_id, line_type, line_no, new_line_no, speaker_label, scene_speaker_id, anno, line_text=line.split('\t')
                except:
                    print(line)
                    print(len(line.split('\t')))

                if filename not in filename_to_filename_id:
                    filename_to_filename_id[filename]=len(filename_to_filename_id)

                filename_id=filename_to_filename_id[filename]

                if title in train_titles:
                    mode='train'
                if title in dev_titles:
                    mode='dev'

                if (new_line_no.startswith('D')) and anno:  
                    lines[mode][(filename_id, new_line_no)]={
                        'corpus': category,
                        'title': title,
                        'scene_id': scene_id,
                        'scene_speaker_id': scene_speaker_id,
                        'turn_line_no': turn_line_no,
                        'reply_to_id': anno
                    }
                    last_turn_line_no=turn_line_no

                if filename_id not in line_id2line_text[mode]:
                    line_id2line_text[mode][filename_id]={}             
                line_id2line_text[mode][filename_id][new_line_no]=f"{line_text}"

                if filename_id not in scene_id2line_ids[mode]:
                    scene_id2line_ids[mode][filename_id]={}                         
                if scene_id not in scene_id2line_ids[mode][filename_id]:
                    scene_id2line_ids[mode][filename_id][scene_id]=[]
                if new_line_no.startswith('A') or new_line_no.startswith('D'):
                    scene_id2line_ids[mode][filename_id][scene_id].append(new_line_no)

                if filename_id not in line_id2scene_id[mode]:
                    line_id2scene_id[mode][filename_id]={}                         
                if new_line_no.startswith('D'):
                    line_id2scene_id[mode][filename_id][new_line_no]=scene_id                
                    
                if filename_id not in line_id2turn_n[mode]:
                    line_id2turn_n[mode][filename_id]={}   
                if new_line_no.startswith('D'):
                    line_id2turn_n[mode][filename_id][new_line_no]=turn_line_no

                if filename_id not in line_id2speaker_n[mode]:
                    line_id2speaker_n[mode][filename_id]={}   
                if new_line_no.startswith('D'):
                    line_id2speaker_n[mode][filename_id][new_line_no]=scene_speaker_id

    reversed_filename_to_filename_id={v: k for k, v in filename_to_filename_id.items()}



    #############################################################################

    pretrained_model_name=args['model_name']
    max_previous_utterance=args['max_previous_utterance']

    main_log(f'Initiating the PrLM: {pretrained_model_name}')

    config_class, model_class, tokenizer_class=BertConfig, Bert_v7, BertTokenizer
    tokenizer=tokenizer_class.from_pretrained(pretrained_model_name, 
                                            do_lower_case=False, 
                                            do_basic_tokenize=False)

    processor=DCDProcessor()

    train_examples, train_filenames=\
        processor.get_examples(tokenizer, 'train',
                               reversed_filename_to_filename_id,
                               line_id2line_text, 
                               line_id2speaker_n, 
                               scene_id2line_ids, 
                               line_id2scene_id,
                               max_previous_utterance, 
                               lines)

    label_list=processor.get_labels(max_previous_utterance)
    num_labels=len(label_list)

    config=config_class.from_pretrained(pretrained_model_name, num_labels=num_labels)
    model=model_class.from_pretrained(pretrained_model_name, config=config)

    HIDDEN_DIM=config.hidden_size
    SEQUENCE_MAX_LEN=tokenizer.model_max_length

    train_features=convert_examples_to_features(train_examples, label_list, SEQUENCE_MAX_LEN, max_previous_utterance, tokenizer)
    train_data=TensorDataset(*prep_tensor_data(train_features))
    train_sampler=RandomSampler(train_data)
    data_loader=DataLoader(train_data, sampler=train_sampler, batch_size=BATCH_SIZE)
    #######
    main_log('Analyzing dev files ...')
    gold_threads={}
    gold_reply_to_dict={}
    dev_lines=[]
    with open(RAW_DEV_FILE, 'r') as f:
        next(f)
        for line in f:
            line=line.replace('\n', '')
            category, filename, title, file_line_no, turn_line_no, scene_id, line_type, line_no, new_line_no, speaker_label, scene_speaker_id, anno, line_text=line.split('\t')

            if title not in dev_titles:
                continue    
            if not new_line_no.startswith('D'):
                continue

            if filename not in gold_reply_to_dict:
                gold_reply_to_dict[filename]={}
                
            gold_reply_to_dict[filename][new_line_no]=anno
            
            if anno.startswith('T'):
                if filename not in gold_threads:
                    gold_threads[filename]={}
                if anno not in gold_threads[filename]:
                    gold_threads[filename][anno]=[]
                dev_lines.append([filename, new_line_no, anno])
                continue
                
            dev_lines.append([filename, new_line_no, anno])

    gold, _=eval_lines_dict_to_clusters(eval_lines_to_lines_dict(dev_lines))

    ####### 


    dev_examples, dev_filenames=\
        processor.get_examples(tokenizer, 'dev',
                               reversed_filename_to_filename_id,
                               line_id2line_text, 
                               line_id2speaker_n, 
                               scene_id2line_ids, 
                               line_id2scene_id,
                               max_previous_utterance, 
                               lines)
    dev_data=TensorDataset(*prep_tensor_data(convert_examples_to_features(dev_examples, label_list, SEQUENCE_MAX_LEN, max_previous_utterance, tokenizer)))
    dev_sampler=SequentialSampler(dev_data)
    dev_data_loader=DataLoader(dev_data, sampler=dev_sampler, batch_size=BATCH_SIZE)

    ### OPTIMIZER
    param_optimizer=list(model.named_parameters())
    no_decay=['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer=optim.AdamW(optimizer_grouped_parameters, lr=args["learning_rate"])
    # criterion=nn.BCEWithLogitsLoss()

    model, optimizer, data_loader, dev_data_loader=accelerator.prepare(
        model, optimizer, data_loader, dev_data_loader
    )

    seen={}

    best_eval_metric=0.

    all_predictions_made, correct_predictions_made=0, 0

    num_steps=args["epochs"]*len(data_loader)

    if use_tqdm:
        progress_bar=tqdm(num_steps)

    main_log('Training starts ...')
    timestamp=datetime.datetime.now().strftime("%m%d%Y-%H%M%S")    
    # best_model_path=f"pytorch_model-{timestamp}.bin"
    for epoch in range(args["epochs"]):
        main_log("Epoch:{}".format(epoch+1)) 
        train_loss = 0

        # pbar=tqdm(data_loader)
        for i, batch in enumerate(data_loader):#
            model.train()
            d={
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2], # if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                'adj_matrix_speaker': batch[3],
                'adj_matrix_scene': batch[4],
                'filename_ids': batch[5],
                'utterance_of_interest_ids': batch[6],
                'candidate_ids_nested': batch[7],
                'true_parent_ids': batch[8],       
                'labels': batch[9]
            }

            outputs=model(**d)
            loss=outputs['loss'].mean()
            train_loss += loss.detach().item()
            accelerator.backward(loss)    
            optimizer.step()
            optimizer.zero_grad()

            if use_tqdm:
                progress_bar.update(1)
                progress_bar.set_description("training Loss: {:.4f}".format(train_loss/(i+1))) 

            if (i == len(data_loader)-1):
                model.eval()

                pred_correct, pred_total=0., 0.
                preds_dict={}
                
                y, y_preds=[], []
                
                preds=None
                pred_lines=[]

                for idx, batch in enumerate(dev_data_loader):

                    d={
                        'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2], # if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                        'adj_matrix_speaker': batch[3],
                        'adj_matrix_scene': batch[4],
                        'filename_ids': batch[5],
                        'utterance_of_interest_ids': batch[6],
                        'candidate_ids_nested': batch[7],
                        'true_parent_ids': batch[8],     
                        'labels': batch[9]
                    }

                    d={key: to_cuda(val) for key, val in d.items()} 
                    
                    with torch.no_grad():        
                        outputs=model(**d)
                        outputs=accelerator.gather(outputs)
                        
                        if preds is None:
                            preds=outputs['logits'].detach().cpu().numpy()
                            out_label_ids=outputs['labels'].detach().cpu().numpy()
                            filename_ids=outputs['filename_ids'].detach().cpu().numpy()
                            utterance_of_interest_ids=outputs['utterance_of_interest_ids'].detach().cpu().numpy()
                            candidate_ids_nested=outputs['candidate_ids_nested'].detach().cpu().numpy()
                            true_parent_ids=outputs['true_parent_ids'].detach().cpu().numpy()
                        else:
                            preds=np.append(preds, outputs['logits'].detach().cpu().numpy(), axis=0)
                            out_label_ids=np.append(out_label_ids, outputs['labels'].detach().cpu().numpy(), axis=0)
                            filename_ids=np.append(filename_ids, outputs['filename_ids'].detach().cpu().numpy(), axis=0)
                            utterance_of_interest_ids=np.append(utterance_of_interest_ids, outputs['utterance_of_interest_ids'].detach().cpu().numpy(), axis=0)
                            candidate_ids_nested=np.append(candidate_ids_nested, outputs['candidate_ids_nested'].detach().cpu().numpy(), axis=0)
                            true_parent_ids=np.append(true_parent_ids, outputs['true_parent_ids'].detach().cpu().numpy(), axis=0)
                    
                        if preds.shape[1] == 1:
                            pred_ids=np.ones(preds.shape)
                            pred_ids[preds < 0]=0
                        else:
                            pred_ids=np.argmax(preds, axis=1)

                last_filename=''
                for filename_id, utterance_of_interest_id, candidate_ids, pred_id, true_parent_id in \
                        zip(filename_ids, utterance_of_interest_ids, candidate_ids_nested, pred_ids, true_parent_ids):
                    
                    filename=reversed_filename_to_filename_id[int(filename_id)]

                    if last_filename != filename:
                        last_filename=filename
                        threads_predicted=0

                    # final_pred=candidate_ids[pred_id]
                    final_pred=f"D{candidate_ids[pred_id]}"
                    if (utterance_of_interest_id == candidate_ids[pred_id]) or (utterance_of_interest_id == 99999):
                        final_pred=f"T{threads_predicted}"
                        threads_predicted += 1

                    pred_lines.append(
                        (
                            reversed_filename_to_filename_id[int(filename_id)],
                            f"D{utterance_of_interest_id}",
                            final_pred, 
                            # pred_id
                        )
                    )
                    if candidate_ids[pred_id] == true_parent_id:
                        correct_predictions_made += 1
                    all_predictions_made +=1 
                        
                
                if (correct_predictions_made/all_predictions_made)>0:
                    auto, _=eval_lines_dict_to_clusters(eval_lines_to_lines_dict(pred_lines))
                    contingency, row_sums, col_sums=None, None, None
                    contingency, row_sums, col_sums=clusters_to_contingency(gold, auto)

                    pairwise_accuracy=correct_predictions_made/all_predictions_made
                    ari=adjusted_rand_index(contingency, row_sums, col_sums)
                    vi=variation_of_information(contingency, row_sums, col_sums)
                    shen=shen_f1(contingency, row_sums, col_sums, gold, auto)
                    oto=one_to_one(contingency, row_sums, col_sums)
                    em_f=exact_match(gold, auto, skip_single=False)
                    
                    main_log(f"Pairwise_accuracy: {pairwise_accuracy*100:.2f}; ari: {ari:5.2f}; 1-vi: {vi:.2f}; shen: {shen:.2f}; oto: {oto:.2f}; em-f: {em_f:.2f}")
                    cluster_metrics=(ari+vi+shen+em_f)/4
                    eval_metric=0.2*pairwise_accuracy+0.8*cluster_metrics

                    if eval_metric>best_eval_metric:
                        best_model_path=f"pytorch_model-{timestamp}-epoch{epoch}.bin"
                        main_log(f"^ New best -- Model saved: {best_model_path}")
                        best_eval_metric=eval_metric
                        # main_log(f"Pairwise_accuracy: {pairwise_accuracy*100:.2f}; ari: {ari:5.2f}; 1-vi: {vi:.2f}; shen: {shen:.2f}; oto: {oto:.2f}; em-f: {em_f:.2f}")
                        # torch.save(model.state_dict(), best_model_path)
                        accelerator.wait_for_everyone()
                        unwrapped_model=accelerator.unwrap_model(model)
                        accelerator.save(unwrapped_model.state_dict(), str(pathlib.Path(args["output_dir"].joinpath(best_model_path))))