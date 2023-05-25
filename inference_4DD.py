########
# Contains code from the original implementation https://github.com/xbmxb/StructureCharacterization4DD
########
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
import datasets
import transformers
import ortools
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import numpy as np
import ortools.graph.pywrapgraph as pywrapgraph
from sklearn import metrics
from pprint import pformat
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn import CosineEmbeddingLoss
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

from models import *
from eval import *

# from datasets import disable_caching

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

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
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
        
        logits = self.classifier(pooled_output) #(batch_size, num_chioce, 1)
        reshaped_logits = logits.squeeze(2) #(batch_size, num_chioce)
        logits = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        
        outputs={
            "filename_ids": filename_ids,
            "logits": logits[0],
            "utterance_of_interest_ids": utterance_of_interest_ids,
            "candidate_ids_nested": candidate_ids_nested,
        }
        if labels is not None:
            loss_fct=CrossEntropyLoss()
            loss=loss_fct(reshaped_logits, labels)        
            outputs["loss"]=loss
            
        return outputs

def merge(sequences, ignore_idx=None):
    '''
    merge from batch * sent_len to batch * max_len 
    '''
    pad_token=0 if type(ignore_idx)==type(None) else ignore_idx
    #pad_token = PAD_token if type(ignore_idx)==type(None) else ignore_idx
    lengths=[len(seq) for seq in sequences]
    max_len=1 if max(lengths)==0 else max(lengths)
    padded_seqs=torch.ones(len(sequences), max_len).long() * pad_token 
    for i, seq in enumerate(sequences):
        end=lengths[i]
        padded_seqs[i, :end]=seq[:end]
    padded_seqs=padded_seqs.detach() #torch.tensor(padded_seqs)
    return padded_seqs, lengths




def find(x, parents):
    while parents[x] != x:
        parent = parents[x]
        parents[x] = parents[parent]
        x = parent
    return x

def union(x, y, parents, sizes):
    # Get the representative for their sets
    x_root = find(x, parents)
    y_root = find(y, parents)
 
    # If equal, no change needed
    if x_root == y_root:
        return
 
    # Otherwise, merge them
    if sizes[x_root] > sizes[y_root]:
        parents[y_root] = x_root
        sizes[x_root] += sizes[y_root]
    else:
        parents[x_root] = y_root
        sizes[y_root] += sizes[x_root]


def union_find(nodes, edges):
    # Make sets
    parents = {n:n for n in nodes}
    sizes = {n:1 for n in nodes}

    for edge in edges:
        union(edge[0], edge[1], parents, sizes)

    clusters = {}
    for n in parents:
        clusters.setdefault(find(n, parents), set()).add(n)
    cluster_list = list(clusters.values())
    return cluster_list

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
    def _get_examples(self, mode, lines_dict, reversed_filename_to_filename_id=None):
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
            examples[filename_id].append((None, utterance_id))

        return examples
    
    def get_examples(self, tokenizer, mode, 
                     reversed_filename_to_filename_id, line_id2line_text, line_id2speaker_n, scene_id2line_ids, line_id2scene_id, 
                     max_previous_utterance, lines_dict):        
        
        start_token=tokenizer.cls_token  
        sep_token=tokenizer.sep_token
        PAD_UTTERANCE_ID='D99999'
    
        filenames=[]
        reshaped_examples=[]
        
        examples=self._get_examples(mode, lines_dict, reversed_filename_to_filename_id)
    
        for filename_id, info_tuples in examples.items():
            filename=reversed_filename_to_filename_id[filename_id]

            if filename not in filenames:
                filenames.append(filename)
                        
            for info_tuple_idx, info_tuple in enumerate(info_tuples):
                seen_cands=[]
                _, utterance_id=info_tuple
            
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

                    # print(f"file: {filename_id}; i: {i}: j: {j}: diff: {diff}, len_info_tuples: {len(info_tuples)}; tuple_idx: {tuple_idx}")

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

                    if line_id2scene_id[mode][filename_id][text_b_utterance_id] != uoi_scene:
                        text_b.append('')
                        text_a.append(uoi_text)
                        candidate_ids.append(PAD_UTTERANCE_ID)

                        adj_matrix_speaker[i][j]=0
                        adj_matrix_speaker[j][i]=0                        
                        adj_matrix_scene[i][j]=0
                        adj_matrix_scene[j][i]=0     
                        continue # if not the same thing can't be in the same thread

                    candidate_ids.append(text_b_utterance_id)

                    if text_b_utterance_id in seen_cands:
                        # an uoi should not have the same candidate (we're operating on the candidate level now)
                        continue

                    text_b.append(line_id2line_text[mode][filename_id][text_b_utterance_id])
                    text_a.append(uoi_text)
                    seen_cands.append(text_b_utterance_id)

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
                        true_parent_id=None,#int(true_parent_id[1:]),                        
                        label=None,
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
            tokens_b=tokens_b + [sep_token]#["[SEP]"]

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
                true_parent_id=None,#example.true_parent_id,
                label=label_id,
                adj_matrix_speaker=example.adj_matrix_speaker,
                adj_matrix_scene=example.adj_matrix_scene
                )
        )

    return features
            
def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
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



def eval_lines_to_lines_dict(eval_lines):
    eval_lines_dict={}
    
    for line in eval_lines:
        filename, uoi_id, parent_id=line
        
        if not uoi_id or not parent_id:
            continue
            
        if not uoi_id.startswith('D'):
            continue
            
        if parent_id.startswith('DA'):
            print(parent_id)
            continue

        if filename not in eval_lines_dict:
            eval_lines_dict[filename]=[]

        if parent_id.startswith('T'):
            parent_id=uoi_id

            
        uoi_id=int(uoi_id.replace('D', ''))
        parent_id=int(parent_id.replace('D', ''))

        eval_lines_dict[filename].append([uoi_id, parent_id])
    
    return eval_lines_dict    
    
def eval_lines_dict_to_clusters(eval_lines_dict):
    nodes={}
    edges={}
    cluster_dict={}
    
    for filename, lines in eval_lines_dict.items():
        for line in lines:
            parts=line[:]
            source=max(parts)
            nodes.setdefault(filename, set()).add(source)
            parts.remove(source)
            for num in parts:
                edges.setdefault(filename, []).append((source, num))
                nodes.setdefault(filename, set()).add(num)

    for filename in nodes:
        cluster_dict[filename]=[]
        clusters=union_find(nodes[filename], edges[filename])
        cluster_dict[filename].extend(clusters)
        
    return cluster_dict


def main_log(msg):
    global logger
    return logger.info(msg, main_process_only=True)

def to_cuda(x):
    if torch.cuda.is_available(): x=x.cuda()
    return x

def gen_file_lines(file_path):
    with open(file_path, 'r') as f:
        for line in f:
            line=line.replace('\n', '')
            if line.startswith('category\t'):
                continue
            cols=line.split('\t')

            if len(cols) == 13:
                category, filename, title, _, turn_line_no, scene_id, _,  _, line_no, speaker_label, _,    anno, line_text=cols


            if len(cols) == 12:
                category, filename, title, _, _, scene_id, turn_line_no, line_no, _, speaker_label, anno, line_text=cols
            yield category, filename, title, turn_line_no, scene_id, line_no, speaker_label, line_text


if __name__=='__main__':

    ROOT=pathlib.Path('/global/scratch/users/kentkchang/dramatic-bert')
    MODEL_PATH=ROOT / 'model'

    CWD=pathlib.Path.cwd()


    LOG_PATH=MODEL_PATH / 'log'

    ######
    # args={}
    arg_parser = argparse.ArgumentParser()

    arg_parser.add_argument('--model_path', help='specify model_path')
    arg_parser.add_argument('--encoder_name', help='specify encoder_name')
    arg_parser.add_argument('--model_folder', help='specify model_folder')
    arg_parser.add_argument('--model_name', help='specify model_name')
    arg_parser.add_argument('--test_folder', help='specify test_folder')
    arg_parser.add_argument('--preds_output_folder', help='specify preds_output_folder')
    arg_parser.add_argument("--batch_size",
                        default=4,
                        type=int,
                        help="specific batch_size.")
    arg_parser.add_argument("--max_previous_utterance",
                        default=50,
                        type=int,
                        help="The maximum of previous utterances considerated.")
    arg_parser.add_argument("--use_tqdm",
                        default=False,
                        type=bool,
                        help="Use tqdm?")

    args=vars(arg_parser.parse_args())


    BATCH_SIZE=args['batch_size']
    OUTPUT_PATH=MODEL_PATH / args["model_folder"]
    args['epochs']=6
    args['batch_size']=BATCH_SIZE
    args["fix_encoder"]=0
    args["n_gpu"]=torch.cuda.device_count()
    args["distance_embedding_size"]=10
    args["output_dir"]=str(LOG_PATH)
    
    use_tqdm=args['use_tqdm']

    TEST_DATA_PATH=CWD / str(args['test_folder'])
    TEST_DATA_PATH.mkdir(exist_ok=True)

    PREDS_FOLDER_PATH=CWD / str(args['preds_output_folder'])
    PREDS_FOLDER_PATH.mkdir(exist_ok=True)

    PREDS_PATH=PREDS_FOLDER_PATH / 'preds'
    PREDS_PATH.mkdir(exist_ok=True)
    
    CLUSTERS_PATH=PREDS_FOLDER_PATH / 'cluster'
    CLUSTERS_PATH.mkdir(exist_ok=True)
        
    ######

    ddp_kwargs=DistributedDataParallelKwargs(find_unused_parameters=False)
    accelerator=Accelerator(kwargs_handlers=[ddp_kwargs])

    timestamp=datetime.datetime.now().strftime("%m%d%Y-%H%M%S")    
    ######

    logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename=os.path.join(args["output_dir"], f"infer_{timestamp}.log"),
                    filemode='w')
    console=logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter=logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()


    logger=get_logger(__name__)
    

    # set_seed(2022)
    accelerator.wait_for_everyone()    

    main_log(f"Cache: {datasets.is_caching_enabled()}")
    datasets.disable_caching()
    main_log(f"Cache, now: {datasets.is_caching_enabled()}")

    pretrained_model_name=args['model_name']
    max_previous_utterance=args['max_previous_utterance']

    main_log(f'Initiating the PrLM: {pretrained_model_name}; max_previous_utterance: {max_previous_utterance}')

    config_class, model_class, tokenizer_class=BertConfig, Bert_v7, BertTokenizer
    tokenizer=tokenizer_class.from_pretrained(pretrained_model_name, 
                                            do_lower_case=False, 
                                            do_basic_tokenize=False)

    processor=DCDProcessor()

    label_list=processor.get_labels(max_previous_utterance)
    num_labels=len(label_list)

    config=config_class.from_pretrained(pretrained_model_name, num_labels=num_labels)
    model=model_class.from_pretrained(pretrained_model_name, config=config)

    HIDDEN_DIM=config.hidden_size
    SEQUENCE_MAX_LEN=tokenizer.model_max_length    

    model.load_state_dict(torch.load(OUTPUT_PATH.joinpath(args["model_path"])))

    main_log(f'Loaded {OUTPUT_PATH.joinpath(args["model_path"])}!')
    model=accelerator.prepare(model)

    main_log('Loading files ...')
    file_paths=TEST_DATA_PATH.glob('*.tsv')

    if use_tqdm:
        main_log('Using tqdm')

    skip=[]
    with open('analysis/scriptbase_j_skip') as skip_file:
        for line in skip_file:
            skip.append(line.strip().split('\t')[1])

    if CLUSTERS_PATH.glob('*.txt'):
        for file in CLUSTERS_PATH.glob('*.txt'):
            slug=file.parts[-1].replace('.txt', '')
            skip.append(slug)

    skipping=', '.join(skip)
    main_log(f"Not making pres for: {skipping}")

    if accelerator.is_main_process:
        if use_tqdm:
            file_paths=tqdm(file_paths)

    main_log('Working through files ...')
    for file_path in file_paths:
        slug=file_path.parts[-1].replace('.tsv', '')

        if slug in skip:
            continue

        start_time=time.monotonic()

        lines={'test': {}}
        scene_id2line_ids={'test': {}}
        line_id2line_text={'test': {}}
        line_id2turn_n={'test': {}}
        line_id2speaker={'test': {}}
        speaker2line_ids={'test': {}}
        speakers={'test': {}}
        line_id2speaker_n={'test': {}}
        line_id2scene_id={'test': {}}

        filename_to_filename_id={}
        last_scene_id=''
        last_turn_line_no=''
        scene_line_ids=[]

        file_lines={}

        file_len=len(list(gen_file_lines(file_path)))
        mode='test' # just to ensure format consistency -- was train, dev, test

        for line in gen_file_lines(file_path):
            category, filename, title, turn_line_no, scene_id, line_no, speaker_label, line_text=line
            if filename not in file_lines:
                file_lines[filename]={}

            file_lines[filename][line_no]=(
                (category, filename, title, turn_line_no, scene_id, speaker_label, line_text)
            )

            if filename not in speakers[mode]:
                speakers[mode][filename]={}
            if filename not in filename_to_filename_id:
                filename_to_filename_id[filename]=len(filename_to_filename_id)        

            filename_id=filename_to_filename_id[filename]

            if filename_id not in speakers[mode]:
                scene_speaker_id=speakers[mode][filename_id]={}

            if (line_no.startswith('D')):  
                speakers[mode][filename_id][speaker_label]=len(speakers[mode][filename_id])
                scene_speaker_id=speakers[mode][filename_id][speaker_label]

                lines[mode][(filename_id, line_no)]={
                    'corpus': category,
                    'filename_id': filename_id,
                    'title': title,
                    'scene_id': scene_id,
                    'scene_speaker_id': scene_speaker_id,
                    'turn_line_no': turn_line_no,
                    'reply_to_id': None
                }
                last_turn_line_no=turn_line_no


            if filename_id not in line_id2line_text[mode]:
                line_id2line_text[mode][filename_id]={}             
            line_id2line_text[mode][filename_id][line_no]=f"{line_text}"

            if filename_id not in scene_id2line_ids[mode]:
                scene_id2line_ids[mode][filename_id]={}                         
            if scene_id not in scene_id2line_ids[mode][filename_id]:
                scene_id2line_ids[mode][filename_id][scene_id]=[]

            if line_no.startswith('A') or line_no.startswith('D'):
                scene_id2line_ids[mode][filename_id][scene_id].append(line_no)

            if filename_id not in line_id2scene_id[mode]:
                line_id2scene_id[mode][filename_id]={}                         
            if line_no.startswith('D'):
                line_id2scene_id[mode][filename_id][line_no]=scene_id                
            if filename_id not in line_id2turn_n[mode]:
                line_id2turn_n[mode][filename_id]={}   
            if line_no.startswith('D'):
                line_id2turn_n[mode][filename_id][line_no]=turn_line_no

            if filename_id not in line_id2speaker[mode]:
                line_id2speaker[mode][filename_id]={}   
            if filename_id not in speaker2line_ids[mode]:
                speaker2line_ids[mode][filename_id]={} 

            if line_no.startswith('D'):
                line_id2speaker[mode][filename_id][line_no]=scene_speaker_id
                if scene_speaker_id not in speaker2line_ids[mode][filename_id]:
                    speaker2line_ids[mode][filename_id][scene_speaker_id]=[]
                speaker2line_ids[mode][filename_id][scene_speaker_id].append(line_no)

            if filename_id not in line_id2speaker_n[mode]:
                line_id2speaker_n[mode][filename_id]={}   
            if line_no.startswith('D'):
                line_id2speaker_n[mode][filename_id][line_no]=scene_speaker_id

        reversed_filename_to_filename_id={v: k for k, v in filename_to_filename_id.items()}

        ########
        
        test_examples, test_filenames=\
            processor.get_examples(tokenizer, 'test',
                                reversed_filename_to_filename_id,
                                line_id2line_text, 
                                line_id2speaker_n, 
                                scene_id2line_ids, 
                                line_id2scene_id,
                                max_previous_utterance, 
                                lines)
        test_data=TensorDataset(*prep_tensor_data(convert_examples_to_features(test_examples, label_list, SEQUENCE_MAX_LEN, max_previous_utterance, tokenizer)))
        test_sampler=SequentialSampler(test_data)
        test_data_loader=DataLoader(test_data, sampler=test_sampler, batch_size=BATCH_SIZE)
        test_data_loader=accelerator.prepare(test_data_loader)
    
        model.eval()
        preds=None
        pred_lines=[]
        preds_dict={}

        for idx, batch in enumerate(test_data_loader):
            d={
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2], # if args.model_type in ['bert', 'xlnet', 'albert'] else None, # XLM don't use segment_ids
                'adj_matrix_speaker': batch[3],
                'adj_matrix_scene': batch[4],
                'filename_ids': batch[5],
                'utterance_of_interest_ids': batch[6],
                'candidate_ids_nested': batch[7],
            }
            d={key: to_cuda(val) for key, val in d.items()} 
            with torch.no_grad():        
                outputs=model(**d)
                outputs=accelerator.gather(outputs)
                
                if preds is None:
                    preds=outputs['logits'].detach().cpu().numpy()
                    filename_ids=outputs['filename_ids'].detach().cpu().numpy()
                    utterance_of_interest_ids=outputs['utterance_of_interest_ids'].detach().cpu().numpy()
                    candidate_ids_nested=outputs['candidate_ids_nested'].detach().cpu().numpy()
                else:
                    preds=np.append(preds, outputs['logits'].detach().cpu().numpy(), axis=0)
                    filename_ids=np.append(filename_ids, outputs['filename_ids'].detach().cpu().numpy(), axis=0)
                    utterance_of_interest_ids=np.append(utterance_of_interest_ids, outputs['utterance_of_interest_ids'].detach().cpu().numpy(), axis=0)
                    candidate_ids_nested=np.append(candidate_ids_nested, outputs['candidate_ids_nested'].detach().cpu().numpy(), axis=0)

                if preds.shape[1] == 1:
                    pred_ids=np.ones(preds.shape)
                    pred_ids[preds < 0]=0
                else:
                    pred_ids=np.argmax(preds, axis=1)

                last_filename=''
                for filename_id, utterance_of_interest_id, candidate_ids, pred_id in \
                        zip(filename_ids, utterance_of_interest_ids, candidate_ids_nested, pred_ids):

                    # main_log((filename_id, utterance_of_interest_id, candidate_ids, pred_id, true_parent_id))

                    filename=reversed_filename_to_filename_id[int(filename_id)]

                    if filename_id not in preds_dict:
                        preds_dict[filename_id]={}                

                    if last_filename != filename:
                        last_filename=filename
                        threads_predicted=0

                    final_pred=f"D{candidate_ids[pred_id]}"
                    if (utterance_of_interest_id == candidate_ids[pred_id]) or (utterance_of_interest_id == 99999):
                        final_pred=f"T{threads_predicted}"
                        threads_predicted+=1

                    preds_dict[int(filename_id)][f"D{utterance_of_interest_id}"]=final_pred


        test_pred_lines=[]
        for filename_id, pred in preds_dict.items():
            # main_log(item)
            filename=reversed_filename_to_filename_id[int(filename_id)]

            with open(PREDS_PATH.joinpath(filename+'.tsv'), 'w') as p:
                # if last_filename != filename:
                #     last_filename=filename
                for utterance_id_str, final_pred_str in pred.items():
                    test_pred_lines.append([filename, utterance_id_str, final_pred_str])
                    category, filename, title, turn_line_no, scene_id, speaker_label, line_text=file_lines[filename][utterance_id_str]

                    out=[category, filename, title, turn_line_no, scene_id, speaker_label, utterance_id_str, final_pred_str, line_text]
                    # out_str='\t'.join(out)+'\n'
                    # main_log(f"Writing: {out_str}")
                    p.write('\t'.join(out)+'\n')


        cluster_dict=eval_lines_dict_to_clusters(eval_lines_to_lines_dict(test_pred_lines))

        for filename, clusters in cluster_dict.items():
            with open(CLUSTERS_PATH.joinpath(filename+'.txt'), 'w') as c:
                for cluster in clusters:
                    vals=[str(v) for v in cluster]
                    vals.sort()
                    c.write(filename +":"+ " ".join(vals)+'\n')

        time_diff=datetime.timedelta(seconds=time.monotonic() - start_time)
        main_log(f"{slug}: {file_len} [{time_diff}]")
        # main_log(dataset.cache_files())
        
                

