# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

#!/usr/bin/env python3

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_span import AdaptiveSpan

# Size notations:
# B = batch_size, H = hidden_size, M = block_size, L = attn_span
# md note:M = block_size ,it is the length of sequence to process in parallel. but what  does exactly it refer to ,
# I still don't know.
# In the trandformer case.
# Firstly,
# A constant rule of the multi head attention mechanism is that the query is destination side, the tensor size of 
# key and value must be equal, and key and value  is as source side ,
# for example ,in self attention layer of encoder case, query=key=value
# in self attention layer of decoder case, query is current step, key and value are previous step.
# in encode-decoder attention case, query is decode side and it is the current step output of self attention layer in decoder.
# key and value are output of last layer of encoder.
# you can see the paper "A COMPARATIVE STUDY ON TRANSFORMER VS RNN IN SPEECH APPLICATIONS"

# Secondly,
# model dimension(e.g: d_model) is hidden_size and also per head attention dimension *number of heads  in multi head attention case.
# M^Q ，M^K and M^V in multi head attention,
# M^Q , M^K and M^V ，their size :d_model * d_model.
#  you can see section 3.1 of the paper:"A COMPARATIVE STUDY ON TRANSFORMER VS RNN IN SPEECH APPLICATIONS"
# in the multi head layer case, query size is (B_K) x M x D 

def _skew(X, pad_value):
    """shift every row 1 step to right"""
    # X = B x M x L
    B, M, L = X.size()
    X = F.pad(X, (0, M + 1), value=pad_value)  # B x M x (L+M+1)
    X = X.view(B, -1)  # B x ML+MM+M
    X = X[:, :-M]  # B x ML+MM
    X = X.view(B, M, M + L)  # B x M x L+M
    return X


def _unskew(X):
    """reverse _skew operation"""
    # X = B x M x L+M
    B, M, L = X.size()
    L -= M
    X = X.view(B, -1)  # B x ML+MM
    X = F.pad(X, (0, M))  # B x ML+MM+M
    X = X.view(B, M, M + L + 1)  # B x M x L+M+1
    X = X[:, :, :L]  # B x M x L
    return X


class SeqAttention(nn.Module):
    """Sequential self-attention layer.
    Each token will attend to its previous fixed number of steps.
    Note that attention doesn't include the current step itself.
    """
    """
    md note: in order to be consitence with class MultiHeadSeqAttention(nn.Module), and easy to understand,
     I  will add some document for this script and modified original author's document.
    """
    def __init__(self, hidden_size, attn_span,
                 dropout, adapt_span_params, **kargs):
        nn.Module.__init__(self)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size # size of a single head, you can see the line (self.attn = SeqAttention(
                                       # hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)) 
                                       # in class MultiHeadSeqAttention(nn.Module), 
                                       # you will know here hidden_size is head_dim.
                                       # head_dim is d_k in “Attention is all you need ”
        self.attn_span = attn_span
        self.adapt_span_enabled = adapt_span_params['adapt_span_enabled']
        if self.adapt_span_enabled:
            self.adaptive_span = AdaptiveSpan(attn_span=attn_span,
                                              **adapt_span_params, **kargs)

    def forward(self, query, key, value, key_pe):
        # md note:in order to be consitence with class MultiHeadSeqAttention(nn.Module)
        # md note: query size = B_K x M x D
        #          key, value sizes = B_K x (M+L) x D
        # query size = B x M x H
        # key, value sizes = B x (M+L) x H

        if self.adapt_span_enabled:
            # [optional] trim out memory to reduce unnecessary computation
            # 
            key, value, key_pe = self.adaptive_span.trim_memory(
                query, key, value, key_pe)

        # compute attention from context
        # B_K x M (dest) x (M+L) (src)
        attn_cont = torch.matmul(query, key.transpose(-1, -2)) # md note:key :B_K x (M+L) x D
        attn_cont = _unskew(attn_cont)  # B_K x M x L

        # compute the effect of position embedding 
        attn_pos = torch.matmul(query, key_pe)  # B_K x M x L_pos , md note: L_pos is equal to L ,otherwise they don't add .
                                                                  # then key_pe should be equal to key.transpose(-1,-2)
                                                                  # key_pe : B_K x D x (M+L)
        attn = attn_cont + attn_pos # md note: B_K x M x L_pos

        attn = attn / math.sqrt(self.hidden_size)  # B_K x M X L_pos, md note:hidden_size is head_dim.
                                                                    
                                                                  
                                                                   
        attn = F.softmax(attn, dim=-1)
        if self.adapt_span_enabled:
            # trim attention lengths according to the learned span
            attn = self.adaptive_span(attn)
        attn = self.dropout(attn)  # B x M X L_pos # md note:B_K x M X L_pos 

        attn_cont = _skew(attn, 0)  # B x M X (L+M) # md note:B_K x M x (L_pos+M)
        out = torch.matmul(attn_cont, value)  # B x M x H ,#md note:B_K x M x D

        return out

    def get_cache_size(self):
        if self.adapt_span_enabled:
            return self.adaptive_span.get_cache_size()
        else:
            return self.attn_span


class MultiHeadSeqAttention(nn.Module):
    def __init__(self, hidden_size, nb_heads, **kargs):
        nn.Module.__init__(self)
        assert hidden_size % nb_heads == 0
        self.nb_heads = nb_heads
        self.head_dim = hidden_size // nb_heads
        self.attn = SeqAttention(
            hidden_size=self.head_dim, nb_heads=nb_heads, **kargs)
        self.proj_query = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_out = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_val = nn.Linear(hidden_size, hidden_size, bias=False)
        self.proj_key = nn.Linear(hidden_size, hidden_size, bias=False)

    def head_reshape(self, x):
        K = self.nb_heads. # hidden_size = nb_heads * head_dim ; in other words, H = K * D
        D = self.head_dim
        x = x.view(x.size()[:-1] + (K, D))  # md note: before x : B x (M+L) x H, after x :B x (M+L) x K x D
        x = x.transpose(1, 2).contiguous()  # B x K x (M+L) x D
        x = x.view(-1, x.size(-2), x.size(-1))  # B_K x (M+L) x D ,it should be (B+K) x M+L) x D, to order to be 
                                                # consistent with original author symbol, i also use B_K to replace B+K.
        return x

    def forward(self, query, key, value, key_pe):
        B = query.size(0)
        K = self.nb_heads
        D = self.head_dim
        M = query.size(1)

        query = self.proj_query(query) # before query : B x M x H ,after query : B x M x H
        query = self.head_reshape(query) # after query : (B_K) x M x D
        value = self.proj_val(value)     #before value : B x (M+L) x H, after value: B x (M+L) x H
        value = self.head_reshape(value) # after value : (B_K) x (M+L) x D
        key = self.proj_key(key)         # before key : B x (M+L) x H, after key: B x (M+L) x H
        key = self.head_reshape(key)     # after key : (B_K) x (M+L) x D

        out = self.attn(query, key, value, key_pe)  # B_K x M x D
        out = out.view(B, K, M, D)  # B x K x M x D
        out = out.transpose(1, 2).contiguous()  # B x M x K x D
        out = out.view(B, M, -1)  # B x M x K_D
        out = self.proj_out(out)
        return out


class FeedForwardLayer(nn.Module):
    def __init__(self, hidden_size, inner_hidden_size, dropout, **kargs):
        nn.Module.__init__(self)
        self.fc1 = nn.Linear(hidden_size, inner_hidden_size)
        self.fc2 = nn.Linear(inner_hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        h1 = F.relu(self.fc1(h))
        h1 = self.dropout(h1)
        h2 = self.fc2(h1)
        return h2


class TransformerSeqLayer(nn.Module):
    def __init__(self, hidden_size, **kargs):
        nn.Module.__init__(self)
        self.attn = MultiHeadSeqAttention(hidden_size=hidden_size, **kargs)
        self.ff = FeedForwardLayer(hidden_size=hidden_size, **kargs)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, h, h_cache, key_pe):
        # h = B x M x H
        # h_cache = B x L x H
        h_all = torch.cat([h_cache, h], dim=1)  # B x (M+L) x H
        attn_out = self.attn(h, h_all, h_all, key_pe)
        h = self.norm1(h + attn_out)  # B x M x H
        ff_out = self.ff(h)
        out = self.norm2(h + ff_out)  # B x M x H
        return out


class TransformerSeq(nn.Module):
    def __init__(self, vocab_size, hidden_size, nb_heads, nb_layers,
                 attn_span, **kargs):
        nn.Module.__init__(self)
        # token embeddings
        self.in_emb = nn.Embedding(vocab_size, hidden_size)
        self.out_emb = nn.Linear(hidden_size, vocab_size)
        # position embeddings
        self.key_pe = nn.Parameter(
            torch.randn(1, hidden_size // nb_heads, attn_span))

        self.layers = nn.ModuleList()
        self.layers.extend(
            TransformerSeqLayer(
                hidden_size=hidden_size, nb_heads=nb_heads,
                attn_span=attn_span, **kargs)
            for _ in range(nb_layers))

    def forward(self, x, h_cache):
        # x size = B x M
        block_size = x.size(1)
        h = self.in_emb(x)  # B x M x H
        h_cache_next = []
        for l, layer in enumerate(self.layers):
            cache_size = layer.attn.attn.get_cache_size()
            if cache_size > block_size:
                h_cache_next_l = torch.cat(
                    [h_cache[l][:, -cache_size + block_size:, :], h],
                    dim=1).detach()
            else:
                h_cache_next_l = h[:, -cache_size:, :].detach()
            h_cache_next.append(h_cache_next_l)
            h = layer(h, h_cache[l], self.key_pe)  # B x M x H

        out = F.log_softmax(self.out_emb(h), dim=-1)

        return out, h_cache_next
