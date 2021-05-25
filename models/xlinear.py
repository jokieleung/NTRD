import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from lib.config import cfg

from low_rank import LowRank
import blocks
import lib.utils as utils
from layers.positional_encoding import PositionalEncoding

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0



class Decoder(nn.Module):
    def __init__(
        self, 
        vocab_size, 
        embed_dim, 
        dropout, 
        att_type, 
        att_heads, 
        att_mid_dim, 
        att_mid_drop,
        bifeat_emb_act, 
        bifeat_emb_drop, 
        ff_dropout, 
        layer_num
    ):
        super(Decoder, self).__init__()
        self.att_heads = att_heads
        self.layers = nn.ModuleList([])
        self.embed_dim = embed_dim
        for i in range(layer_num):
            sublayer = DecoderLayer( 
                embed_dim = embed_dim, 
                dropout = dropout, 
                att_type = att_type, 
                att_heads = att_heads, 
                att_mid_dim = att_mid_dim, 
                att_mid_drop = att_mid_drop, 
                bifeat_emb_act = bifeat_emb_act, 
                bifeat_emb_drop = bifeat_emb_drop, 
                ff_dropout = ff_dropout,
                last_layer = (i == layer_num -1))
            self.layers.append(sublayer)

        # self.dropout = nn.Dropout(dropout)
        # self.embed_tokens = nn.Embedding(vocab_size, embed_dim)
        # self.embed_scale = math.sqrt(embed_dim)
        # self.embed_positions = PositionalEncoding(
        #     embed_dim, cfg.MODEL.TRANSFORMER.PE_MAX_LEN
        # )

        # self.layer_norm_word = torch.nn.LayerNorm(embed_dim)
        # self.generator = nn.Linear(embed_dim, vocab_size)

        self.wbil1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            utils.activation('CELU'),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbil2 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            utils.activation('CELU'),
            torch.nn.LayerNorm(embed_dim)
        )
        self.wbi_drop = nn.Dropout(dropout)
        self.dropout_lm  = nn.Dropout(dropout)

        self.proj_norm = nn.Sequential(
            nn.Linear(embed_dim * (layer_num + 1), 2 * embed_dim),
            nn.GLU(),
            torch.nn.LayerNorm(embed_dim))

        self.clear_buffer()

    def init_buffer(self, batch_size):
        self.seq_len = 0
        self.x = torch.zeros((batch_size, 1, self.embed_dim)).cuda()
        for layer in self.layers:
            layer.init_buffer(batch_size)

    def clear_buffer(self):
        self.seq_len = None
        self.x = None
        for layer in self.layers:
            layer.clear_buffer()

    def apply_to_states(self, fn):
        self.x = fn(self.x)
        for layer in self.layers:
            layer.apply_to_states(fn)

    def precompute(self, encoder_out):
        p_att_feats = []
        for layer in self.layers:
            key, value2 = layer.precompute(encoder_out)
            p_att_feats.append((key, value2))
        return p_att_feats

    def forward(self, gx, x, encoder_out, att_mask, seq_mask=None, p_att_feats=None, precompute=False):
        att_mask = att_mask.unsqueeze(1)
        
        # embed positions
        # seq_len = prev_output_tokens.size(1)
        # if self.seq_len is not None:
        #     seq_len = self.seq_len + seq_len
        #     self.seq_len = seq_len
        #     positions = self.embed_positions(seq_len)[:,-1,:].unsqueeze(1)
        # else:
        #     positions = self.embed_positions(seq_len)

        # embed tokens and positions
        # x = self.embed_scale * self.embed_tokens(prev_output_tokens)
        # x = x + positions
        
        gx = x
        # x = self.layer_norm_word(x)
        # if self.dropout is not None:
        #     x = self.dropout(x)
        
        # decoder layers
        gx = self.wbil1(gx)
        if self.x is None:
            x_gx = (torch.sum(x.unsqueeze(1) * seq_mask.unsqueeze(-1), -2) / torch.sum(seq_mask, -1).unsqueeze(-1))
        else:
            self.x = self.x + x
            x_gx = self.x / seq_len
        x_gx = self.wbil2(x_gx)
        gx = gx.unsqueeze(1)
        gx = gx * x_gx
        gx = self.wbi_drop(gx)

        gx_arr = [gx]
        for layerid, layer in enumerate(self.layers):
            if precompute == False:
                p_key = None
                p_value2 = None
            else:
                p_key, p_value2 = p_att_feats[layerid] 
            gx, x = layer(gx, x, encoder_out, att_mask, seq_mask=seq_mask, p_key=p_key, p_value2=p_value2, precompute=precompute)
            gx_arr.append(gx)

        gx = torch.cat(gx_arr, dim = -1)
        gx = self.proj_norm(gx)

        gx = self.dropout_lm(gx)
        out = gx
        # out = self.generator(gx)
        return out

class DecoderLayer(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        dropout, 
        att_type, 
        att_heads, 
        att_mid_dim, 
        att_mid_drop,
        bifeat_emb_act, 
        bifeat_emb_drop, 
        ff_dropout, 
        last_layer = False
    ):
        super(DecoderLayer, self).__init__()
        self.last_layer = last_layer
        self.word_attn = LowRank(
            embed_dim = embed_dim, 
            att_type = att_type, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop)
        self.word_dropout = nn.Dropout(dropout)

        self.cross_att = LowRank(
            embed_dim = embed_dim, 
            att_type = att_type, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop)
        self.cross_dropout = nn.Dropout(dropout)
        self.layer_norm_cross = torch.nn.LayerNorm(embed_dim)
        
        self.cross_att2 = LowRank(
            embed_dim = embed_dim, 
            att_type = att_type, 
            att_heads = att_heads, 
            att_mid_dim = att_mid_dim, 
            att_mid_drop = att_mid_drop)
        self.cross_dropout2 = nn.Dropout(dropout)
        self.layer_norm_cross2 = torch.nn.LayerNorm(embed_dim)

        if self.last_layer == False:
            self.bifeat_emb = nn.Sequential(
                nn.Linear(2 * embed_dim, embed_dim),
                utils.activation(bifeat_emb_act),
                nn.Dropout(bifeat_emb_drop)
            )
            self.layer_norm_x = torch.nn.LayerNorm(embed_dim)

            self.ff_layer = blocks.create(
                'FeedForward',
                embed_dim = embed_dim, 
                ffn_embed_dim = embed_dim * 4, 
                relu_dropout = ff_dropout, 
                dropout = ff_dropout)

        self.layer_norm_gx = torch.nn.LayerNorm(embed_dim)

    def apply_to_states(self, fn):
        self.word_attn.apply_to_states(fn)

    def init_buffer(self, batch_size):
        self.word_attn.init_buffer(batch_size)

    def clear_buffer(self):
        self.word_attn.clear_buffer()

    def precompute(self, encoder_out):
        key, value2 = self.cross_att.precompute(encoder_out, encoder_out)
        return key, value2

    def forward(
        self, 
        gx, 
        x, 
        encoder_out, 
        att_mask, 
        seq_mask, 
        p_key=None, 
        p_value2=None, 
        precompute=False
    ):

        #self attn
        word_x = x
        residual = x
        x = self.word_attn.forward2(
            query = gx,
            key = x,
            mask = seq_mask,
            value1 = gx,
            value2 = x)
        x = self.word_dropout(x)
        x = residual + x

        #cross attn
        residual = x
        x = self.layer_norm_cross(x)
        x = self.cross_att.forward2(
            query = x,
            key = encoder_out if precompute == False else p_key,
            mask = att_mask,
            value1 = x,
            value2 = encoder_out if precompute == False else p_value2,
            precompute=precompute)
        x = self.cross_dropout(x)
        x = residual + x
        # gx = self.layer_norm_gx(gx)


        #cross attn 2
        residual = x
        x = self.layer_norm_cross2(x)
        x = self.cross_att2.forward2(
            query = x,
            key = encoder_out if precompute == False else p_key,
            mask = att_mask,
            value1 = x,
            value2 = encoder_out if precompute == False else p_value2,
            precompute=precompute)
        x = self.cross_dropout2(x)
        gx = residual + x
        gx = self.layer_norm_gx(gx)

        if self.last_layer == False:
            x_ = torch.cat([gx, word_x], dim = -1)
            x = self.bifeat_emb(x_) + word_x
            x = self.layer_norm_x(x)

            if self.ff_layer is not None:
                x = self.ff_layer(x)
        else:
            x = None
        return gx, x
