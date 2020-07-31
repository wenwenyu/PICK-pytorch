# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/8/2020 10:54 PM

from typing import *

import torch
import torch.nn as nn
import numpy as np

from .encoder import Encoder
from .graph import GLCN
from .decoder import Decoder
from utils.class_utils import keys_vocab_cls, iob_labels_vocab_cls


class PICKModel(nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        embedding_kwargs = kwargs['embedding_kwargs']
        encoder_kwargs = kwargs['encoder_kwargs']
        graph_kwargs = kwargs['graph_kwargs']
        decoder_kwargs = kwargs['decoder_kwargs']
        self.make_model(embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs)

    def make_model(self, embedding_kwargs, encoder_kwargs, graph_kwargs, decoder_kwargs):

        embedding_kwargs['num_embeddings'] = len(keys_vocab_cls)
        self.word_emb = nn.Embedding(**embedding_kwargs)

        encoder_kwargs['char_embedding_dim'] = embedding_kwargs['embedding_dim']
        self.encoder = Encoder(**encoder_kwargs)

        graph_kwargs['in_dim'] = encoder_kwargs['out_dim']
        graph_kwargs['out_dim'] = encoder_kwargs['out_dim']
        self.graph = GLCN(**graph_kwargs)

        decoder_kwargs['bilstm_kwargs']['input_size'] = encoder_kwargs['out_dim']
        if decoder_kwargs['bilstm_kwargs']['bidirectional']:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size'] * 2
        else:
            decoder_kwargs['mlp_kwargs']['in_dim'] = decoder_kwargs['bilstm_kwargs']['hidden_size']
        decoder_kwargs['mlp_kwargs']['out_dim'] = len(iob_labels_vocab_cls)
        decoder_kwargs['crf_kwargs']['num_tags'] = len(iob_labels_vocab_cls)
        self.decoder = Decoder(**decoder_kwargs)

    def _aggregate_avg_pooling(self, input, text_mask):
        '''
        Apply mean pooling over time (text length), (B*N, T, D) -> (B*N, D)
        :param input: (B*N, T, D)
        :param text_mask: (B*N, T)
        :return: (B*N, D)
        '''
        # filter out padding value, (B*N, T, D)
        input = input * text_mask.detach().unsqueeze(2).float()
        # (B*N, D)
        sum_out = torch.sum(input, dim=1)
        # (B*N, )
        text_len = text_mask.float().sum(dim=1)
        # (B*N, D)
        text_len = text_len.unsqueeze(1).expand_as(sum_out)
        text_len = text_len + text_len.eq(0).float()  # avoid divide zero denominator
        # (B*N, D)
        mean_out = sum_out.div(text_len)
        return mean_out

    @staticmethod
    def compute_mask(mask: torch.Tensor):
        '''
        :param mask: (B, N, T)
        :return: True for masked key position according to pytorch official implementation of Transformer
        '''
        B, N, T = mask.shape
        mask = mask.reshape(B * N, T)
        mask_sum = mask.sum(dim=-1)  # (B*N,)

        # (B*N,)
        graph_node_mask = mask_sum != 0
        # (B * N, T)
        graph_node_mask = graph_node_mask.unsqueeze(-1).expand(B * N, T)  # True for valid node
        # If src key are all be masked (indicting text segments is null), atten_weight will be nan after softmax
        # in self-attention layer of Transformer.
        # So we do not mask all padded sample. Instead we mask it after Transformer encoding.
        src_key_padding_mask = torch.logical_not(mask.bool()) & graph_node_mask  # True for padding mask position
        return src_key_padding_mask, graph_node_mask

    def forward(self, **kwargs):
        # input
        whole_image = kwargs['whole_image']  # (B, 3, H, W)
        relation_features = kwargs['relation_features']  # initial relation embedding (B, N, N, 6)
        text_segments = kwargs['text_segments']  # text segments (B, N, T)
        text_length = kwargs['text_length']  # (B, N)
        iob_tags_label = kwargs['iob_tags_label'] if self.training else None  # (B, N, T)
        mask = kwargs['mask']  # (B, N, T)
        boxes_coordinate = kwargs['boxes_coordinate']  # (B, num_boxes, 8)

        ##### Forward Begin #####
        ### Encoder module ###
        # word embedding
        text_emb = self.word_emb(text_segments)

        # src_key_padding_mask is text padding mask, True is padding value (B*N, T)
        # graph_node_mask is mask for graph, True is valid node, (B*N, T)
        src_key_padding_mask, graph_node_mask = self.compute_mask(mask)

        # set of nodes, (B*N, T, D)
        x = self.encoder(images=whole_image, boxes_coordinate=boxes_coordinate, transcripts=text_emb,
                         src_key_padding_mask=src_key_padding_mask)

        ### Graph module ###
        # text_mask, True for valid, (including all not valid node), (B*N, T)
        text_mask = torch.logical_not(src_key_padding_mask).byte()
        # (B*N, T, D) -> (B*N, D)
        x_gcn = self._aggregate_avg_pooling(x, text_mask)
        # (B*N, 1)，True is valid node
        graph_node_mask = graph_node_mask.any(dim=-1, keepdim=True)
        # (B*N, D), filter out not valid node
        x_gcn = x_gcn * graph_node_mask.byte()

        # initial adjacent matrix (B, N, N)
        B, N, T = mask.shape
        init_adj = torch.ones((B, N, N), device=text_emb.device)
        boxes_num = mask[:, :, 0].sum(dim=1, keepdim=True)  # (B, 1)
        # (B, N, D)
        x_gcn = x_gcn.reshape(B, N, -1)
        # (B, N, D), (B, N, N), (B,)
        x_gcn, soft_adj, gl_loss = self.graph(x_gcn, relation_features, init_adj, boxes_num)
        adj = soft_adj * init_adj

        ### Decoder module ###
        logits, new_mask, log_likelihood = self.decoder(x.reshape(B, N, T, -1), x_gcn, mask, text_length,
                                                        iob_tags_label)
        ##### Forward End #####

        output = {"logits": logits, "new_mask": new_mask, "adj": adj}

        if self.training:
            output['gl_loss'] = gl_loss
            crf_loss = -log_likelihood
            output['crf_loss'] = crf_loss
        return output

    def __str__(self):
        '''
        Model prints with number of trainable parameters
        '''
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def model_parameters(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return params
