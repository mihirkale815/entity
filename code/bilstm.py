import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import json
torch.manual_seed(1)
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator
from torch.autograd import Variable
import torch.nn.functional as F
import os



class BiLSTM(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)
        self.ix_to_tag = {v:k for k,v in self.tag_to_ix.items()}

        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=bidirectional,dropout=0.25)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)
        self.loss_function = nn.NLLLoss(ignore_index=1)



    def init_hidden(self,batch_size):
        return (autograd.Variable(torch.randn(2, batch_size, self.hidden_dim // 2)).cuda(),
                autograd.Variable(torch.randn(2, batch_size, self.hidden_dim // 2)).cuda())


    def _get_lstm_features(self, batch):
        batch_size = batch.size()[1]
        #self.hidden = self.init_hidden(batch_size)
        embeds = self.word_embeds(batch)
        lstm_out, self.hidden = self.lstm(embeds)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def loss(self, batch, tags):
        batch_size = batch.size()[1]
        seq_len = batch.size()[0]
        score, tag_seq = self.forward(batch)
        total_loss = self.loss_function(score, tags.view(batch_size * seq_len))
        return total_loss


    def forward(self, batch):
        batch_size = batch.size()[1]
        seq_len = batch.size()[0]
        lstm_feats = self._get_lstm_features(batch)
        lstm_feats = lstm_feats.view(batch_size * seq_len, self.tagset_size)
        score = F.log_softmax(lstm_feats, dim = 1)
        _, tag_seq  = torch.max(score, dim=1)
        tag_seq = tag_seq.view(seq_len, batch_size).transpose(0,1)
        return score, tag_seq