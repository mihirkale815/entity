import pandas as pd
import numpy as np
import torch
from torchtext.data import Field
from torchtext import data
from torchtext import datasets
from torchtext.vocab import GloVe
from torchtext.data import Iterator, BucketIterator



def extract_chunks(tagged_sent, chunk_type):
    grp1, grp2, chunk_type = [], [], "-" + chunk_type
    for ind, (s, tp) in enumerate(tagged_sent):
        if tp.endswith(chunk_type):
            if not tp.startswith("b"):
                grp2.append(str(ind))
                grp1.append(s)
            else:
                if grp1:
                    yield " ".join(grp1), "-".join(grp2)
                grp1, grp2 = [s], [str(ind)]
    yield " ".join(grp1), "-".join(grp2)


class DatasetUtil:

    def __init__(self,args):
        self.args = args
        self.use_cuda = self.args['cuda'] and torch.cuda.is_available()
        self.tokenize = lambda x: x.split()

    def get_train_iterator(self):

        self.WORD = data.Field(init_token="<bos>", eos_token="<eos>",tokenize=self.tokenize, lower=True)
        self.TAG = data.Field(init_token="<bos>", eos_token="<eos>",tokenize=self.tokenize, lower=True)

        train_dataset = datasets.SequenceTaggingDataset(path=self.args['datapath']+self.args['filename'],
                                                                fields=[('tag',self.TAG),('word',self.WORD)])


        emb_size = self.args.get(['pretrain_size'],None)
        emb_type = self.args.get(['pretrain_type'],None)    
        if emb_type and emb_size : self.WORD.build_vocab(train_dataset,vectors = GloVe(name=emb_type, dim=emb_size))
        else : self.WORD.build_vocab(train_dataset)
        self.TAG.build_vocab(train_dataset)



        device = 0 if self.use_cuda else None
        train_iter = BucketIterator.splits(
            [train_dataset],
            batch_sizes=[self.args['batch_size']],
            device=device,
            sort_key=lambda x: len(x.word),
            sort_within_batch=False,
            repeat=False)[0]

        return train_iter


    def get_iterator(self,args):

        dataset = datasets.SequenceTaggingDataset(path=args['datapath']+args['filename'],
                                                                fields=[('tag',self.TAG),('word',self.WORD)])

        device = 0 if self.use_cuda else -1
        dataset_iter = BucketIterator.splits(
            [dataset],
            batch_sizes=[self.args['batch_size']],
            device=device,
            sort_key=lambda x: len(x.word),
            sort_within_batch=False,
            repeat=False)[0]

        return dataset_iter

    def save(self,path_pt):
        torch.save((self.WORD,self.TAG), path_pt)



def sentence_generator(path):
	f = open(path)
	curr_sent = []
	for line in f:
		line = line.strip("\n")
		if line == '':
			yield curr_sent
			curr_sent = []
			continue
		tag,word = line.split("\t")
		curr_sent.append((word,tag))
	yield curr_sent
	f.close()
