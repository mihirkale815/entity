import sys
sys.path.append("../code/")
from data.utils import DatasetUtil
from bilstm import BiLSTM
import torch
import torch.optim as optim
torch.manual_seed(1)
import argparse

def get_loss_for_dataset(data_iter,model):
    epoch_loss = 0
    for idx ,batch in enumerate(data_iter):
        loss = model.loss(batch.word.cuda(), batch.tag.cuda())
        epoch_loss+=loss
    return epoch_loss


def persist_output(path,Actual,Preds,Data):
    fout = open(path,"w")
    for i in range(0,len(Preds)):
        sentence = Data[i]
        predicted = Preds[i]
        actual = Actual[i]
        for idx in range(0,len(actual)):
            if actual[idx].lower() in ['pad','<eos>','<bos>','<pad>'] : continue
            if predicted[idx].lower() in ['pad', '<eos>', '<bos>', '<pad>']: predicted[idx] = 'o'
            fout.write(" ".join([sentence[idx],'UNK','UNK',actual[idx].upper(),predicted[idx].upper()]) + "\n")
        fout.write("\n")
    fout.close()


def get_output(model,dataset_iter,datasetutil):
    Actual = []
    Preds = []
    Sentences = []
    for batch in dataset_iter:
        batch.word = batch.word.cuda()
        batch.tag = batch.tag.cuda()
        score,tags = model(batch.word.cuda())
        for i in range(tags.size()[0]):
            sentence = batch.word[:,i]
            actual_tags = batch.tag[:,i]
            predicted_tags = tags[i,:]
            sentence = [datasetutil.WORD.vocab.itos[idx.data[0]] for idx in sentence]
            predicted = [datasetutil.TAG.vocab.itos[idx.data[0]] for idx in predicted_tags]
            actual = [datasetutil.TAG.vocab.itos[idx.data[0]] for idx in actual_tags]
            Actual.append(actual)
            Preds.append(predicted)
            Sentences.append(sentence)
    return Actual,Preds,Sentences


parser = argparse.ArgumentParser(description='bilstm for sequence tagging')
parser.add_argument('--data_dir', action="store", dest="data_dir", type=str,default="")
parser.add_argument('--train_path', action="store", dest="train_path", type=str)
parser.add_argument('--dev_path', action="store", dest="dev_path", type=str)
parser.add_argument('--test_path', action="store", dest="test_path", type=str)
parser.add_argument('--train_batch_size', action="store", dest="train_batch_size", type=int,default=32)
parser.add_argument('--dev_batch_size', action="store", dest="dev_batch_size", type=int,default=32)
parser.add_argument('--test_batch_size', action="store", dest="test_batch_size", type=int,default=32)
parser.add_argument('--embedding_dim', action="store", dest="embedding_dim", type=int,default=100)
parser.add_argument('--hidden_dim', action="store", dest="hidden_dim", type=int,default=200)
parser.add_argument('--learning_rate', action="store", dest="learning_rate", type=float,default=0.001)
parser.add_argument('--num_epochs', action="store", dest="num_epochs", type=int,default=10)
parser.add_argument('--early_stopping', action="store", dest="early_stopping", type=int,default=1)
parser.add_argument('--train_output_path', action="store", dest="train_output_path", type=str)
parser.add_argument('--dev_output_path', action="store", dest="dev_output_path", type=str)
parser.add_argument('--test_output_path', action="store", dest="test_output_path", type=str)
parser.add_argument('--glove', action="store", dest="glove", type=str,default=None)

args = parser.parse_args()

datasetutil_args = {}
datasetutil_args['cuda'] = True
datasetutil_args['pretrain_type'] = args.glove
datasetutil_args['pretrain_size'] = 100


datasetutil_args['datapath'] = args.data_dir
datasetutil_args['filename'] = args.train_path
datasetutil_args['batch_size'] = args.train_batch_size
datasetutil = DatasetUtil(datasetutil_args)
train_iter = datasetutil.get_train_iterator()

datasetutil_args['filename'] = args.test_path
datasetutil_args['batch_size'] = args.test_batch_size
test_iter = datasetutil.get_iterator(datasetutil_args)

datasetutil_args['filename'] = args.dev_path
datasetutil_args['batch_size'] = args.dev_batch_size
dev_iter = datasetutil.get_iterator(datasetutil_args)


model = BiLSTM(len(datasetutil.WORD.vocab.stoi),datasetutil.TAG.vocab.stoi, args.embedding_dim, args.hidden_dim).cuda()
if datasetutil_args['cuda'] : model = model.cuda()

optimizer = optim.Adam(model.parameters(), lr = args.learning_rate)


prev_dev_epoch_loss = None
for epoch in range(args.num_epochs):
    print("epoch = ", epoch)
    epoch_loss = 0

    for idx, batch in enumerate(train_iter):
        optimizer.zero_grad()
        loss = model.loss(batch.word.cuda(), batch.tag.cuda())
        epoch_loss += loss
        loss.backward()
        optimizer.step()


    print("train epoch_loss = ", epoch_loss.data[0])
    dev_epoch_loss = get_loss_for_dataset(dev_iter, model)
    print("dev epoch_loss = ", dev_epoch_loss.data[0])

    if args.early_stopping==1 and (prev_dev_epoch_loss is not None and (dev_epoch_loss.data[0] > prev_dev_epoch_loss.data[0])):
        print("Development loss decreased. Stopping")
        break
    prev_dev_epoch_loss = dev_epoch_loss



actual,preds,sentences = get_output(model,train_iter,datasetutil)
output_path = args.train_output_path
persist_output(output_path,actual,preds,sentences)

actual,preds,sentences = get_output(model,dev_iter,datasetutil)
output_path = args.test_output_path
persist_output(output_path,actual,preds,sentences)

actual,preds,sentences = get_output(model,test_iter,datasetutil)
output_path = args.dev_output_path
persist_output(output_path,actual,preds,sentences)
