from utils import sentence_generator
import random
import argparse

def write_to_file(path,sentences):
    f = open(path,"w")
    for sentence in sentences :
        for (word,tag) in sentence:
            f.write(tag + "\t" + word + "\n")
        f.write("\n")
    f.close()


if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='entity stats')
    parser.add_argument('--dev', action="store", dest="dev_path",type=str)
    parser.add_argument('--train', action="store", dest="train_path", type=str)
    parser.add_argument('--new_train', action="store", dest="new_train_path", type=str)
    parser.add_argument('--num_dev_samples', action="store", dest="num_dev_samples", type=int)
    args = parser.parse_args()

    sentence_g = sentence_generator(args.train_path)
    sentences = [sentence for sentence in sentence_g]
    random.shuffle(sentences)
    train_sentences = sentences[0:-args.num_dev_samples]
    dev_sentences = sentences[-args.num_dev_samples:]
    write_to_file(args.new_train_path,train_sentences)
    write_to_file(args.dev_path, dev_sentences)

