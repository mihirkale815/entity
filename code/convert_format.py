import utils
import argparse


if __name__=='__main__':

    parser = argparse.ArgumentParser(description='entity stats')
    parser.add_argument('--output', action="store", dest="output_path",type=str)
    parser.add_argument('--input', action="store", dest="input_path", type=str)
    parser.add_argument('--entity', action="store", dest="entity", type=str)
    args = parser.parse_args()

    sentence_g = utils.sentence_generator(args.input_path)
    fout = open(args.output_path, "w")

    for sentence in sentence_g:
        token_sequence = ["BOS"] + [ token for (token,tag) in sentence ] + ["EOS"]
        tag_sequence = ["O"] + [tag for (token, tag) in sentence] + ["O"]
        fout.write(" ".join(token_sequence) + "\t" + " ".join(tag_sequence) + "\n")

    fout.close()