import json
import utils
from collections import defaultdict
import argparse

if __name__ == '__main__' :
    parser = argparse.ArgumentParser(description='entity stats')
    parser.add_argument('--input', action="store", dest="input",type=str)
    parser.add_argument('--slot_types', action="store", dest="slot_types", type=str)
    parser.add_argument('--output', action="store", dest="output", type=str)
    args = parser.parse_args()


slot_types = json.load(open(args.slot_types))
fout = open(args.output,"w")
sentence_g = utils.sentence_generator(args.input)

for tagged_sentence in sentence_g:
    tagged_sentence = [ (token.lower(),tag.lower()) for (token,tag) in tagged_sentence]
    tokens = [token for (token, _) in tagged_sentence]
    tags = [tag for (_, tag) in tagged_sentence]
    slot_dict = defaultdict(list)

    for slot_type in slot_types:
        chunk_g = utils.extract_chunks(tagged_sentence, slot_type)
        for text,indices in chunk_g:
            if text == '' : continue
            indices = indices.split("-")
            start_idx = int(indices[0])
            end_idx = int(indices[-1])


            slot_dict[slot_type].append((text,start_idx,end_idx))
    output = [" ".join(tokens),slot_dict]
    fout.write(json.dumps(output)+"\n")

fout.close()

