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


