from collections import defaultdict, Counter
from string import punctuation
import random
import re


master_file = "../data/all_normalization_data.tsv"
ud_nhi_file = "../../UD_Western_Sierra_Puebla_Nahuatl-ITML/nhi_itml-ud-test.conllu"

with open(master_file) as f:
    master_rows = [line.strip("\n").split("\t") for line in f]

master_uniqs = defaultdict(list)
for row in master_rows:
    t = tuple([s.lower() for s in row ])
    master_uniqs[t].append(row)

train_nhi, dev_nhi, test_nhi = [], [], []
for row in master_uniqs:
    n = random.random()
    if n < 0.8:
        train_nhi.append(row)
    elif 0.8 <= n < 0.9:
        dev_nhi.append(row)
    else:
        test_nhi.append(row)


translations = []
with open(ud_nhi_file) as f:
    txt = f.read()
    for row in txt.split("\n"):
        if row.startswith("# text[spa]"):
            trans = row.split(" = ")
            if len(trans) > 1 and trans[1]:
                translations.append(trans[1])

spanish_tokens = []

for t in translations:
    t = re.sub("[{}]+".format(punctuation), " ", t)
    tokens = t.split()
    tokens = [toke for i, toke in enumerate(tokens) if i == 0 or toke[0].islower()]
    spanish_tokens.extend(tokens)

uniq_spa_tokens = Counter()
for tok in spanish_tokens:
    uniq_spa_tokens[tok] += 1

spanish_data_to_add = []
for tok in uniq_spa_tokens:
    for orth in ("<ack>", "<inali>", "<sep>", "<ilv>"):
        row = [f"{orth} {' '.join(list(tok))}", " ".join(list(tok)), "spa"]
        spanish_data_to_add.append(row)


train = train_nhi + spanish_data_to_add
random.shuffle(train)
with open("../data/train_w_xtra_spa_ALL.tsv", "w") as f:
    f.write("\n".join(["\t".join(r) for r in train]))

random.shuffle(train_nhi)
with open("../data/train.tsv", "w") as f:
    f.write("\n".join(["\t".join(r) for r in train_nhi]))

random.shuffle(dev_nhi)
with open("../data/dev.tsv", "w") as f:
    f.write("\n".join(["\t".join(r) for r in dev_nhi]))

random.shuffle(test_nhi)
with open("../data/test.tsv", "w") as f:
    f.write("\n".join(["\t".join(r) for r in test_nhi]))