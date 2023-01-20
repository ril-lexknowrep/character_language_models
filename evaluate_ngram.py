import pickle

import ngram_model
from ngram_model import MultiNgramTrie, MultiNgramTrieRoot, BiMultiNgramModel
from lstm_model import timestamp
import time
from sys import stderr, exit
import sys

with open("test_files/1.press_hu_nem_007.txt", encoding='utf-8') as infile:
    very_long_string = infile.read()

n = sys.argv[1]  # order of the n-gram-model

print(f"Loading {n}-gram model")
with open(f'{n}-gram-forward.model', 'rb') as savefile:
    trie_forward = pickle.load(savefile)
with open(f'{n}-gram-backward.model', 'rb') as savefile:
    trie_backward = pickle.load(savefile)
bi_ngram_model = BiMultiNgramModel(trie_forward, trie_backward)

start = f"Starting evaluation of {n}-gram model at {timestamp()}"
print(start, file=stderr)
print(start)
start_time = time.time()
acc, perpl = bi_ngram_model.metrics_on_string(very_long_string)
end_time = time.time()
print(acc, perpl, file=stderr)

end = f"Completed evaluation of {n}-gram model at {timestamp()}"
print(end, file=stderr)
print(end)
print("Evaluation completed in", end_time - start_time)
print(f"Results for {n=}: {acc=}, {perpl=}")

exit(0)
