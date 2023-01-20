
from ngram_model import MultiNgramTrieRoot
from encode_characters import OutputEncoder

output_enc = OutputEncoder()
output_enc.load('output_encoder.pickle')

x = MultiNgramTrieRoot(output_enc, 2)
x.add_input('babababc')


def sort_by_value(dict, n=5):
    s = sorted(dict.items(), key=lambda x: -x[1])[:n]
    return s


def print_trie(trie, level=0):
    print(f'/{trie.total}')
    for t in trie.children.items():
        if level == 0: print()

        indent = "=" * level
        char = t[0] if t[0] != "\u0000" else '@' # padding

        print(f'{indent}{char}', end='')
        print_trie(t[1], level+1)

print_trie(x)


def eval_by_char(model, s):
    for i in range(5, len(s)+1):
        ss = s[0:i]
        print(f'[{ss}]')
        print(model.metrics_on_string(ss))
        print()
        print(model.predict_next(ss, m=5))

eval_by_char(trie_forward, "Ez a szöveg most már nem túl rövid.")


def bi_eval_by_char(model, s, order=5):
    for i in range(2*order-1, len(s)+1):
        left = s[i-(2*order-1):i-order]
        right = s[i-(order-1):i]
        print(f'[{left}] [{right}]')
        #print(model.metrics_on_string(s[0:i])) # index ???
        print(model.forward_model.predict_next(left))
        print(model.backward_model.predict_next(right[::-1]))
        print(model.predict_next(left, right[::-1]))
        print()

b = BiMultiNgramModel(trie_forward, trie_backward)
bi_eval_by_char(b, "Ez a szöveg most már nem túl rövid.")


# :)
def generate(s=trie_forward.char_encoder.PADDING_CHAR, size=50, which=2);
    for i in range(size):
        s += trie_forward.predict_next(s, m=which)[-1]
        print(s)

