
from scipy.stats import entropy

import tensorflow as tf
import lstm_model
from encode_characters import InputEncoder, OutputEncoder, character_to_features

input_enc = InputEncoder()
output_enc = OutputEncoder()
input_enc.load("input_encoder.pickle")
output_enc.load("output_encoder.pickle")

bilstm_model = lstm_model.BiLSTM_Model.load('bilstm_model_512.h5',
                                            input_enc, output_enc)

VERBOSE = False # False -- output for matplotlib :)

ORDER = 15 # model order XXX how to get it from the loaded model?
TH = 0.001 # minimum prob for a tip to consider

TO_ESTIMATE = '● '

print('\n\n\n')

s = "Miután mind a tanításra, mind a predikcióra megvannak a szükséges kényelmi metódusok, általában nem kell bajlódnunk a sztringkódoló használatával. OCR-hiba: nem krumpli, hanem krumpIi. A majornmal hasonlítjuk össze. Legalább 17500 vagy 9734 fajtája van. És ha széttöre dezik a szó?"
# XXX "szőringkódoló" és "sztringhódoló" xdlol
# XXX írásjelek eléggé felcserélhetők: ? vs ,
# XXX nagybetűs rövidítés -- hát ja, bármi lehet kb.
# XXX ott van az (egykarakteres) OCR-hiba, ahol 100%-os tipp van,
#     körülötte meg teljes bizonytalanság!!! EZ FONTOS!
#     -- vagy ez trivi, mert pont ezt mondja meg a perplexitás??? XXX
# XXX töredezettség... nagy zavar a kornyeken
# XXX számok -- barmi lehet
#
# XXX mindig ra kene jonni apatternbol, hogy mit erdemes megprobalni
#     EZT meg lehetne valahogy tanulni?? :) neuralisan?? :)

def vis(char):
    """Make char visible."""
    return '␣' if char == ' ' else char

# data arrays for matplotlib
y_best = []
y_entr = []
x_labels = []


def char_prediction(text, target_index=0):
    """Return best_char, best_prob and entropy for char at `text[target_index]`."""
    res = bilstm_model.estimate_alternatives(text, target_index)

    PROB = 1

    entr = entropy([x[PROB] for x in res.items()])

    tips = list(filter(lambda x: x[PROB] > TH,
        sorted(res.items(), key=lambda x: x[PROB], reverse=True)))
    best, best_prob = tips[0]

    return best, best_prob, entr


for ti in range(len(s)): # ti = target index = we estimate this char

    best, best_prob, entr = char_prediction(s, ti)

    left = max(0, ti - ORDER)
    right = min(len(s), ti + 1 + ORDER)

    msg = "OK" if best == s[ti] else "ERR!"

    padding_size = min(ORDER, ti)
    padding = ' ' * padding_size

    if VERBOSE:
        print(f'{s[left:ti]}{TO_ESTIMATE}{s[ti+1:right]}')
        print(f'{padding}{vis(best)} -- {msg} -- H={entr:.4f}')
        for char, prob in tips:
            print(f'{vis(char)} {prob:.4f}')
        print()
    else:
        curr = s[ti]
        best_print = '' if curr == best else best
        print(f'{vis(curr)}\t{vis(best_print)}\t{best_prob:.4f}\t{entr:.4f}\t{msg}')

        label = curr if curr == best else f'{curr}..{best}'

        y_best.append(best_prob)
        y_entr.append(entr)
        x_labels.append(label)

import matplotlib.pyplot as plt
import numpy as np

LEN = 280

y_best = y_best[0:LEN]
y_entr = y_entr[0:LEN]
x_labels = x_labels[0:LEN]

x = range(len(x_labels))

plt.rcParams["figure.figsize"] = (30, 5)

fig, ax = plt.subplots() # XXX mi a frasz ez a plt / fig / ax ? sose ertem..

ax.plot(x, y_best, y_entr)

plt.xticks(x, x_labels, rotation='vertical', fontsize=6)

ax.set(xlabel='',
    ylabel='entrópia és legvalszg tipp valszge',
    title='kétirányú LSTM modell karakterjóslása')
ax.grid()

fig.savefig("use_lm.png", dpi=300)
# plt.show()

