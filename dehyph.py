"""
Use biLSTM for dehyphenation.
"""

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

TH = 0.001 # minimum prob for a tip to consider

    
def vis(char): 
    """Make char visible."""
    return '␣' if char == ' ' else char


def char_prediction(text, target_index=0):
    """Return best_char, best_prob, entropy and tips for char at `text[target_index]`."""
    res = bilstm_model.estimate_alternatives(text, target_index)

    PROB = 1

    entr = entropy([x[PROB] for x in res.items()])

    tips = list(filter(lambda x: x[PROB] > TH,
        sorted(res.items(), key=lambda x: x[PROB], reverse=True)))
    best, best_prob = tips[0]

    return best, best_prob, entr, tips


VERBOSE = False

def main():
    """Main."""

    texts = [
        "döbbenetes élményeire. A fut- ballpálya négyszögében létünk",
        "színvonaláról, az szíveskedjen egyszer- kétszer elolvasni a személyes",
        "annyi éven át én és a többi másod- és harmadrendű statiszta ebben",
        "hogy a körfrekvencián a másod- percenkénti radiánokat értjük",
    ]

    variations = []
    for text in texts:
        variations.append(text)                    # fut- ballpálya
        variations.append(text.replace('- ', '-')) # fut-ballpálya
        variations.append(text.replace('- ', ''))  # futballpálya

    for text in variations:
        print()
        print(text)

        # XXX mi a jó mérőszám?
        sum_entr = 0 # karakterenkénti entrópia összege
        cnt_err = 0 # rossz előrejelzések száma

        for ti, char in enumerate(text): # ti = target index = we estimate this char

            best, best_prob, entr, tips = char_prediction(text, ti)

            if best != char:
                cnt_err += 1

            sum_entr += entr

            if VERBOSE:
                if best == char:
                    msg, best_print = "OK", ''
                else:
                    msg, best_print = "ERR!", best
                print(f'{vis(char)}\t{vis(best_print)}\t{best_prob:.4f}\t{entr:.4f}\t{msg}')
                #print(f' {tips}')

        print(f'sum_entr={sum_entr}')
        print(f'cnt_err={cnt_err}')


if __name__ == '__main__':
    main()

