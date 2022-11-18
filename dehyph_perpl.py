"""
Use biLSTM for dehyphenation.
"""

import argparse
import os
import re
import sys

import itertools
import more_itertools

from scipy.stats import entropy

import tensorflow as tf
import lstm_model
from encode_characters import InputEncoder, OutputEncoder, character_to_features


def disable_print():
    """Use to switch off printing, usually before function call."""
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Use to switch on printing, usually after function call."""
    sys.stdout = sys.__stdout__


def perplexity(model, text):
    """Return perplexity."""
    PERPL_INDEX = 1

    disable_print()
    res = model.metrics_on_string(text)[PERPL_INDEX]
    enable_print()

    return res


def stdinchars():
    """Read by char from stdin."""
    for line in sys.stdin:
        yield from line


def eprint(*args, **kwargs):
    """Print to stderr."""
    # https://stackoverflow.com/a/14981125/8746466 
    print(*args, file=sys.stderr, **kwargs)


def main():
    """Main."""

    # XXX ez a script egyelőre "futtatás" = kiírja a tippjét
    # XXX utána jön a kiértékelés, amikor összevetjük a golddal
    # XXX itt szóbajön a difflib.SequenceMatcher(autojunk=False)

    args = get_args()
    VERBOSE = args.verbose

    # --- init model
    input_enc = InputEncoder(file="input_encoder.json")
    output_enc = OutputEncoder(file="output_encoder.json")
    disable_print()
    MODEL_FILENAME = "bilstm_model_512.h5"
    try:
        bilstm_model = lstm_model.BiLSTM_Model.load(
            MODEL_FILENAME, input_enc, output_enc)
    except RuntimeError as e:
        eprint()
        eprint(e)
        eprint(f"\nModelfile ({MODEL_FILENAME})'s version differs, loading with check_version=False")
        bilstm_model = lstm_model.BiLSTM_Model.load(
            MODEL_FILENAME, input_enc, output_enc,
        check_version=False)
    enable_print()

    # --- parameters
    CS = 20 # = left context size = right context size
            # min 15 kell, hogy legyen, a modell rendje miatt
            # XXX 16 esetén nem tökéletes eredményt ad...
            # XXX 5-tel hogy tudom megpróbálni??? ERROR: "Too short"
    # XXX how to define padding to be consistent with BiLSTM_Model?
    PADDING = ' ' * CS

    # define target point -- the 1st is the potential error to fix!
#    REPLACEMENTS = [' ', ''] # fragm -- fail XXX too frequent!
#    REPLACEMENTS = ['e', 'é'] # diacr -- fail XXX too frequent!
#    REPLACEMENTS = ['ö', 'ő'] # diacr -- success
#    REPLACEMENTS = ['i', 'í'] # diacr -- success
#    REPLACEMENTS = ['l', 'll'] # spelling -- success
#    REPLACEMENTS = ['j', 'ly'] # spelling -- success

#    TARGET = re.compile(REPLACEMENTS[0])
#    TARGET_LENGTH = len(REPLACEMENTS[0])

    # dehyphenation -- regexes needed because of digraphs
    REPLACEMENTS = [r'\1- ', r'\1-', r'\1', '']
    # asz- szony / asz-szony / aszszony / asszony
    TARGET = re.compile(r'(.)- ')
    TARGET_LENGTH = 3 # real length in chars!
        # XXX how to handle '-\n' as well???

    # --- stdin as char stream
    chars_iter = stdinchars()

    # padding
    padded_text = itertools.chain(PADDING, chars_iter, PADDING)

    # sliding window on padded char stream
    window_iter = more_itertools.windowed(padded_text, 2 * CS + 1)

    for chars in window_iter:
        text = ''.join(chars)

        targettext = text[CS:CS+TARGET_LENGTH]

        # skip if no TARGET here
        if not TARGET.match(targettext):
            print(text[CS], end='')
            continue

        variations = []
        for repl in REPLACEMENTS:
            # replace only in target position
            replaced = TARGET.sub(repl, targettext)
            vari = text[:CS] + replaced + text[CS+TARGET_LENGTH:]
            perpl = perplexity(bilstm_model, vari)
            variations.append([replaced, vari, perpl])

        FOUND_INDEX, VARI_INDEX, PERPL_INDEX = 0, 1, 2

        if VERBOSE:
            print(']')
            for _, vari, perpl in variations:
                print()
                print(f' vari="{vari}"')
                print(f' mos_perpl={perpl}')
            print()
            print('[', end='')

        best = min(variations, key=lambda x: x[PERPL_INDEX])[FOUND_INDEX]
        # XXX valszeg nem kéne változtatni, ha nagyon kicsi
        #     perpl-k eltérése, mondjuk <10% vagy ilyesmi...
        print(best, end='')

        # skip chars processed as part of TARGET
        for i in range(TARGET_LENGTH - 1):
            next(window_iter)

    print()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # boolean argument: True if present, False if not present
    parser.add_argument(
        '-v', '--verbose',
        help='verbose output',
        action='store_true'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    main()

