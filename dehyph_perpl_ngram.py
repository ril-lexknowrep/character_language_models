"""
Use n-gram model for dehyphenation.
"""

import argparse
import os
import re
import sys

import itertools
import more_itertools
from contextlib import redirect_stdout
import pickle

import ngram_model
from ngram_model import MultiNgramTrie, MultiNgramTrieRoot, BiMultiNgramModel


def disable_print():
    """Use to switch off printing, usually before function call."""
    sys.stdout = open(os.devnull, 'w')


def enable_print():
    """Use to switch on printing, usually after function call."""
    sys.stdout = sys.__stdout__


def perplexity(model, text):
    """Return perplexity."""
    PERPL_INDEX = 1

    with redirect_stdout(sys.stderr):
        res = model.metrics_on_string(text)[PERPL_INDEX]

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
    EVAL = args.eval

    # --- init model
    MODEL_FORWARD = '5-gram-forward.model'
    MODEL_BACKWARD = '5-gram-backward.model'
    with open(MODEL_FORWARD, 'rb') as savefile:
        trie_forward = pickle.load(savefile)
    with open(MODEL_BACKWARD, 'rb') as savefile:
        trie_backward = pickle.load(savefile)

    bi_ngram_model = BiMultiNgramModel(trie_forward, trie_backward)

    MIN_CONTEXT = bi_ngram_model.forward_model.n
    PADDING = [bi_ngram_model.forward_model.PADDING] * (MIN_CONTEXT - 1)

    # context width to be evaluated on either side of target
    EVALUATE_CONTEXT = 12

    # --- parameters
#    CS = 20 # = left context size = right context size
            # min 15 kell, hogy legyen, a modell rendje miatt
            # XXX 16 esetén nem tökéletes eredményt ad...
            # XXX 5-tel hogy tudom megpróbálni??? ERROR: "Too short"

    CS = MIN_CONTEXT + EVALUATE_CONTEXT

    # define target point -- the 1st is the potential error to fix!
    # XXX should be updated to dict format (see below)
#    REPLACEMENTS = [' ', ''] # fragm -- fail XXX too frequent!
#    REPLACEMENTS = ['e', 'é'] # diacr -- fail XXX too frequent!
#    REPLACEMENTS = ['ö', 'ő'] # diacr -- success
#    REPLACEMENTS = ['i', 'í'] # diacr -- success
#    REPLACEMENTS = ['l', 'll'] # spelling -- success
#    REPLACEMENTS = ['j', 'ly'] # spelling -- success

#    TARGET = re.compile(REPLACEMENTS[0])
#    TARGET_LENGTH = len(REPLACEMENTS[0])

    # XXX labels from dehyphenation repo / scripts/consts.py
    BREAKING_HYPHEN_LABEL = "1"
    DIGRAPH_HYPHEN_LABEL = "2"
    ORTHOGRAPHIC_HYPHEN_LABEL = "3"
    HYPHEN_PLUS_SPACE_LABEL = "4"
    DASH_PLUS_SPACE_LABEL = "5"

    # dehyphenation -- regexes needed because of digraphs
    REPLACEMENTS = {
        r'\1- ': HYPHEN_PLUS_SPACE_LABEL,
        r'\1-': ORTHOGRAPHIC_HYPHEN_LABEL,
        r'\1': BREAKING_HYPHEN_LABEL,
        '': DIGRAPH_HYPHEN_LABEL
    }

    # asz- szony / asz-szony / aszszony / asszony
    TARGET = re.compile(r'(.)-\n')  # handle hyphens at end of line
    TARGET_LENGTH = 3  # real length in chars!

    # --- stdin as char stream
    chars_iter = stdinchars()

    # padding
    padded_text = itertools.chain(
        # the specific character is completely unimportant, since the
        # EVALUATE CONTEXT first characters are never processed by the LSTM:
        [' '] * EVALUATE_CONTEXT,  # this must be added, since window_iter is based on CS, not MIN_CONTEXT
        PADDING,  # adds MIN_CONTEXT left padding, as required by the model
        chars_iter,
        PADDING,
        [' '] * EVALUATE_CONTEXT)

    # sliding window on padded char stream
    window_iter = more_itertools.windowed(padded_text, 2 * CS + TARGET_LENGTH)

    DIGRAPHS = {'cs', 'dz', 'gy', 'ly', 'ny', 'sz', 'ty', 'zs'}

    for chars in window_iter:
        text = ''.join(chars)

        targettext = text[CS:CS+TARGET_LENGTH]

        # skip if no TARGET here
        if not TARGET.match(targettext):
            print(text[CS], end='')
            continue

        variations = []

        two_chars_before_hyphen = text[CS-1:CS+1]
        two_chars_after_newline = text[-CS:-(CS-2)]

        if targettext[0] == ' ':
            replaced = targettext.replace('\n', ' ')
            vari = text[:CS] + replaced + text[CS+TARGET_LENGTH:]
            perpl = perplexity(bi_ngram_model, vari)
            variations.append([replaced, vari, perpl,
                               DASH_PLUS_SPACE_LABEL])
        else:
            for repl, label in REPLACEMENTS.items():
                if (label == DIGRAPH_HYPHEN_LABEL
                    and (two_chars_before_hyphen not in DIGRAPHS
                         or two_chars_after_newline not in DIGRAPHS
                         or (two_chars_after_newline
                             != two_chars_before_hyphen))):
                    continue
                # replace only in target position
                replaced = TARGET.sub(repl, targettext)
                vari = text[:CS] + replaced + text[CS+TARGET_LENGTH:]
                perpl = perplexity(bi_ngram_model, vari)
                variations.append([replaced, vari, perpl, label])

        REPLACED_INDEX, VARI_INDEX, PERPL_INDEX, LABEL_INDEX = 0, 1, 2, 3

        if VERBOSE:
            print(']')
            for _, vari, perpl, _ in variations:
                print()
                print(f' vari="{vari}"')
                print(f' mos_perpl={perpl}')
            print()
            print('[', end='')

        best = min(variations, key=lambda x: x[PERPL_INDEX])
        # XXX valszeg nem kéne változtatni, ha nagyon kicsi
        #     perpl-k eltérése, mondjuk <10% vagy ilyesmi...

        if not EVAL:
            print(best[REPLACED_INDEX], end='')
        else: # EVAL
            # XXX gigahekk, kézzel szedem le az újsort,
            #     amit a TARGET ptn megtalált,
            #     ezzel teljesen elrontva a TARGET általánosságát.
            #     persze itt az adott esetben éppen jó! :)
            target_without_newline = targettext.rstrip('\n')
            print(f'{target_without_newline}\t{{{best[LABEL_INDEX]}}}')

        # skip chars processed as part of TARGET
        for i in range(TARGET_LENGTH - 1):
            try:
                next(window_iter)
            except StopIteration:
                print("Reached end of file and ADDED AN INCORRECT LABEL TO THE END OF THE LAST LINE!!!", file=sys.stderr)

    print()


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '-v', '--verbose',
        help='verbose output',
        action='store_true'
    )
    parser.add_argument(
        '-e', '--eval',
        help='output labels for evaluation',
        action='store_true'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
