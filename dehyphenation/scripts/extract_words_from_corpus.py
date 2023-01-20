"""
Extract counts of word tokens, including hyphenated ones,
from a corpus of plain text Hungarian corpus files, stored
in the specified directory, iterating through any subirectories
recursively.
Save the token counts to a pickle file to be used by the
rule-based dehyphenation algorithm.
"""


import sys
import os
import collections
import pickle
import regex

script_dir = os.path.dirname(__file__)

WORDS_FILE_NAME = script_dir + "/corpus_word_counts.pickle"
HYPHEN_SUFFIXES_FILE_NAME = script_dir + "/hyphen_word_counts.pickle"
DASH_SUFFIXES_FILE_NAME = script_dir + "/dash_word_counts.pickle"


def main():
    corpus_dir = sys.argv[1]

    corpus_word_types = collections.Counter([])
    hyphen_word_types = collections.Counter([])
    dash_word_types = collections.Counter([])

    walk_data = os.walk(corpus_dir)
    file_names = list()
    for dirfiles in walk_data:
        if '.git' not in dirfiles[0] and len(dirfiles[2]):
            file_names.extend([dirfiles[0] + '/' + file_name
                               for file_name in dirfiles[2]])
    print("Total number of files:", len(list(file_names)))

    # Matches sequences of punctuation symbols except dashes and hyphens,
    # as well as all whitespace.
    SPLIT_PATTERN = regex.compile(r'(\s|\p{Pe}|\p{Pf}|\p{Pi}|\p{Po}|\p{Ps})+')

    # Matches maximal substrings that start with a letter and contain
    # only letters possibly zero but at most two hyphen or dash symbols.
    WORD_PATTERN = regex.compile(r"\p{L}+(\p{Pd}\p{L}*){,2}")

    # Matches hyphens surrounded by alphabetic strings on both sides,
    # both directly adjacent to the hyphen.
    HYPHEN_PATTERN = regex.compile(r"\p{L}+\p{Pd}(\p{Ll}{2,})\W")

    # Matches dashes surrounded by alphabetic strings on both sides,
    # separated by a single space from the dash.
    DASH_PATTERN = regex.compile(r"\p{L}+ \p{Pd} (\p{Ll}{2,})\W")

    for i, fn in enumerate(file_names):
        with open(fn, encoding='utf-8') as infile:
            # Split up the corpus file into tokens, collect all resulting
            # word tokens that contain letters and possibly hyphens or dashes.
            doc_text = infile.read()
            corpus_word_types.update(word
                        for word in SPLIT_PATTERN.split(doc_text)
                        if WORD_PATTERN.fullmatch(word))
            hyphen_word_types.update(regex.findall(HYPHEN_PATTERN, doc_text))
            dash_word_types.update(regex.findall(DASH_PATTERN, doc_text))

        if (i + 1) % 100 == 0:
            print("Processed file", str(i+1), "/", len(list(file_names)))
            print("Results so far:")
            print_counter_state(corpus_word_types)
            print()

    print("*****************************************")
    print("Corpus processing completed.")
    print_counter_state(corpus_word_types)
    print()

    print("Sanity check: top 20 most frequent word tokens")
    print(corpus_word_types.most_common(20))

    with open(WORDS_FILE_NAME, "wb") as pickle_file:
        pickle.dump(corpus_word_types, pickle_file)

    with open(HYPHEN_SUFFIXES_FILE_NAME, "wb") as pickle_file:
        pickle.dump(hyphen_word_types, pickle_file)

    with open(DASH_SUFFIXES_FILE_NAME, "wb") as pickle_file:
        pickle.dump(dash_word_types, pickle_file)


def print_counter_state(counter):
    count_total = 0
    for i, elem in enumerate(counter.items()):
        count_total += elem[1]

    print("Total word types in corpus:", len(counter))
    print("Total number of word tokens in corpus:", count_total)


if __name__ == '__main__':
    main()
