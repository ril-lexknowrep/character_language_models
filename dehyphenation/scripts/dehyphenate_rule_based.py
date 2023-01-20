"""
Remove line-ending and line-internal hyphens and extra whitespace and OCR junk
following the hyphens from a raw text input file based on a hand-crafted
decision tree and word counts from a large corpus and the words in the input
document.
"""


import sys
import os
import collections
import pickle
import argparse

import hunspell
import regex

from consts import NO_HYPHEN_LABEL, BREAKING_HYPHEN_LABEL, DIGRAPH_HYPHEN_LABEL, ORTHOGRAPHIC_HYPHEN_LABEL, HYPHEN_PLUS_SPACE_LABEL

script_dir = os.path.dirname(os.path.abspath(__file__))

hobj = hunspell.HunSpell(script_dir + '/hu_HU.dic', script_dir + '/hu_HU.aff')

WORD_COUNTER_FILE_NAME = script_dir + "/corpus_word_counts.pickle"

# Matches sequences of punctuation symbols except dashes and hyphens,
# as well as all whitespace.
SPLIT_PATTERN = regex.compile(r'(\s|\p{Pe}|\p{Pf}|\p{Pi}|\p{Po}|\p{Ps})+')

# Matches maximal substrings that start with a letter and contain
# only letters and possibly zero but at most two hyphen or dash symbols.
WORD_PATTERN = regex.compile(r"\p{L}+(\p{Pd}\p{L}*){,2}")

VOWEL_PATTERN = regex.compile("[aeiouáéíóöőúüűAEIOUÁÉÍÓÖŐÚÜŰ]")
CONSONANTS = "bcdfghjklmnpqrstvwxyz"

ITER_PATTERN        = regex.compile(r"(\p{L}+|\d+)\p{Pd}\W*(\p{Ll}+)")
ITER_PATTERN_STRICT = regex.compile(r"(\p{L}+|\d+)\p{Pd}\s*(\p{Ll}+)")
ITER_PATTERN_EOL    = regex.compile(r"(\p{L}+|\d+)\p{Pd}\n+(\p{Ll}+)")

DIGRAPHS = {
    'sz': 'ssz',
    'zs': 'zzs',
    'cs': 'ccs',
    'gy': 'ggy',
    'ly': 'lly',
    'ny': 'nny',
    'ty': 'tty',
    'dz': 'ddz'
}

EXCLUDE = {
    'öt': 'öt',
    'hat': 'hat',
}

CONJUNCTIONS = set(["és", "s", "stb", "vagy", "hanem",
                "meg", "illetve", "sőt", "majd",
                "de", "akár", "mind", "sem", "valamint",
                "avagy", "ráadásul", "helyett", "mellett",
                "und"])

def main():
    args = get_args()

    # specify infile's and outfile's name on command line
    infile_name = args.infile
    outfile_name = args.outfile

    EOL_EVAL = args.end_of_line_only
    STRICT = args.strict
    VERBOSE = args.verbose
    # `-e' implies `-s` and ignores `-v`
    if EOL_EVAL:
        STRICT = True
        VERBOSE = False

    doc_text, words_in_doc = count_words(infile_name)

    large_word_list = load_large_word_list()

    search_start_pos = 0

    outfile = open(outfile_name, 'w', encoding='utf-8')

    while search_start_pos < len(doc_text):
        if EOL_EVAL:
            matching_str = ITER_PATTERN_EOL.search(doc_text,
                                                      search_start_pos)
        elif STRICT:
            matching_str = ITER_PATTERN_STRICT.search(doc_text,
                                                      search_start_pos)
        else:
            matching_str = ITER_PATTERN.search(doc_text,
                                               search_start_pos)
        if matching_str is None:
            outfile.write(doc_text[search_start_pos:])
            break

        outfile.write(doc_text[search_start_pos:matching_str.start()])

        if EOL_EVAL:
            first, second = matching_str[0].split('\n')
            # ITER_PATTERN_EOL should contain exactly one \n

        dehyphenated, return_str = dehyphenate(matching_str[0],
                                    large_word_list,
                                    doc_words=words_in_doc,
                                    verbose=VERBOSE,
                                    strict=STRICT)

        if return_str != matching_str[0]:
            if EOL_EVAL:
                outfile.write(f'{first}\t{{{dehyphenated}}}\n{second}')
            elif VERBOSE:
                outfile.write("*" + return_str + "*")
            else:
                outfile.write(return_str)
            search_start_pos = matching_str.end()
        else:
            if EOL_EVAL:
                outfile.write(f'{first}\t{{{dehyphenated}}}\n{second}')
            elif VERBOSE:
                outfile.write("$" + matching_str[1] + "$")
            else:
                outfile.write(matching_str[1])
            search_start_pos = matching_str.end(1)

def load_large_word_list():
    with open(WORD_COUNTER_FILE_NAME, 'rb') as f:
        large_word_list = pickle.load(f)
    return large_word_list

def count_words(file_name):
    words_in_doc = collections.Counter([])

    with open(file_name, encoding='utf-8') as infile:
        doc_text = infile.read()
        words_in_doc.update(word
                for word in SPLIT_PATTERN.split(doc_text)
                if WORD_PATTERN.fullmatch(word))

    return doc_text, words_in_doc


def dehyphenate(string, large_word_list, doc_words=None, verbose=False,
                strict=False):
    """
    Dehyphenate string.
    If the string consists of letters, directly followed by a hyphen,
    and ends in letters, then this function checks whether the letters
    before and after the hyphen can be combined into a non-hyphenated
    valid word.
    Return a 2-tuple (True, '<WORD>') if the hyphen can be removed,
    '<WORD>' being the combined dehyphenated word.
    Return (False, '<WORD>') if the hyphen cannot be removed, but
    '<WORD>' appears to be a valid hyphenated word. Extraneous whitespace
    and other characters appearing in the input string are removed.
    Return (False, string) if the input string does not appear to be a
    valid hyphenated or non-hyphenated word. The input string is
    returned unchanged.
    """

    if doc_words is None:
        doc_words = collections.Counter([])

    if strict:
        e = ITER_PATTERN_STRICT.fullmatch(string)
    else:
        e = ITER_PATTERN.fullmatch(string)
    # XXX maybe we should handle ITER_PATTERN_EOL as well
    # no problem while ITER_PATTERN_STRICT matches everywhere, where ITER_PATTERN_EOL does

    if e is None:
        return (NO_HYPHEN_LABEL, string)

    inp_str = e[0].replace('\n', '\\n').replace('\t', '\\t')

    dehyph_parts = e[1] + e[2]
    hyph_parts = e[1] + "-" + e[2]
    hyph_space_parts = e[1] + "- " + e[2]

    if "-, " in e[0] and (hobj.spell(e[2]) or large_word_list[e[2]] > 2):
        if verbose:
            printt("mellérendelő vesszővel:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, string)
    elif "- " in e[0] and e[2] in CONJUNCTIONS:
        if verbose:
            printt("mellérendelő:", hyph_space_parts)
        return (HYPHEN_PLUS_SPACE_LABEL, string)
    elif e[1].isupper():
        if verbose:
            printt("nagybetűs előtag:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif e[1].isnumeric():
        if verbose:
            printt("szám előtag:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif len(e[2]) == 1 and e[2] in CONSONANTS:
        if verbose:
            printt("msh utótag:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif e[1] == "e" and hobj.spell(e[2]):
        if verbose:
            printt("e-összetétel:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif hobj.spell(e[1]) and e[2] == 'e':
        if verbose:
            printt("-e kérdőszó:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif e[1] in EXCLUDE and e[2] == EXCLUDE[e[1]]:
        if verbose:
            printt("kivétel:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif e[1][-2:] in DIGRAPHS and e[2][:2] == e[1][-2:]:
        combo = e[1][:-2] + DIGRAPHS[e[1][-2:]] + e[2][2:]
        if hobj.spell(combo):
            if verbose:
                printt("hosszú digráf", inp_str, combo)
            return (DIGRAPH_HYPHEN_LABEL, combo)
        else:
            if verbose:
                printt("dupla digráf", inp_str)
            return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif large_word_list[hyph_parts] > large_word_list[dehyph_parts]:
        if verbose:
            printt("gyakrabban szerepel a korpuszban kjellel:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif (large_word_list[dehyph_parts] > large_word_list[hyph_parts]
          or large_word_list[e[1] + e[2] + '-'] > large_word_list[hyph_parts]
          ):
        if verbose:
            printt("gyakrabban szerepel a korpuszban kjel nélkül:",
                  inp_str, dehyph_parts)
            if check_long_compound(e, hobj):
                printt("túl sok szótag 2:", inp_str)
        return (BREAKING_HYPHEN_LABEL, dehyph_parts)
    elif doc_words[dehyph_parts] > doc_words[hyph_parts]:
        if verbose:
            printt("gyakrabban szerepel a szövegben kjel nélkül:",
                  inp_str, dehyph_parts)
            if check_long_compound(e, hobj):
                printt("túl sok szótag 1:", inp_str)
        return (BREAKING_HYPHEN_LABEL, dehyph_parts)
    elif (e[1][0].isupper()
        and e[1].lower() not in large_word_list
        and (e[1] in large_word_list
             or hobj.spell(e[1]))
        and hobj.spell(e[2])):
            if verbose:
                printt("név előtag:", inp_str)
            return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif hobj.spell(dehyph_parts):
        if verbose:
            printt("hunspell szerint jó kjel nélkül:",
                    inp_str, dehyph_parts)
                    # bár sokszor téved
        return (BREAKING_HYPHEN_LABEL, dehyph_parts)
#    elif hobj.spell(hyph_parts):
#        if verbose:
#            printt("hunspell szerint jó kötőjellel:", inp_str)
#        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)
    elif (not strict and "m" in dehyph_parts
        # elég gyakori OCR-hiba
          and (hobj.spell((dehyph_parts).replace("m", "rn"))
          or (dehyph_parts).replace("m", "rn") in doc_words
          or (dehyph_parts).replace("m", "rn") in large_word_list)):
        if verbose:
            printt("m -> rn kjel nélkül:", inp_str,
                   (dehyph_parts).replace('m', 'rn'))
        return (BREAKING_HYPHEN_LABEL, (dehyph_parts).replace("m", "rn"))
    elif (not strict and "m" in hyph_parts
        # elég gyakori OCR-hiba
          and (hobj.spell((hyph_parts).replace("m", "rn"))
          or (hyph_parts).replace("m", "rn") in doc_words
          or (hyph_parts).replace("m", "rn") in large_word_list)):
        if verbose:
            printt("m -> rn kjellel:", inp_str,
                   (hyph_parts).replace('m', 'rn'))
        return (ORTHOGRAPHIC_HYPHEN_LABEL, (hyph_parts).replace("m", "rn"))
    elif (regex.search("\s", e[0])):
        if verbose:
            printt("maradék whitespace-es:", inp_str)
        return (HYPHEN_PLUS_SPACE_LABEL, hyph_space_parts)
    else:
        if verbose:
            printt("maradék kötőjeles:", inp_str)
        return (ORTHOGRAPHIC_HYPHEN_LABEL, hyph_parts)


def check_long_compound(match_obj, speller_obj):
    if (speller_obj.spell(match_obj[1])
            and speller_obj.spell(match_obj[2])
            and len(VOWEL_PATTERN.findall(match_obj[1] + match_obj[2])) >= 7):
        return True


def printt(*args):
    print('\t'.join(args))


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'infile',
        help='name of hyphenated input text file',
        type=str
    )
    parser.add_argument(
        'outfile',
        help='name of dehyphenated output text file',
        type=str
    )
    parser.add_argument(
        '--verbose', '-v',
        help='indicate dehyphenable hyphens by symbols in the output',
        action='store_true'
    )
    parser.add_argument(
        '--strict', '-s',
        help=('do not attempt to filter out junk OCR characters between '
              + 'the hyphen and the next word or correct OCR mistakes'),
        action='store_true'
    )
    parser.add_argument(
        '--end_of_line_only', '-e',
        help='dehyphenate only around newlines, not within lines + output labels for evaluation, implies `-s`, ignores `-v`',
        action='store_true'
    )
    return parser.parse_args()


if __name__ == '__main__':
    main()
