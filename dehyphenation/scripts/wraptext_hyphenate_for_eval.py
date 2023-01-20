"""
Wrap an input text, breaking its rows on spaces or
by hyphenating words if possible.
Retain line breaks in the input. Strip spaces at the
beginning and end of input rows.
Tabs in the input are replaced by spaces, and
multiple horizontal whitespace characters in the input
are replaced by a single space.
"""

import os
import argparse
import re
import pyphen
from sys import stdin, stdout

from consts import NO_HYPHEN_LABEL, BREAKING_HYPHEN_LABEL, DIGRAPH_HYPHEN_LABEL, ORTHOGRAPHIC_HYPHEN_LABEL, HYPHEN_PLUS_SPACE_LABEL, DASH_PLUS_SPACE_LABEL, STARTING_HYPHEN_LABEL, DASH_PUNCT_LABEL, HYPHEN_PUNCT_LABEL

NEW_LINE = "\n"
SPACE = " "
BREAKING_HYPHEN = '#'
ORTHOGRAPHIC_HYPHEN = '~'

LISTING_PUNCT = ',;'

def write_code(label, annot_file):
    annot_file.write(label)
    stdout.write(f'\t{{{label}}}')

def main():
    """
    Hyphen class labels are written optionally to an annotation file
    for the last token of every output line.
    Labels from consts.py are used.
    """

    args = get_args()

    hyphenator = pyphen.Pyphen(lang='hu_HU')
    output_cursor = 0

    if args.annotate:
        annot_file = open(args.annotate, 'w')
    else:
        annot_file = open(os.devnull, 'w')

    breaking_hyphen = BREAKING_HYPHEN if args.symbols else "-"
    orthogr_hyphen = ORTHOGRAPHIC_HYPHEN if args.symbols else "-"

    for line in stdin:

        line = line.strip()

        line = re.sub("(\t| )+", " ", line)

        if args.symbols:
            # in order to prevent ambiguity
            line = line.replace(BREAKING_HYPHEN, "")
            line = line.replace(ORTHOGRAPHIC_HYPHEN, "")

        # handle short and empty rows
        if len(line) <= args.maxlength:
            print(line)
            output_cursor += len(line) + 1
            continue

        words = line.split()

        current_length = 0

        while True:
            # PREVIOUS word...
            if words[0] == '-': # ...is a standalone hyphen
                label_acc_prev = DASH_PLUS_SPACE_LABEL
            elif words[0][-1] == '-': # ...ends with a hyphen
                label_acc_prev = HYPHEN_PLUS_SPACE_LABEL
            else: # ...ends with no hyphen
                label_acc_prev = NO_HYPHEN_LABEL

            current_length += len(words[0])
            output_cursor += len(words[0])
            stdout.write(words.pop(0))

            # no more words in input row
            if not words:
                break

            remaining_length = args.maxlength - (current_length + 1)

            # does the next word still fit on the output line?
            if len(words[0]) <= remaining_length:
                stdout.write(SPACE)
                current_length += len(SPACE)
                output_cursor += len(SPACE)
            else:
                hyphen_index = words[0][:remaining_length + 1].rfind("-")
                next_hyphenated = hyphenator.wrap(words[0], remaining_length)

                if remaining_length <= 0:
            # label based on prev word's ending as prev word is the last word in the line
                    write_code(label_acc_prev, annot_file)
            # Is the next word a single hyphen?
            # XXX no example for this -- do we need to check prev word somehow?
                elif words[0] == '-':
                    # move it to the next line
                    write_code(NO_HYPHEN_LABEL, annot_file)
            # Does the next word contain a hyphen that will just fit?
                elif hyphen_index == remaining_length - 1:
                    out = " " + words[0][:hyphen_index] + orthogr_hyphen
                    stdout.write(out)
                    # next word is `-,`
                    if words[0] in {'-' + x for x in LISTING_PUNCT}:
                        write_code(DASH_PUNCT_LABEL, annot_file)
                    # next word begins with a hyphen
                    elif words[0][0] == '-':
                        write_code(STARTING_HYPHEN_LABEL, annot_file)
                    # next word ends with `-,`
                    elif words[0][hyphen_index+1] in LISTING_PUNCT:
                        write_code(HYPHEN_PUNCT_LABEL, annot_file)
                    else:
                        write_code(ORTHOGRAPHIC_HYPHEN_LABEL, annot_file)
                    output_cursor += len(out)
                    words[0] = words[0][hyphen_index + 1:]
            # Can the next word be hyphenated to fit on the output line?
                elif next_hyphenated:
                    if next_hyphenated[0][-2] == '-':
            # Pyphen adds a second hyphen when breaking a word that already
            # contains an orthographic hyphen.
                        out = " " + next_hyphenated[0][:-2] + orthogr_hyphen
                        stdout.write(out)
                        write_code(ORTHOGRAPHIC_HYPHEN_LABEL, annot_file)
                    else:
                        out = " " + next_hyphenated[0][:-1] + breaking_hyphen
                        stdout.write(out)
                        if len(next_hyphenated[0] + next_hyphenated[1]) > \
                                len(words[0]) + 1:
                            write_code(DIGRAPH_HYPHEN_LABEL, annot_file)
                        else:
                            write_code(BREAKING_HYPHEN_LABEL, annot_file)
                    output_cursor += len(out)
                    words[0] = next_hyphenated[1]
                else:
            # The next word will go on the next line.
            # label based on prev word's ending as prev word is the last word in the line
                    write_code(label_acc_prev, annot_file)

                stdout.write(NEW_LINE)
                output_cursor += len(NEW_LINE)
                annot_file.write(':' + str(output_cursor) + ",")
                current_length = 0

        output_cursor += len(NEW_LINE)
        stdout.write(NEW_LINE)

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--maxlength', '-m',
        help='maximum length of wrapped lines',
        type=int,
        default=40
    )
    parser.add_argument(
        '--annotate', '-a',
        help='file name to save hyphen class for the end of each output line',
        type=str
    )
    parser.add_argument(
        '--symbols', '-s',
        help='indicate dehyphenable hyphens by symbols in the output',
        action='store_true'
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()
