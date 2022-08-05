import sys
import os
import difflib
from types import SimpleNamespace
from typing import Set, List

import tensorflow as tf
import lstm_model
from encode_characters import InputEncoder, OutputEncoder

INPUT_ENCODER_FILE = "input_encoder.pickle"
OUTPUT_ENCODER_FILE = "output_encoder.pickle"
LANGUAGE_MODEL = 'bilstm_model_512.h5'

# Az alábbit lényegében innen loptam, de átírtam, kiegészítettem:
# https://github.com/tamvar/OCR-cleaning/blob/master/arcanum_lanc/py3bs/nlp/corr/spelling.py
OCR_MISTAKES = [
#    ("a", "á", "ä", "à", "u", "ú", "ű", "ü", "û"),
#    ("i", "l", "1", "f", "í", "j", "t", "r"),
#    ("u", "ú", "ű", "ü", "û", "ù", "v", "y", "ý", 'n'),
    {"ű", "ü", "ii", "ll", "u", '11'},
    {"lc", "k", "lr", "ic", "h"},
    {"k", 'h', "lt", 'li'},
#    ("ii", "n"),
    {"o", "ó", "ö", "ő", "ô", "õ", "e", "é", "c"},
    {"rn", "m", "nn", "in", "ln", "jn", "fn"},
    {"ivi", "íví", "m"},
    {"g", "q", "y"},
    {"ó'", "ő", 'o\'', 'ó\''},
    {"w", "v"},
    {"b", "lo", "lc", "le"},
    {"d", "ol", "cl", "oi", "ci"},
    {"d", "tl"},
    {"c", "cz"},
    {"cs", "ts"},
    {"6", "ő", "ó", "ö"},
    {'1', 'I', 'l', '/'},
    {'i', 'í'},
    {'Y', 'A'},
    {'f', 'ri', 'n'},
    {'"', '”'}
]

# flatten OCR_MISTAKES
MISTAKE_CANDIDATES = {e for alt_set in OCR_MISTAKES for e in alt_set}

IGNORE_DELETE_THRESHOLD = 10
ACCEPT_INSERT_THRESHOLD = 10

START_LETTER = ord('c')
MAX_BEAMS = 5
BEAM_MAX_PERPLEXITY = 5
MEASURE_RIGHT_CONTEXT = 1

DEBUG = False


def main():
    '''
    Compare the two files specified on the command line and
    output a corrected version as a result of combining the
    good bits from both files.
    If there are longer insertions/deletions in one or both
    of the files, the version in the first file is ignored
    and that in the second file is accepted automatically for
    these segments. The same applies to replacements of very
    short segments in the first text by substantially longer
    segments in the second text (which are thus regarded as
    insertions), and to replacements of much longer segments
    in the first text by shorter segments in the second text
    (which are regarded as deletions).
    What counts as (substantially) 'longer' in this respect
    is determined by IGNORE_DELETE_THRESHOLD and
    ACCEPT_INSERT_THRESHOLD respectively.
    '''

    tf.config.set_visible_devices([], 'GPU')

    if DEBUG:
        debug_file = sys.stdout
    else:
        debug_file = open(os.devnull, 'w')

    input_enc = InputEncoder()
    output_enc = OutputEncoder()
    input_enc.load(INPUT_ENCODER_FILE)
    output_enc.load(OUTPUT_ENCODER_FILE)

    bilstm_model = lstm_model.BiLSTM_Model.load(LANGUAGE_MODEL,
                                                input_enc, output_enc)
    left_window = bilstm_model.encoder.left_context
    right_window = bilstm_model.encoder.right_context

    with open(sys.argv[1], encoding='utf-8') as infile:
        text_a = infile.read().replace('\t', ' ').rstrip()
    with open(sys.argv[2], encoding='utf-8') as infile:
        text_b = infile.read().replace('\t', ' ').rstrip()
    texts = SimpleNamespace()
    texts.a = text_a
    texts.b = text_b

    output_file = open(sys.argv[3], 'w', encoding='utf-8')
    log_file = open(sys.argv[3] + '.log', 'w', encoding='utf-8')

    matcher = difflib.SequenceMatcher(autojunk=False)
    matcher.set_seqs(text_a, text_b)
    opcodes = matcher.get_opcodes()

    segs = [DiffSegment(*opcode, text_a, text_b) for opcode in opcodes]

    beam_start = 0
    beams = [Beam(beam_start, segs)]
    for i, seg in enumerate(segs):

        print(seg, file=log_file)

        if (seg.tag in ('equal', 'insert', 'delete')
                and len(seg.b) >= left_window):
            # write best beam to output and flush beams
            beam_text = str(beams[0]) + seg.b
            beam_start += len(beams[0]) + 1
            print("Writing A:", beam_text[:-left_window], file=debug_file)
            output_file.write(beam_text[:-left_window])
            print(f"text: {beam_text}; start: {beam_start},",
                  f"prefix: {beam_text[-left_window:]}", file=debug_file)
            beams = [Beam(beam_start, segs, path='',
                          prefix=beam_text[-left_window:])]

        elif (seg.tag == 'equal'
              or (seg.tag == 'delete'
                  and len(seg.a) > IGNORE_DELETE_THRESHOLD)
              or (seg.tag == 'insert'
                  and len(seg.b) > ACCEPT_INSERT_THRESHOLD)
              or (seg.tag == 'replace'
                  and (len(seg.a) > len(seg.b) + IGNORE_DELETE_THRESHOLD))
              or (seg.tag == 'replace'
                  and (len(seg.b) > len(seg.a) + ACCEPT_INSERT_THRESHOLD))):
            for i, beam in enumerate(beams):
                beams[i].path += '_'
            print("_ segment:", str(beams[0]), file=debug_file)

        else:
            beam_to_perpl = {}
            alternatives = seg.find_alternatives()

            for beam in beams:
                left_context = str(beam)
                left_annotated = beam.annotated_str()
                if len(left_context) < left_window:
                    # add sufficient left padding
                    left_context = (input_enc.PADDING_CHAR
                                    * (left_window - len(left_context))
                                    + left_context)

                right_contexts = {texts.a[seg.a_end:
                                          seg.a_end + right_window
                                          + MEASURE_RIGHT_CONTEXT],
                                  texts.b[seg.b_end:
                                          seg.b_end + right_window
                                          + MEASURE_RIGHT_CONTEXT]}

                if any(len(rc) < right_window for rc in right_contexts):
                    # add sufficient right padding
                    right_contexts = map(
                        lambda x: x + (input_enc.PADDING_CHAR
                                       * (right_window - len(x)
                                          + MEASURE_RIGHT_CONTEXT)),
                        right_contexts)

                for key, alt in alternatives.items():
                    right_perplexities = []
                    for right_ctxt in right_contexts:
                        eval_string = left_context + alt + right_ctxt
                        perplexity = bilstm_model\
                            .metrics_on_string(eval_string)[1]
                        print('\t'.join([beam.path + key,
                                         str(perplexity),
                                         left_annotated + '{' + alt + '}'
                                         + right_ctxt]),
                              file=log_file)
                        right_perplexities.append(perplexity)
                    beam_to_perpl[beam.path + key] = min(right_perplexities)

            print("Before filter:", file=debug_file)
            sorted_beams = sorted(beam_to_perpl,
                                  key=lambda x: beam_to_perpl[x])
            for b in sorted_beams:
                print(b, beam_to_perpl[b], file=debug_file)

            if beam_to_perpl[sorted_beams[0]] > BEAM_MAX_PERPLEXITY:
                sorted_beams = [sorted_beams[0]]
            else:
                sorted_beams = list(
                    filter(lambda x: beam_to_perpl[x] < BEAM_MAX_PERPLEXITY,
                           sorted_beams))

            print("After filter:", file=debug_file)
            for b in sorted_beams:
                print(b, beam_to_perpl[b], file=debug_file)

            if len(sorted_beams) == 1:
                beams = [Beam(beam_start, segs, sorted_beams[0],
                              beams[0].prefix)]
                print("Beam string:", str(beams[0]), file=debug_file)
                if len(str(beams[0])) > left_window:
                    output_file.write(str(beams[0])[:-left_window])
                    beam_start += len(beams[0])
                    beams = [Beam(beam_start, segs, '',
                                  str(beams[0])[-left_window:])]
                continue

            for k, seg_letter in enumerate(sorted_beams[0]):
                if any(path[k] != seg_letter for path in sorted_beams):
                    first_diff = k
                    break

            # consolidate beams

            print("first diff:", first_diff, file=debug_file)

            if first_diff > 0:
                # common prefix of all beams
                common_prefix = str(Beam(beam_start, segs,
                                         sorted_beams[0][:first_diff],
                                         beams[0].prefix))
                print("Beams:", file=debug_file)
                for path in sorted_beams:
                    print(path, str(Beam(beam_start + 1, segs,
                                         path[1:first_diff],
                                         beams[0].prefix)),
                          file=debug_file)

                print("Writing B:", common_prefix[:-left_window],
                      file=debug_file)
                # write beginning of common prefix to output
                output_file.write(common_prefix[:-left_window])

                # keep necessary part of common prefix
                beams[0].prefix = common_prefix[-left_window:]

                # remove identical segments from all beams
                # up to but not including the first differring segment
                sorted_beams = [path[first_diff:] for path in sorted_beams]
                beam_start += first_diff

            print(sorted_beams, file=debug_file)

            beams = [Beam(beam_start, segs, path, beams[0].prefix)
                     for path
                     in sorted_beams[:MAX_BEAMS]]

            for i, b in enumerate(beams):
                print(i, b, file=debug_file)

    print(beams[0], file=output_file)
    output_file.close()
    log_file.close()


class DiffSegment:
    '''A class that provides a comfortable interface for difflib diffs.'''

    def __init__(self, tag: str,
                 a_start: int, a_end: int,
                 b_start: int, b_end: int,
                 a_text: str, b_text: str):
        self.tag = tag
        self.a_start = a_start
        self.a_end = a_end
        self.b_start = b_start
        self.b_end = b_end
        self.alternatives = {'a': a_text[a_start:a_end],
                             'b': b_text[b_start:b_end]}
        self._extended_alternatives = None

    def find_alternatives(self) -> Set[str]:
        '''
        Suggest further alternatives based on known OCR error patterns
        in addition to the two matching blocks that were found
        in the input files.
        '''
        if self._extended_alternatives is None:
            alternatives_set = {self.a, self.b}
            further_alts = set()

            self._extended_alternatives = self.alternatives

            if alternatives_set & MISTAKE_CANDIDATES:  # non-empty intersection
                for ocr_set in OCR_MISTAKES:
                    if alternatives_set & ocr_set:
                        further_alts |= ocr_set - alternatives_set
                for i, alt in enumerate(further_alts):
                    self._extended_alternatives[chr(START_LETTER + i)] = alt

        return self._extended_alternatives

    def __getitem__(self, item):
        if item == '_':
            return self.alternatives['b']
        elif (self._extended_alternatives is not None
              and item in self._extended_alternatives):
            return self._extended_alternatives[item]
        else:
            return self.alternatives[item]

    def __getattr__(self, attr):
        if attr in self.alternatives:
            return self.alternatives[attr]
        else:
            raise AttributeError()

    def __str__(self):
        msg = f'\n{self.tag}'

        if self.tag == 'equal':
            msg += (f'\n{self.a_start}..{self.a_end} / '
                    + f'{self.b_start}..{self.b_end} = {self.a}')
        else:  # replace, insert, delete
            msg += (f'\n{self.a_start}..{self.a_end} = {{{self.a}}}'
                    + f'\n{self.b_start}..{self.b_end} = {{{self.b}}}')

        return msg


class Beam:
    '''
    A class that provides a comfortable interface for handling search beams.
    '''

    def __init__(self, start_segment: int, segments: List[DiffSegment],
                 path: str = '', prefix: str = ''):
        self.start_segment = start_segment
        self.segments = segments
        self.path = path
        self.prefix = prefix

    def __len__(self):
        return len(self.path)

    def __str__(self):
        return self.prefix \
            + ''.join(seg[letter]
                      for seg, letter
                      in zip(self.segments[self.start_segment:
                                           self.start_segment + len(self)],
                             self.path))

    def annotated_str(self):
        return self.prefix \
            + ''.join(seg[letter] if letter == '_'
                      else '{' + seg[letter] + '}'
                      for seg, letter
                      in zip(self.segments[self.start_segment:
                                           self.start_segment + len(self)],
                             self.path))


if __name__ == '__main__':
    main()
