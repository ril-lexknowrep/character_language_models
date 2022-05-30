import sys
import difflib
import numpy as np

import tensorflow as tf
import lstm_model
from encode_characters import InputEncoder, OutputEncoder

INPUT_ENCODER_FILE = "input_encoder.pickle"
OUTPUT_ENCODER_FILE = "output_encoder.pickle"
LANGUAGE_MODEL = 'bilstm_model_512.h5'

LOG_FILE = 'correct_ocr_diffs.log'

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

IGNORE_DELETE_THRESHOLD = 3
ACCEPT_INSERT_THRESHOLD = 3


def main():
    '''
    Compare the two files specified on the command line and
    output a corrected version as a result of combining the
    good bits from both files.
    If there are longer insertions/deletions in one or both
    of the files, those in the first file are ignored and
    those in the second file accepted automatically.
    What counts as 'longer' in this respect is determined
    by IGNORE_DELETE_THRESHOLD and ACCEPT_INSERT_THRESHOLD
    respectively.
    '''

    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()

    input_enc = InputEncoder()
    output_enc = OutputEncoder()
    input_enc.load(INPUT_ENCODER_FILE)
    output_enc.load(OUTPUT_ENCODER_FILE)

    bilstm_model = lstm_model.BiLSTM_Model.load(LANGUAGE_MODEL,
                                                input_enc, output_enc)
    left_window = bilstm_model.encoder.left_context
    right_window = bilstm_model.encoder.right_context

    mistakes_set = set()
    for alt_set in OCR_MISTAKES:
        mistakes_set |= alt_set

    with open(sys.argv[1], encoding='utf-8') as infile:
        text_a = infile.read().replace('\t', ' ')
    with open(sys.argv[2], encoding='utf-8') as infile:
        text_b = infile.read().replace('\t', ' ')

    log_file = open(LOG_FILE, 'w', encoding='utf-8')

    matcher = difflib.SequenceMatcher()
    matcher.set_seqs(text_a, text_b)
    opcodes = matcher.get_opcodes()

    segs = [DiffSegment(*opcode, text_a, text_b) for opcode in opcodes]

    done_string = ''

    alternatives = None
    for i, seg in enumerate(segs):
        if alternatives is None:
            if seg.tag == 'equal':
                done_string += seg.a_segment
            elif (seg.tag == 'delete' and 
                  len(seg.a_segment) > IGNORE_DELETE_THRESHOLD):
                pass
            elif (seg.tag == 'insert' and
                  len(seg.b_segment) > ACCEPT_INSERT_THRESHOLD):
                done_string += seg.b_segment
            elif (seg.tag == 'replace' and
                  len(seg.a_segment) > 
                            len(seg.b_segment) + IGNORE_DELETE_THRESHOLD):
                done_string += seg.b_segment
            elif (seg.tag == 'replace' and
                  len(seg.b_segment) >
                            len(seg.a_segment) + ACCEPT_INSERT_THRESHOLD):
                done_string += seg.b_segment
            else:
                left_context = done_string[-left_window:]

                diffs = [seg.a_segment, seg.b_segment]
                diffs.extend(additional_alternatives(mistakes_set,
                                                     seg.a_segment,
                                                     seg.b_segment))
                start_letter = ord('a')
                alternatives = {}
                annotated_alternatives = {}
                for i, diff in enumerate(diffs):
                    alternatives[chr(start_letter + i)] =\
                                        left_context + diffs[i]
                    annotated_alternatives[chr(start_letter + i)] =\
                                        left_context + "{" + diffs[i] + "}"
        else:
            if (seg.tag == 'equal' or
                (seg.tag == 'insert' and
                 len(seg.b_segment) > ACCEPT_INSERT_THRESHOLD)
                ):
                    if len(seg.b_segment) >= right_window + 1:
                        right_context = seg.b_segment[:right_window + 1]
                        right_rest = seg.b_segment[right_window + 1:]
                        alternatives = {k: v + right_context for k, v
                                                    in alternatives.items()}
                        annotated_alternatives =\
                                {k: v + right_context for k, v
                                    in annotated_alternatives.items()}
                        best_alternative =\
                                evaluate_alternatives(alternatives,
                                                      annotated_alternatives,
                                                      bilstm_model,
                                                      log_file)
                        done_string += (best_alternative[left_window:] +
                                        right_rest)
                        alternatives = None
                    else:
                        alternatives = {k: v + seg.b_segment
                                                    for k, v
                                                    in alternatives.items()}
                        annotated_alternatives =\
                                        {k: v + seg.b_segment
                                            for k, v
                                            in annotated_alternatives.items()}
            elif (seg.tag == 'delete' and
                  len(seg.a_segment) > IGNORE_DELETE_THRESHOLD):
                pass
            elif seg.tag == 'replace' and (
                        len(seg.a_segment) >
                                len(seg.b_segment) + IGNORE_DELETE_THRESHOLD or
                        len(seg.b_segment) > 
                                len(seg.a_segment) + ACCEPT_INSERT_THRESHOLD
                                            ):
                alternatives = {k: v + seg.b_segment for k, v
                                                     in alternatives.items()}
                annotated_alternatives = {k: v + seg.b_segment 
                                            for k, v
                                            in annotated_alternatives.items()}
            else:
                diffs = [seg.a_segment, seg.b_segment]
                diffs.extend(additional_alternatives(mistakes_set,
                                                     seg.a_segment,
                                                     seg.b_segment))
                start_letter = ord('a')
                new_alternatives = {}
                new_annotated = {}
                for key, value in alternatives.items():
                    for i, diff in enumerate(diffs):
                        new_alternatives[key + chr(start_letter + i)] =\
                                                            value + diffs[i]
                        new_annotated[key + chr(start_letter + i)] =\
                            annotated_alternatives[key] + "{" + diffs[i] + "}"
                alternatives = new_alternatives
                annotated_alternatives = new_annotated

    output_file = open(sys.argv[3], 'w', encoding='utf-8')
    output_file.write(done_string)


def evaluate_alternatives(alternatives, annotated_alternatives,
                          model, log_file):
    '''
    Measure the perplexity of the members of a dict of alternative
    substrings using a language model, log the intermediate
    results, and return the alternative substring with the
    best perplexity score.
    '''
    log_file.write('Alternatives:\n')
    best_perplexity = np.inf
    best_index = 0
    for i, (key, value) in enumerate(alternatives.items()):
        perplexity = model.metrics_on_string(value)[1]
        if perplexity < best_perplexity:
            best_index = i
            best_perplexity = perplexity
            best_key = key
        log_file.write('\t'.join([key,
                         str(perplexity),
                         list(annotated_alternatives.values())[i]
                         ]) + '\n')
    log_file.write(f"Best:\t{best_key}\t" +
                   list(annotated_alternatives.values())[best_index] +
                   '\n\n')
    return list(alternatives.values())[best_index]


def additional_alternatives(mistakes_set, alt_a, alt_b):
    '''
    Suggest further alternatives in addition to the two matching blocks
    that were found in the input files based on known OCR error patterns.
    '''
    if alt_a not in mistakes_set and alt_b not in mistakes_set:
        return []
    
    further_alts = []
    for alt_set in OCR_MISTAKES:
        if alt_a in alt_set or alt_b in alt_set:
            further_alts.extend(list(alt_set - {alt_a, alt_b}))
    return further_alts


class DiffSegment:
    '''A class that provides a comfortable interface for difflib diffs'''
    def __init__(self, tag, a_start, a_end, b_start, b_end, a_text, b_text):
        self.tag = tag
        self.a_start = a_start
        self.a_end = a_end
        self.b_start = b_start
        self.b_end = b_end
        self.a_segment = a_text[a_start:a_end]
        self.b_segment = b_text[b_start:b_end]


if __name__ == '__main__':
    main()
