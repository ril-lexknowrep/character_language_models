import os
import random
from itertools import islice

import tensorflow as tf
from more_itertools import split_into
import gc

import lstm_model
from encode_characters import InputEncoder, OutputEncoder

BATCH_SIZE = 256
LEFT_CONTEXT_WIDTH = 30
RIGHT_CONTEXT_WIDTH = 30
LSTM_UNITS = 1024
DENSE_DIM = 512
MAX_SEGMENT_LENGTH = 4 # max number of text lines to be combined into one segment.
LINES_PER_ITERATION = 60000
MODEL_FILE_NAME = 'bilstm_model_512_compare_no_dropout_2_lstms_768-384-w30.h5'

CORPUS_DIR = '/home/pgergo/ds_lexknowrep/lexknowrep/lstm_input/random_subset/' # change to corpus directory

VALIDATION_FILES = ['test_files/2.press_hu_promenad_003_2011.txt']

def main():
    input_enc = InputEncoder()
    output_enc = OutputEncoder()
    input_enc.load("input_encoder.pickle")
    output_enc.load("output_encoder.pickle")

    line_counts = {}

    corpus_files = [f_name for f_name in os.listdir(CORPUS_DIR)
                                            if f_name.endswith('.txt')]

    for f_name in corpus_files:
        line_count = 0
        with open(CORPUS_DIR + f_name, encoding='utf-8') as infile:
            for _ in infile:
                line_count += 1
        line_counts[f_name] = line_count

    bilstm_model = lstm_model.BiLSTM_Model(input_encoder=input_enc,
                                           output_encoder=output_enc,
                                           left_context=LEFT_CONTEXT_WIDTH,
                                           right_context=RIGHT_CONTEXT_WIDTH,
                                           lstm_units=LSTM_UNITS,
                                           dense_neurons=DENSE_DIM)

#   Uncomment the two lines below if resuming training.
#   Adjust file name, starting_line and total_processed_chars if applicable.

#    bilstm_model = lstm_model.BiLSTM_Model.load('bilstm_model_512.h5',
#                                                input_enc, output_enc)

    # This is the number of the first line that is read from each input file.
    # All lines before it are skipped. Can be used to resume processing
    # at a reasonably specific point after training has crashed or was aborted.
    starting_line = 0

    # This keeps track of the total number of processed characters.
    # Set to the number output after the most recent training "stage" when
    # resuming training after an interruption.
    total_processed_chars = 0

    validation_texts = []
    for val_file in VALIDATION_FILES:
        with open(val_file, encoding='utf-8') as val:
            validation_texts.extend(val.read().split("\n"))

    while True:
        corpus_files = list(filter(lambda x : 
                                starting_line < line_counts[x],
                                corpus_files))

        if len(corpus_files) == 0:
            print('All files have been processed')
            break

        lines_per_file = LINES_PER_ITERATION // len(corpus_files)

        line_sum = 0
        input_text_lengths = []
        while line_sum < lines_per_file:
            if line_sum >= lines_per_file - MAX_SEGMENT_LENGTH:
                input_text_lengths.append(lines_per_file - MAX_SEGMENT_LENGTH)
                break
            input_text_lengths.append(
                    random.choices(range(1, MAX_SEGMENT_LENGTH + 1),
                                   weights=range(1, MAX_SEGMENT_LENGTH + 1),
                                   k=1)[0])
            line_sum += input_text_lengths[-1]

        epoch_texts = []

        for f_name in corpus_files:
            with open(CORPUS_DIR + f_name, encoding="utf-8") as infile:
                file_slice = islice(infile,
                                    starting_line,
                                    starting_line + lines_per_file)
                input_text_lines = split_into(file_slice, input_text_lengths)
                # Strip final newline(s) from the very end
                # of each input string.
                input_texts = list(map(lambda x : ''.join(x).rstrip('\n'),
                                           input_text_lines))
                input_texts = list(filter(lambda x : len(x) > 0, input_texts))

            epoch_texts.extend(input_texts)

        bilstm_model.train(epoch_texts,
                           validation_texts=validation_texts)

        bilstm_model.model.save(MODEL_FILE_NAME)

        # Tensorflow leaks gigabytes of RAM over the course of a couple
        # million of training batches, so without explicit garbage
        # collection all free RAM fills up with garbage over time and
        # training eventually crashes.
        # If the two lines below are not sufficient to relieve this
        # problem, and the memory leak is still critical over time, then
        # an explicit call of `del bilstm_model` before clear_session,
        # and reloading the saved model after gc.collect()
        # should get rid of it completely.
        tf.keras.backend.clear_session()
        gc.collect()

        starting_line += lines_per_file

        total_processed_chars += sum(list(map(lambda x : len(x),
                                              epoch_texts)))

        print("Characters processed so far:", total_processed_chars)
        print("Lines processed so far:", starting_line)

if __name__ == '__main__':
    main()
