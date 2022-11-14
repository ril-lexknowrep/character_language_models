import os
import random
from itertools import islice
from dataclasses import dataclass

import tensorflow as tf
from more_itertools import split_into
import gc

import yaml

import lstm_model
from encode_characters import InputEncoder, OutputEncoder

BATCH_SIZE = 256
LEFT_CONTEXT_WIDTH = 30
RIGHT_CONTEXT_WIDTH = 30
LSTM_UNITS = 1024
DENSE_DIM = 512

# max number of text lines to be combined into one segment:
MAX_SEGMENT_LENGTH = 4
LINES_PER_ITERATION = 60000

# change to corpus directory:
CORPUS_DIR = '/home/pgergo/ds_lexknowrep/lexknowrep/lstm_input/random_subset/'

VALIDATION_FILES = ['test_files/2.press_hu_promenad_003_2011.txt']


def main():
    m_data = read_model_config('model_config_sample.yml')[0]

    input_enc = InputEncoder(file=m_data.input_encoder)
    output_enc = OutputEncoder(file=m_data.output_encoder)

    line_counts = {}

    corpus_files = [f_name for f_name in os.listdir(CORPUS_DIR)
                    if f_name.endswith('.txt')]

    for f_name in corpus_files:
        line_count = 0
        with open(CORPUS_DIR + f_name, encoding='utf-8') as infile:
            for _ in infile:
                line_count += 1
        line_counts[f_name] = line_count

    bilstm_model = lstm_model.BiLSTM_Model(
        input_encoder=input_enc, output_encoder=output_enc,
        left_context=m_data.left_context, right_context=m_data.right_context,
        lstm_units=m_data.lstm_units, dense_neurons=m_data.dense_neurons,
        dropout_ratios=m_data.dropout_ratios,
        pass_final_output_only=m_data.pass_final_output_only,
        log_file=m_data.log_file, model_name=m_data.model_name)

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
        corpus_files = list(filter(lambda x: 
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
                input_texts = list(map(lambda x: ''.join(x).rstrip('\n'),
                                       input_text_lines))
                input_texts = list(filter(lambda x: len(x) > 0, input_texts))

            epoch_texts.extend(input_texts)

        bilstm_model.train(epoch_texts,
                           validation_texts=validation_texts)

        bilstm_model.model.save(bilstm_model.name + ".h5")

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

        total_processed_chars += sum(list(map(lambda x: len(x),
                                              epoch_texts)))

        print("Characters processed so far:", total_processed_chars)
        print("Lines processed so far:", starting_line)


def read_model_config(file_name):
    '''
    Read list of model configs from YAML file.
    Return list of ModelConfig dataclasses.
    '''
    @dataclass
    class ModelConfig:
        input_encoder: str
        output_encoder: str
        left_context: int
        right_context: int
        lstm_units: list[int]
        dense_neurons: list[int]
        dropout_ratios: list[int]
        model_name: str | None = None
        log_file: str | None = None
        embedding: int = 0
        pass_final_output_only: bool = False

    configs = []

    with open(file_name, encoding='utf-8') as cfg:
        cfg_list = yaml.load(cfg.read(), Loader=yaml.CLoader)

    for cfg_dict in cfg_list:
        try:
            dense_layers = cfg_dict['dense_layers']
            neurons_list = []
            dropout_list = []
            for layer in dense_layers:
                neurons_list.append(layer['neurons'])
                dropout_list.append(layer.get('dropout', 0))

            data = ModelConfig(cfg_dict['input_encoder'],
                               cfg_dict['output_encoder'],
                               cfg_dict['left_context'],
                               cfg_dict['right_context'],
                               cfg_dict['lstm_layers'],
                               neurons_list, dropout_list,
                               cfg_dict.get('name', None),
                               cfg_dict.get('log_file', None),
                               cfg_dict.get('embedding', False),
                               cfg_dict.get('final_only', False))
        except KeyError as k_err:
            print("Required key missing in YAML config:", k_err)
        except TypeError as t_err:
            print("Invalida data in YAML config:", t_err)
        configs.append(data)
    return configs


if __name__ == '__main__':
    main()
