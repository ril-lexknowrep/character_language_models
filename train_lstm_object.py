'''
Module for training several LSTM models with different
configurations, keeping track of training progress and
results.
'''

import random
from itertools import islice
from dataclasses import dataclass
from subprocess import check_output
import gc
import argparse
from pathlib import Path
import logging
from time import time

import yaml
import tensorflow as tf
from more_itertools import split_into

import lstm_model
from encode_characters import InputEncoder, OutputEncoder

# max number of text lines to be combined into one segment:
MAX_SEGMENT_LENGTH = 8
LINES_PER_ITERATION = 40000
GARBAGE_COLLECT_TRIGGER = 5 * 1000 * 1000  # 5 GB free RAM remaining


def main():
    '''
    Runs training of one or more models based on a configuration file.
    Training run must be provided a name, which is set to the name
    of the provided training configuration file.
    Training progress is tracked in a file called
    {training_name}.train.yaml
    Results of the training for each model are written to
    {training_name}.results.yaml
    '''
    args = parse_arguments()

    (training_name, corpus_files, corpus_dir, validation_files,
     general_train_batch, general_val_batch, m_configs) =\
        read_training_config(args.config_file)

    progress_file = Path(training_name + '.progress.yml')
    summary_file = Path(training_name + '.summary.yml')

    if progress_file.exists():
        (line_count, word_count, char_count,
         val_line_count, val_word_count, val_char_count,
         line_counts, model_progress) = read_progress_file(progress_file)
    else:
        progress = {'name': training_name, 'start': lstm_model.timestamp()}

        line_counts = {}

        if corpus_dir:
            corpus_files = [f for f in Path(corpus_dir).glob('**/*')
                            if f.is_file()]
        else:
            corpus_files = [Path(f) for f in corpus_files]

        wc_out = check_output(["wc", "-wml",
                               *[str(cf) for cf in corpus_files]]).rstrip()

        WC_OUTPUT_TOTALS_ROW = -1
        WC_OUTPUT_LINES_COLUMN = 0
        wc_out_lines = wc_out.split(b'\n')
        line_count, word_count, char_count, _ =\
            wc_out_lines[WC_OUTPUT_TOTALS_ROW].split()
        line_count, word_count, char_count =\
            int(line_count), int(word_count), int(char_count)

        progress['training'] = {'lines': line_count, 'words': word_count,
                                'characters': char_count}

        if len(corpus_files) == 1:
            line_counts[str(corpus_files[0])] = line_count
        else:
            for fname, wc_file in zip(corpus_files,
                                      wc_out_lines[:WC_OUTPUT_TOTALS_ROW]):
                line_counts[str(fname)] =\
                    int(wc_file.split()[WC_OUTPUT_LINES_COLUMN])

        progress['line_counts'] = line_counts

        if validation_files:
            wc_out = check_output(["wc", "-wml", *validation_files])
            wc_out_lines = wc_out.rstrip().split(b'\n')
            val_line_count, val_word_count, val_char_count, _ =\
                wc_out_lines[WC_OUTPUT_TOTALS_ROW].split()
            val_line_count, val_word_count, val_char_count =\
                int(val_line_count), int(val_word_count), int(val_char_count)

        progress['validation'] = {'lines': val_line_count,
                                  'words': val_word_count,
                                  'characters': val_char_count}
        model_progress = []

    validation_texts = []
    for f_name in validation_files:
        with open(f_name, encoding='utf-8') as f:
            validation_texts.append(f.read())

    num_models = len(m_configs)

    for m_id, m_data in enumerate(m_configs):
        input_enc = InputEncoder(file=m_data.input_encoder)
        output_enc = OutputEncoder(file=m_data.output_encoder)

        # This is the number of the first line that is read from each input file
        # during training.
        # All lines before it are skipped. This is used to resume processing
        # at a reasonably specific point after training has crashed or was aborted.
        starting_line = 0

        # This keeps track of the total number of characters during training on
        # this corpus so far.
        processed_chars = 0

        if m_id >= len(model_progress):
            # no progress yet on this model
            if (m_data.model_name is not None and
                    Path(lstm_model.get_full_name(m_data.model_name)
                         + '.h5').exists()):
                model_name = lstm_model.get_full_name(m_data.model_name)
                model_file = Path(model_name + '.h5')
                bilstm_model = lstm_model.BiLSTM_Model.load(
                    model_file, input_enc, output_enc)
                info = ("Continuing training of existing model "
                        + f"#{m_id + 1}/{num_models} {model_name} "
                        + "on a new corpus.")
            else:
                bilstm_model = lstm_model.BiLSTM_Model(
                    input_encoder=input_enc, output_encoder=output_enc,
                    left_context=m_data.left_context,
                    right_context=m_data.right_context,
                    embedding=m_data.embedding,
                    lstm_units=m_data.lstm_units,
                    dense_neurons=m_data.dense_neurons,
                    dropout_ratios=m_data.dropout_ratios,
                    pass_final_output_only=m_data.pass_final_output_only,
                    log_file=m_data.log_file, model_name=m_data.model_name)

                model_name = bilstm_model.name
                info = (f"Created new model #{m_id + 1}/{num_models} "
                        + f"{model_name} for training.")
                model_file = Path(model_name + '.h5')
        elif m_id == len(model_progress) - 1:
            # Training of this model was already in progress,
            # but training of the next model hasn't started yet. 
            model_name = model_progress[m_id]['name']
            model_iterations = model_progress[m_id]['iterations']
            if model_iterations[-1]['complete']:
                # training of this model has been completed
                print(f"Training of model #{m_id + 1}/{num_models} {model_name} "
                      + "is already complete. Moving on to next model.")
                continue
            else:
                # training of this model is still in progress
                starting_line = model_iterations[-1]['next_start']
                processed_chars = model_iterations[-1]['proc_chars']
                model_file = Path(model_name + '.h5')
                info = (f"Resuming training of model #{m_id + 1}/{num_models}"
                        + f" {model_name} at line {starting_line} "
                        + f" after {processed_chars} processed characters.")
                bilstm_model = lstm_model.BiLSTM_Model.load(
                    model_file, input_enc, output_enc)
        else:
            # training of this model has been completed
            model_name = model_progress[m_id]['name']
            print(f"Training of model #{m_id + 1}/{num_models} {model_name} is "
                  + "already complete. Moving on to next model.")
            continue

        logging.basicConfig(filename=bilstm_model.log_file,
                            format='*%(asctime)s,%(name)s,%(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.DEBUG)

        logger = logging.getLogger(name='train_lstm_logger')
        if bilstm_model.log_file is None:
            logging.disable()

        logger.info(info)
        logger.info(f"Training {model_name} on corpus "
                    + (f"{corpus_dir}" if corpus_dir
                       else f"{[str(f) for f in corpus_files]}")
                    + f" consisting of {len(corpus_files)} files containing "
                    + f"{line_count:,} lines, {word_count:,} words and "
                    + f"{char_count:,} characters.")

        while True:
            model_corpus_files = [c for c in corpus_files
                                  if starting_line < line_counts[str(c)]]

            if len(model_corpus_files) == 0:
                print(f'Training complete for {model_name}')
                logger.info(f'Training complete for {model_name}')

                # Determining total training time and rates
                with open(progress_file, encoding='utf-8') as p_file:
                    iterations_log =\
                        yaml.load(
                            p_file.read(),
                            Loader=yaml.CLoader)['models'][m_id]['iterations']
                total_time = sum(i_data['train_secs']
                                 for i_data in iterations_log)
                chars_per_sec = int(char_count / total_time)
                words_per_sec = int(word_count / total_time)

                # Writing summary of traning results
                last_iter = iterations_log[-1]

                batch_size = lstm_model.DEFAULT_BATCH_SIZE
                if m_data.train_batch:
                    batch_size = m_data.train_batch
                elif general_train_batch:
                    batch_size = general_train_batch

                summary_dict = {
                    'model_name': model_name,
                    'parameters': bilstm_model.model.count_params(),
                    'batch_size': batch_size,
                    'total_time': total_time,
                    'chars_per_sec': chars_per_sec,
                    'words_per_sec': words_per_sec,
                    'final_loss': last_iter['loss'],
                    'final_accuracy': last_iter['acc'],
                    'final_perplexity': last_iter['perpl'],
                    'final_validation_loss': last_iter['val_loss'],
                    'final_validation_accuracy': last_iter['val_acc'],
                    'final_validation_perplexity': last_iter['val_perpl']}

                save_mode = 'w' if (m_id == 0) else 'a'

                with open(summary_file, save_mode) as s_file:
                    s_file.write(yaml.dump([summary_dict],
                                 sort_keys=False, Dumper=yaml.CDumper))

                # Marking traning as complete in progress file
                append_data = [{'complete': True}]
                with open(progress_file, 'a', encoding='utf-8') as p_file:
                    p_file.write(
                        indent_yaml(yaml.dump(append_data, sort_keys=False,
                                              Dumper=yaml.CDumper)))

                break

            lines_per_file = LINES_PER_ITERATION // len(model_corpus_files)

            line_sum = 0
            input_text_lengths = []
            while line_sum < lines_per_file:
                if line_sum >= lines_per_file - MAX_SEGMENT_LENGTH:
                    input_text_lengths.append(lines_per_file
                                              - MAX_SEGMENT_LENGTH)
                    break
                input_text_lengths.append(
                    random.choices(range(1, MAX_SEGMENT_LENGTH + 1),
                                   weights=range(1, MAX_SEGMENT_LENGTH + 1),
                                   k=1)[0])
                line_sum += input_text_lengths[-1]

            epoch_texts = []

            for f_name in model_corpus_files:
                with open(f_name, encoding="utf-8") as infile:
                    file_slice = islice(infile,
                                        starting_line,
                                        starting_line + lines_per_file)
                    input_text_lines = split_into(file_slice,
                                                  input_text_lengths)

                    # Strip newline(s) from both sides of each input string.
                    input_texts =\
                        [''.join(lines).strip('\n')
                         for lines in input_text_lines
                         # discard any empty splits
                         # at the end of a sequence of splits
                         if len(lines)]

#                    input_texts = list(map(lambda x: ''.join(x).rstrip('\n'),
#                                        input_text_lines))
#                    input_texts = list(filter(lambda x: len(x) > 0, input_texts))

                epoch_texts.extend(input_texts)

            batch_size = lstm_model.DEFAULT_BATCH_SIZE
            if m_data.train_batch:
                batch_size = m_data.train_batch
            elif general_train_batch:
                batch_size = general_train_batch

            val_batch_size = lstm_model.VALIDATION_BATCH_SIZE
            if m_data.val_batch:
                val_batch_size = m_data.val_batch
            elif general_val_batch:
                val_batch_size = general_val_batch

            train_start = time()
            metrics = bilstm_model.train(epoch_texts,
                                         validation_texts=validation_texts,
                                         batch_size=batch_size,
                                         validation_batch_size=val_batch_size)
            train_end = time()
            train_secs = int(train_end - train_start)

            loss, acc, perpl, val_loss, val_acc, val_perpl =\
                (metrics['loss'][0], metrics['accuracy'][0],
                 metrics['perplexity'][0], metrics['val_loss'][0],
                 metrics['val_accuracy'][0], metrics['val_perplexity'][0])

            # Note that 'complete' is always False here. Completion of model
            # training is checked at the beginning of the "while True" loop.
            iteration_data = {
                'complete': False, 'start': starting_line,
                'next_start': starting_line + lines_per_file,
                'loss': loss, 'acc': acc, 'perpl': perpl,
                'val_loss': val_loss, 'val_acc': val_acc,
                'val_perpl': val_perpl, 'train_secs': train_secs}

            train_input_len = sum(len(t) for t in epoch_texts)

            logger.info(f'Trained {model_name} on {lines_per_file} lines of '
                        + f'{len(model_corpus_files)} corpus files '
                        + f'containing {train_input_len} characters '
                        + f'in {train_secs} seconds.')
            logger.info(f"Saving model file {str(model_file)}")
            bilstm_model.model.save(model_file)

            starting_line += lines_per_file
            processed_chars += sum(len(t) for t in epoch_texts)

            iteration_data['proc_chars'] = processed_chars

            print("Total characters processed: "
                  + f"{processed_chars:,} / {char_count:,}")

            logger.info(f"{processed_chars=} {starting_line=}")

            logger.info(f"Saving training state to {str(progress_file)}")

            # save training progress
            if progress_file.exists():
                # Since just a few lines are added here to a potentially
                # rather long progress file, it would be inefficient to read
                # the whole file from disk, add the few lines, and write it
                # back again. Instead the data for the most recent iteration
                # are appended to the end of the progress file, which
                # necessitates the adding of some extra indentation.
                # This is a harmless and useful hack.
                if iteration_data['start'] == 0:
                    # first iteration of the training run for this model
                    append_data = [{'name': model_name,
                                    'iterations': [iteration_data]}]
                    with open(progress_file, 'a', encoding='utf-8') as p_file:
                        p_file.write(yaml.dump(append_data, sort_keys=False,
                                               Dumper=yaml.CDumper))
                else:
                    # all further iterations
                    append_data = [iteration_data]
                    with open(progress_file, 'a', encoding='utf-8') as p_file:
                        p_file.write(
                            indent_yaml(yaml.dump(append_data, sort_keys=False,
                                                  Dumper=yaml.CDumper)))
            else:
                # very first iteration of the training run
                with open(progress_file, 'w', encoding='utf-8') as p_file:
                    progress['models'] = [{'name': model_name,
                                           'iterations': [iteration_data]}]
                    p_file.write(yaml.dump(progress, Dumper=yaml.CDumper,
                                           sort_keys=False))

            # Tensorflow leaks gigabytes of RAM over the course of a couple
            # million of training batches, so without explicit garbage
            # collection all free RAM fills up with garbage over time and
            # training eventually crashes.

            # Check free memory:

            with open('/proc/meminfo') as meminfo:
                for line in meminfo:
                    if 'MemFree' in line:
                        free_mem_in_kb = int(line.split()[1])

            logger.info(f"Memory remaining: {free_mem_in_kb:,} kb")

            if free_mem_in_kb < GARBAGE_COLLECT_TRIGGER:
                logger.info("Running out of memory. Initiating garbage "
                            "collection and reloading model.")
                del bilstm_model
                tf.keras.backend.clear_session()
                gc.collect()
                bilstm_model = lstm_model.BiLSTM_Model.load(
                    model_file, input_enc, output_enc, check_version=False)


def parse_arguments():
    parser = argparse.ArgumentParser(
        prog='train_lstm_object',
        description='Trains LSTM objects on a specified corpus')
    parser.add_argument('config_file', type=Path,
                        help='path to YAML file containing model configs',
                        )
    parser.add_argument('--corpus-directory', '-d', type=str,
                        help='path to dictionary containing the training '
                        + 'corpus files')
    parser.add_argument('--corpus-files', '-f', nargs='+', type=str,
                        help='list of corpus files to train the models on')
    parser.add_argument('--validation-files', '-v', nargs='+', type=Path,
                        help='list of corpus files to validate the models on')
    return parser.parse_args()


def read_training_config(file_path):
    '''
    Read training configuration from YAML file.
    This includes the corpus that is used for training and the
    configuration of each model to be trained.
    The YAML file's base file name without the final extension
    is treated as the name of the training run.
    Return name of training run and list of ModelConfig dataclasses.
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
        train_batch: int | None = None
        val_batch: int | None = None

    configs = []

    training_run_name = file_path.stem

    with open(file_path, encoding='utf-8') as cfg:
        training_cfg = yaml.load(cfg.read(), Loader=yaml.CLoader)

    corpus_files = training_cfg.get('corpus_files', None)
    corpus_dir = training_cfg.get('corpus_dir', None)

    if corpus_files and corpus_dir:
        raise ValueError("Only either a corpus directory or a list of "
                         + "corpus files can be specified in the training "
                         + "configuration, not both.")
    if (corpus_files is None) and (corpus_dir is None):
        raise ValueError("No corpus specified in training configuration.")

    validation_files = training_cfg.get('validation_files', None)

    general_train_batch = training_cfg.get('train_batch', None)
    general_val_batch = training_cfg.get('val_batch', None)

    for cfg_dict in training_cfg['models']:
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
                               cfg_dict.get('embedding', 0),
                               cfg_dict.get('final_only', False),
                               cfg_dict.get('train_batch', None),
                               cfg_dict.get('val_batch', None))
        except KeyError as k_err:
            print(f"\nRequired key missing in YAML config: {k_err}\n")
            raise
        except TypeError as t_err:
            print(f"\nInvalid data in YAML config: {t_err}\n")
            raise
        configs.append(data)
    return (training_run_name, corpus_files, corpus_dir,
            validation_files, general_train_batch,
            general_val_batch, configs)


def read_progress_file(p_file):
    '''Read current training progress state from YAML file'''
    with open(p_file, encoding='utf-8') as progress:
        p_dict = yaml.load(progress.read(), Loader=yaml.CLoader)
        line_count, word_count, char_count =\
            (p_dict['training']['lines'], p_dict['training']['words'],
                p_dict['training']['characters'])
        val_line_count, val_word_count, val_char_count =\
            (p_dict['validation']['lines'], p_dict['validation']['words'],
                p_dict['validation']['characters'])
        line_counts = p_dict['line_counts']
        model_progress = p_dict['models']
    return (line_count, word_count, char_count,
            val_line_count, val_word_count, val_char_count,
            line_counts, model_progress)


def indent_yaml(lines, level=1):
    '''Indent a YAML file to nest data'''
    INDENTATION = "  "
    return ''.join([INDENTATION * level + line
                    for line in lines.splitlines(True)])


if __name__ == '__main__':
    main()
