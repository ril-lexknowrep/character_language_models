import math
import numpy as np
import datetime
import logging
import re
import json
from contextlib import redirect_stdout
from io import StringIO
from itertools import accumulate, chain
import pathlib

import tensorflow as tf
import keras
from tensorflow.keras.layers import Concatenate, LSTM, Dense, Dropout,\
    Embedding, Masking
from more_itertools import windowed

MODULE_VERSION = "0.2.0"

DEFAULT_BATCH_SIZE = 1024
GLOBAL_CLIPNORM = 0.01

# The Keras documentation for the Model.predict() method says:
# "Computation is done in batches. This method is designed for batch
# processing of large numbers of inputs. It is not intended for use
# inside of loops that iterate over your data and process small numbers
# of inputs at a time.
# For small numbers of inputs that fit in one batch, directly use
# __call__() for faster execution."
# We treat less than 16 inputs as small.
MIN_PREDICT_BATCH = 16

# Validation always uses a large batch size as set here.
# If Tensorflow runs out of memory at the validation stage
# during training, which can happen if LSTM and dense
# layer dimensions are set to high values, reduce this
# number until the validation runs correctly.
# This value considerably affects the speed at which
# validation is carried out at the end of each call of
# the train() method, so don't set it much lower
# than what is absolutely necessary for stability.
VALIDATION_BATCH_SIZE = 8192


def timestamp():
    '''Return a timestamp that can be used as part of a file name'''
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def files_to_texts(file_names):
    '''Open and read files, return content of each file'''
    texts = []
    for f_name in file_names:
        with open(f_name, encoding='utf-8') as f:
            texts.append(f.read())
    return texts


def get_full_name(model_name):
    '''Return full name of a named model based on the base name'''
    return f'BiLSTM_Model_v{MODULE_VERSION}>{model_name}>'


def rename_model(model_file, new_name,
                 file_name=None, force=False, rename_file=True):
    '''
    Rename model in saved model file to new name.
    By default the file is renamed along with it to the new name
    plus the original file extension.
    Alternatively a different full file name can be specified under
    which the renamed model is saved.
    '''
    KERAS_NAME_PATTERN = r'^[A-Za-z0-9.][A-Za-z0-9_.\\/>-]*$'
    if not force:
        if re.match(KERAS_NAME_PATTERN, new_name) is None:
            raise ValueError("Invalid new name. Name must match "
                             + f"{KERAS_NAME_PATTERN}")
        if not new_name.startswith(f"BiLSTM_Model_v{MODULE_VERSION}"):
            raise ValueError("Invalid new name. Name should start with "
                             + MODULE_VERSION)
    model = tf.keras.models.load_model(
        model_file,
        custom_objects={"PerElementPerplexity": PerElementPerplexity})
    model._name = new_name

    if file_name is not None:
        model.save(file_name)
    else:
        model.save(model_file)
        if rename_file:
            model_file = pathlib.Path(model_file)
            model_file.rename(new_name + model_file.suffix)


class BiLSTM_Model:
    '''
    Class for bidirectional LSTM language models.
    * embedding specifies the embedding dimension, or False/0 if
    BiLSTM_Encoder should use custom feature vectors as provided
    by the input_encoder instead of an Embedding layer.
    * lstm_units specifies the output dimensions of the subsequent
    LSTM layers. As many stacked LSTM layers are added to the model
    as layer sizes are specified.
    * dense_neurons specifies the output dimension of each subsequent
    dense layer stacked on top of the combined bidirectional output of
    the LSTM layers.
    * dropout_ratios specifies the individual dropout ratio for each
    dropout layer that comes directly AFTER each dense layer. Thus the
    number of dropout layers should accordingly be identical to the
    number of dense layers. A ratio of 0 for a layer means that no
    dropout layer is added after the corresponding dense layer.
    * pass_final_output_only determines whether only the output
    of the left and right LSTM layers on the top of the LSTM stack
    should be concatenated and passed to the dense layers (if True),
    or whether the outputs of all LSTM layers should be concatenated
    (False).
    '''

    def __init__(self, input_encoder, output_encoder,
                 left_context, right_context, embedding=False,
                 lstm_units=[512], dense_neurons=[512],
                 dropout_ratios=[0], pass_final_output_only=False,
                 verbose=True, log_file=None, model_name=None):

        self.str = ((f'name="{model_name}"_' if model_name else '')
                    + f'w_{left_context}-{right_context}_'
                    + (f'embedding_{embedding}_' if embedding else '')
                    + f'lstm_{"-".join(str(x) for x in lstm_units)}_'
                    + f'dense_{"-".join(str(x) for x in dense_neurons)}_'
                    + f'dropout_{"-".join(str(x) for x in dropout_ratios)}'
                    + ('_final-only' if pass_final_output_only else ''))

        if embedding:
            input_encoder.input_char_to_int =\
                {input_encoder.PADDING: 0,
                 input_encoder.START_TEXT: 1,
                 input_encoder.END_TEXT: 2,
                 input_encoder.MASKING: 3}
            input_encoder.input_char_to_int |=\
                {key: i + 4 for i, key
                 in enumerate(input_encoder.keys())}
            input_encoder.code_dimension = 1

        # Naming philosophy:
        # The idea is that if a user has specified a name for the model, 
        # then that name can be assumed to be unique, so adding an extra
        # timestamp is not necessary. In fact it would make things
        # unnecessarily difficult if the user wants to continue
        # training a named model on the same or on a different corpus.
        # On the other hand, the timestamp is necessary to tell apart two
        # or more identical anonymous models that were e.g. trained on
        # different corpora, since without the timestamp these would be
        # saved to a model file with the same name (i.e. one would
        # overwrite the other).
        if model_name is None:
            self.name = repr(self) + '>' + timestamp()
        else:
            self.name = get_full_name(model_name)

        self.log_file = log_file
        self.logger = logging.getLogger(name=self.name)

        logging.basicConfig(filename=log_file,
                            format='*%(asctime)s,%(name)s,%(message)s',
                            datefmt='%Y/%m/%d %H:%M:%S',
                            level=logging.DEBUG)

        if log_file is None:
            logging.disable()

        # Configure bidirectional LSTM model
        if embedding:
            input_left = tf.keras.Input(shape=(left_context,),
                                        name="input_left")
            input_right = tf.keras.Input(shape=(right_context,),
                                         name="input_right")
            preprocessed_left = Embedding(
                input_dim=len(input_encoder.input_char_to_int) + 1,  # +1 OOV
                output_dim=embedding, mask_zero=True)(input_left)
            right_embedding = Embedding(
                input_dim=len(input_encoder.input_char_to_int) + 1,  # +1 OOV
                mask_zero=True,
                output_dim=embedding)
            preprocessed_right = right_embedding(input_right)
            right_mask = right_embedding.compute_mask(input_right)
        else:
            input_left = tf.keras.Input(shape=(left_context,
                                               input_encoder.code_dimension),
                                        name="input_left")
            input_right = tf.keras.Input(shape=(right_context,
                                                input_encoder.code_dimension),
                                         name="input_right")
            preprocessed_left = Masking(name="mask_left")(input_left)

            right_masking = Masking(name="mask_right")
            preprocessed_right = right_masking(input_right)
            right_mask = right_masking.compute_mask(input_right)

        # Only the first LSTM for the right context should go backward,
        # as it returns the read sequence in reverse order. Reversing
        # this again in subsequent LSTM layers would mean that the right
        # context is read from right to left, then from left to right,
        # then again from right to left, and so on, which obviously
        # makes no sense and understandably yields poor results.
        # Keras uses the right-side mask in the first LSTM layer
        # correctly, skipping the correct time steps while going
        # backward. However, the reading order is reversed in the
        # sequence output of the first layer (so to speak), therefore
        # the mask must be reversed on the time axis as well for these
        # layers, otherwise as many tokens would be skipped just
        # to the right of the center token, i.e. from the *left* edge
        # of the context as there are padding tokens in the input,
        # while these token should in fact be skipped on the *right*
        # edge of the right context.
        reversed_right_mask = tf.reverse(right_mask, axis=[1])

        lstm_states = []

        if len(lstm_units) == 1:
            forward_output = LSTM(lstm_units[0],
                                  name="lstm_forward",
                                  )(preprocessed_left)
            backward_output = LSTM(lstm_units[0],
                                   name="lstm_backward",
                                   go_backwards=True,
                                   )(preprocessed_right)
            lstm_states.extend([forward_output, backward_output])
        else:
            # All but the final LSTM layers have to return sequences.
            # Only the first LSTM layer for the right input goes backward.

            # first LSTM layers:
            forward_seq, forward_final_state, _ = LSTM(
                                                lstm_units[0],
                                                name='lstm_forward_1',
                                                return_sequences=True,
                                                return_state=True
                                                )(preprocessed_left)
            backward_seq, backward_final_state, _ = LSTM(
                                                lstm_units[0],
                                                name='lstm_backward_1',
                                                return_sequences=True,
                                                return_state=True,
                                                go_backwards=True
                                                )(preprocessed_right)
            if not pass_final_output_only:
                lstm_states.extend([forward_final_state,
                                    backward_final_state])

            # any intermediate LSTM layers
            for i in range(1, len(lstm_units) - 1):
                forward_seq, forward_final_state, _ = LSTM(
                                            lstm_units[i],
                                            name=f'lstm_forward_{i + 1}',
                                            return_sequences=True,
                                            return_state=True
                                            )(forward_seq)
                backward_seq, backward_final_state, _ = LSTM(
                                            lstm_units[i],
                                            name=f'lstm_backward_{i + 1}',
                                            return_sequences=True,
                                            return_state=True
                                           )(backward_seq,
                                             mask=reversed_right_mask)
                if not pass_final_output_only:
                    lstm_states.extend([forward_final_state,
                                        backward_final_state])

            # final LSTM layers
            forward_output = LSTM(lstm_units[-1],
                                  name=f"lstm_forward_{len(lstm_units)}"
                                  )(forward_seq)
            backward_output = LSTM(lstm_units[-1],
                                   name=f"lstm_backward_{len(lstm_units)}"
                                   )(backward_seq, mask=reversed_right_mask)
            lstm_states.extend([forward_output, backward_output])

        forward_model = tf.keras.Model(inputs=input_left,
                                       outputs=forward_output)

        backward_model = tf.keras.Model(inputs=input_right,
                                        outputs=backward_output)

        bidirectional = Concatenate()(lstm_states)

        dense_layers = Dense(dense_neurons[0],
                             name="dense_1",
                             activation="relu")(bidirectional)
        if dropout_ratios[0] > 0:
            dense_layers = Dropout(dropout_ratios[0],
                                   name="dropout_1")(dense_layers)

        for i in range(1, len(dense_neurons)):
            dense_layers = Dense(dense_neurons[i],
                                 name=f"dense_{i + 1}",
                                 activation="relu")(dense_layers)

            if dropout_ratios[i] > 0:
                dense_layers = Dropout(dropout_ratios[i],
                                       name=f"dropout_{i + 1}"
                                       )(dense_layers)

        output_classifier = Dense(output_encoder.code_dimension,
                                  name="softmax",
                                  activation="softmax")(dense_layers)

        self.model = tf.keras.Model(inputs=[forward_model.input,
                                            backward_model.input],
                                    outputs=[output_classifier],
                                    name=self.name)

        self.model.compile(loss='categorical_crossentropy',
                           optimizer='adam',
                           metrics=['accuracy', PerElementPerplexity()],
                           run_eagerly=True)
        # run_eagerly is required for the .numpy() function in the
        # perplexity metric function to work.

        if verbose:
            self.model.summary()

        summary_output = StringIO()
        with redirect_stdout(summary_output):
            self.model.summary()

        self.logger.info(f'New model created: {repr(self)}\n' +
                         summary_output.getvalue())

        self.model.optimizer.learning_rate = BiLSTM_Model.learning_rate_decay()

        # Largeish models (above a million or so parameters) tend to
        # exhibit exploding gradients during training with catastrophic
        # consequences if gradients are not clipped.
        self.model.optimizer.global_clipnorm = GLOBAL_CLIPNORM

        self.encoder = BiLSTM_Encoder(input_encoder, output_encoder,
                                      left_context=left_context,
                                      right_context=right_context,
                                      logger=self.logger)

    def __str__(self):
        return self.str

    def __repr__(self):
        return f'BiLSTM_Model_v{MODULE_VERSION}>{str(self)}>'

    def learning_rate_decay():
        # Decay learning rate gradually from 1e-3 to 1e-5 until a total of
        # about STOP_DECAY_AFTER (character or word) tokens have been
        # trained on.
        # Note that in one iteration the optimizer processes a whole batch,
        # so the specified number of tokens needs to be divided by
        # the batch size to get the number of iterations in which the
        # learning rate will decay.

        STOP_DECAY_AFTER = 2e9
        steps = STOP_DECAY_AFTER // DEFAULT_BATCH_SIZE
        INITIAL_LR = 1e-3
        FINAL_LR = 1e-5
        DECAY_POWER = 2

        return tf.keras.optimizers.schedules.PolynomialDecay(
                                    initial_learning_rate=INITIAL_LR,
                                    decay_steps=steps,
                                    end_learning_rate=FINAL_LR,
                                    power=DECAY_POWER)

    @classmethod
    def load(cls, model_file, input_encoder, output_encoder,
             check_version=True, embedding=False):
        '''
        Load Keras model from file and restore its state before it was saved.
        '''
        saved_model =\
            tf.keras.models.load_model(model_file,
                                       custom_objects={"PerElementPerplexity":
                                                       PerElementPerplexity})

        left_context = saved_model.inputs[0].shape[1]
        right_context = saved_model.inputs[1].shape[1]

        if any(isinstance(layer, keras.layers.core.embedding.Embedding)
               for layer in saved_model.layers):
            input_encoder.input_char_to_int =\
                {input_encoder.PADDING: 0,
                 input_encoder.START_TEXT: 1,
                 input_encoder.END_TEXT: 2,
                 input_encoder.MASKING: 3}
            input_encoder.input_char_to_int |=\
                {key: i + 4 for i, key
                 in enumerate(input_encoder.keys())}
            input_encoder.code_dimension = 1

        version_match = re.match(r"BiLSTM_Model_v([0-9.]+)>", saved_model.name)

        return_model = BiLSTM_Model(input_encoder,
                                    output_encoder,
                                    left_context,
                                    right_context,
                                    verbose=False,
                                    model_name=saved_model.name)

        return_model.logger.info(f"Loading model from file {model_file}.")

        if version_match is None:
            error_message =\
                ("Unable to establish module version with which the saved "
                 + "model was created based one saved model name "
                 + f"{saved_model.name}. Consider renaming the model using "
                 + "rename_model() or set check_version() to False.")
            if check_version:
                return_model.logger.exception(error_message)
                raise RuntimeError(error_message)
            else:
                return_model.name = saved_model.name
                return_model.str = saved_model.name
                return_model.logger.warning(
                    error_message
                    + "Loading anyway because check_version=False.")
        elif version_match[1] != MODULE_VERSION:
            error_message =\
                ("Saved model was created with "
                 + version_match[0]
                 + f" but the current version is {MODULE_VERSION}. ")
            if check_version:
                try:
                    raise RuntimeError(
                        error_message
                        + "Revert or update lstm_model.py to correct version"
                        + " or load the model with check_version=False.")
                except RuntimeError as error:
                    return_model.logger.exception(str(error))
                    raise
            else:
                return_model.logger.warning(
                    error_message
                    + "Loading anyway because check_version=False.")

        if version_match:
            whole_match = version_match[0]
            return_model.name = saved_model.name
            return_model.str = saved_model.name[len(whole_match):]

        # This part is necessary, because on loading the model "run_eagerly"
        # is automatically deactivated. It seems that the model must be
        # recompiled every time on loading to run TF eagerly, which is in
        # turn required for calculation of the perplexity metric.

        saved_optimizer = saved_model.optimizer
        saved_model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy', PerElementPerplexity()],
                            run_eagerly=True)
        # Compile creates a new optimizer, so data on training progress,
        # particularly the number of iterations so far, is lost.
        saved_model.optimizer = saved_optimizer
        return_model.model = saved_model

        return_model.model.optimizer.learning_rate =\
            BiLSTM_Model.learning_rate_decay()

        print("Loaded model:")
        return_model.model.summary()

        summary_output = StringIO()
        with redirect_stdout(summary_output):
            return_model.model.summary()

        return_model.logger.info("Loaded model:\n"
                                 + summary_output.getvalue())
        return_model.logger.info("Trained for",
                                 int(return_model.model.optimizer.iterations),
                                 "iterations")
        print("Trained for", int(return_model.model.optimizer.iterations),
              "iterations")

        return return_model

    def train(self, texts, batch_size=DEFAULT_BATCH_SIZE,
              validation_batch_size=VALIDATION_BATCH_SIZE,
              validation_texts=None, num_epochs=1, csv_log_dir='./',
              text_files=False, tensorboard_log_dir=None, **tfargs):
        '''
        Train model on an iterable of texts. Each of these can be
        as short as a single sentence or as long as a document,
        as required. If text_files is True, then the elements of the
        iterable are interpreted as file names, which are opened and
        read. Otherwise the texts are treated as strings for the model
        to be trained on.
        If a TensorBoard log directory is specified, this is used
        as the root directory of all TensorBoard logs, and subdirectories
        with timestamps are created under this directory for each training
        run. If no log directory is specified, TensorBoard logging is
        disabled.
        The TensorBoard callback's keyword arguments are passed on,
        except for the log_dir argument, which is handled as described
        previously.
        If csv_log_dir is not None, training progress is logged by the
        CSVLogger callback in Keras to a log file with the same name as
        the model that is being trained. If training is continued, new
        training results for the model are appended to the existing log.

        Validation on a fraction of the training data would not be
        particularly informative because of the way the BiLSTM_Sequence
        works. In a nutshell, different, often overlapping windows
        of *the same* text would be used for validation, so this could
        be expected to drastically overestimate prediction accuracy of
        the model. Thus an iterable of validation texts which are not
        included in the training corpus can be optionally passed, on
        which the model is validated after each training epoch.
        If such an iterable is not provided, there is no validation
        after training on each epoch. If text_files is True, the
        validation texts are interpreted as file names which are opened
        and read; otherwise they are treated as the input strings for
        the model to be validated on.

        Return the history object returned by Model.fit().
        '''
        # If a single string is provided in texts, treat it as a
        # one-element list.
        if isinstance(texts, str):
            texts = [texts]
        if text_files:
            texts = files_to_texts(texts)

        sequence_object = BiLSTM_Sequence(texts, batch_size,
                                          self.encoder)

        texts_length = sum(len(t) for t in texts)
        self.logger.info(f'Training model on {len(texts)} texts '
                         + f'of total length {texts_length} '
                         + f'with {batch_size=} for {num_epochs=}')

        validation_sequence = None
        if validation_texts:
            if isinstance(validation_texts, str):
                validation_texts = [validation_texts]
            if text_files:
                validation_texts = files_to_texts(validation_texts)

            texts_length = sum(len(t) for t in validation_texts)
            validation_sequence = BiLSTM_Sequence(validation_texts,
                                                  validation_batch_size,
                                                  self.encoder)
            self.logger.info(f'Validating on {len(validation_texts)} texts '
                             + f'of total length {texts_length} with '
                             + f'{validation_batch_size=}')
        else:
            self.logger.info('Training without validation')

        if tensorboard_log_dir is not None:
            log_dir = tensorboard_log_dir \
                + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            tensorboard_callback = tf.keras.\
                callbacks.TensorBoard(log_dir=log_dir, **tfargs)
            callbacks = [tensorboard_callback]
            self.logger.info(
                f'Writing Tensorboard log to {log_dir=} with {tfargs=}')
        else:
            callbacks = []

        if csv_log_dir is not None:
            csv_log_name = csv_log_dir + f'{self.name}_log_{timestamp()}.csv'
            csv_logger = tf.keras.callbacks.CSVLogger(
                csv_log_name, separator=",", append=True)
            callbacks.append(csv_logger)
            self.logger.info(
                f'Writing CSVLogger callback log to {csv_log_name=}')

        history = self.model.fit(
            x=sequence_object,
            validation_data=validation_sequence,
            epochs=num_epochs,
            callbacks=callbacks
        )

        self.logger.info(
            f'Training completed. Metrics: {json.dumps(history.history)}')
        return history.history

    def evaluate(self, texts, text_files=False,
                 batch_size=VALIDATION_BATCH_SIZE):
        '''
        Evaluate model on an iterable of texts. If text_files is True,
        the elements of the iterable are interpreted as file names which
        are opened and read. Otherwise they are treated as the input strings
        on which the model is evaluated.
        Return metrics dict returned by Model.evaluate()
        '''
        # If a single string is provided in texts, treat it as a
        # one-element list.
        if isinstance(texts, str):
            texts = [texts]
        if text_files:
            texts = files_to_texts(texts)
        texts_length = sum(len(t) for t in texts)
        self.logger.info(f'Evaluating model on {len(texts)} texts of '
                         + f'total length {texts_length} with {batch_size=}')
        sequence_object = BiLSTM_Sequence(texts, batch_size, self.encoder)
        metrics = self.model.evaluate(x=sequence_object, return_dict=True)
        self.logger.info(
            f'Evaluation completed. Results: {json.dumps(metrics)}')
        return metrics

    def predict_on_string(self, sequence, target_index=None,
                          return_metric='p'):
        '''
        Get model's predictions for all possible output elements at
        position target_index in the specified sequence as a Numpy array.
        If a target_index is not specified, the target position is the
        position of the element following a full left context window
        the width of which is set in the language model.
        If either the part of the sequence to the left or to the right of
        the target is longer than the left and right context parameter of
        the model respectively, then their final and initial subsequences
        of the appropriate length are used respectively.
        If either the part of the sequence to the left of the target is
        shorter than the model's left context parameter, or the part to
        the right is shorter than the right context parameter, then the
        necessary number of padding tokens are added to the sequence
        at the beginning and/or the end.
        By default, the prediction is the probability distribution of
        every possible token at target_index. If return_metric is set
        to "perplexity" or "perpl", the probability for each token
        is converted into perplexity. If return_metric is "entropy" or
        "s", the entropy of the probability distribution is returned
        instead of the distribution itself.
        '''
        if target_index is not None:
            target_index = [target_index]
        return self.predict_targets([sequence], target_index,
                                    return_metric=return_metric)[0]

    def predict_targets(self, sequences, target_indices=None,
                        return_metric='p', batch_size=DEFAULT_BATCH_SIZE):
        '''
        Get model's predictions for all possible output tokens at
        the target index position in each sequence as a Numpy array.
        Sequences is an iterable of sequences. The optional
        target_indices is an iterable of the same length as "sequences".
        Each target index specifies the target position in the
        corresponding sequence, i.e. the first index for the first
        sequence, the second index for the second sequence, etc. Each
        sequence is processed as explained in the docstring of
        predict_on_string().
        Return a Numpy matrix, or, if entropy is returned, an array of
        predictions. Each row of the matrix (each element of the array
        in the case of entropy) corresponds to the target position in
        the respective sequence, i.e. the first row (element) for the
        first sequence, etc.
        By default, the predictions contain a token probability
        distribution for each sequence element. If return_metric is set
        to "perplexity" or "perpl", the probability for each token
        is converted into perplexity. If return_metric is "entropy" or
        "s", the entropy of each probability distribution is returned
        instead of the distribution itself.
        '''
        assert return_metric in ('p', 'perpl', 'perplexity', 'entropy', 's')

        if return_metric == 'perplexity':
            return_metric = 'perpl'
        elif return_metric == 'entropy':
            return_metric = 's'

        # If a single string is provided as sequence, treat it as a
        # one-element list.
        if isinstance(sequences, str):
            sequences = [sequences]

        if target_indices is None:
            # Target index points to the first character after a full
            # left context for each sequence.
            target_indices = [self.encoder.left_context] * len(sequences)
        elif isinstance(target_indices, int):
            target_indices = [target_indices]

        preds = self.predict_subsequences(sequences,
                                          start_indices=target_indices,
                                          end_indices=target_indices,
                                          token_dicts=False,
                                          batch_size=DEFAULT_BATCH_SIZE)

        return np.array([pred[return_metric][0] for pred in preds])

    def predict_substrings(self, strings, substrings=None,
                           batch_size=DEFAULT_BATCH_SIZE):
        '''
        Return probability distributions, perplexities and entropies for
        every character position in a specified substring of each string,
        as well as the average perplexity and accuracy scores for each
        substring.
        If no substrings are specified, every character position of each
        string is evaluated.
        The return value is a list of dicts, each dict containing the
        above-mentioned information for a string-substring pair:
        {"p": a list of len(substring) dicts, with the possible characters
              as keys and their probabilities as values,
         "perpl": same as "p", but with character perplexities as values,
         "s": an array of len(substring) entropies,
         "substr-perpl": overall perplexity of the substring,
         "acc": prediction accuracy on the substring}

        "Strings" and "substrings" must be iterables of the same length.
        Every substring must be contained in the string at the same index.
        It is always the first occurrence of the substring in the string
        that is evaluated if the substring appears several times in it.
        Strings are padded to the left or right as required if the string's
        prefix or suffix preceding and following the substring respectively
        are shorter than the context windows required by the model.

        This method only works for character-level models, i.e. the
        tokens of which are characters and the sequences of which are
        strings.
        '''
        if substrings is None:
            return self.predict_subsequences(strings, batch_size=batch_size)
        if isinstance(strings, str):
            strings = [strings]
        if isinstance(substrings, str):
            substrings = [substrings]

        start_indices = []
        end_indices = []
        for ss, s in zip(substrings, strings):
            start = s.index(ss)
            start_indices.append(start)
            end_indices.append(start + len(ss) - 1)

        return self.predict_subsequences(strings, start_indices=start_indices,
                                         end_indices=end_indices,
                                         batch_size=batch_size)

    def predict_subsequences(self, sequences, start_indices=None,
                             end_indices=None, token_dicts=True,
                             batch_size=DEFAULT_BATCH_SIZE):
        '''
        Return probability distributions, perplexities and entropies for
        every token position in a specified subsequence of each sequence,
        as well as the average perplexity and accuracy scores for each
        subsequence.
        The return value is a list of dicts, each dict containing the
        above-mentioned information for a sequence-subsequence pair:
        {"p": a list of len(subsequence) dicts, with the possible tokens
              as keys and their probabilities as values,
         "perpl": same as "p", but with character perplexities as values,
         "s": an array of len(subsequence) entropies,
         "substr-perpl": overall perplexity of the subsequence,
         "acc": prediction accuracy on the subsequence}

        If "token_dicts" is False, a numpy matrix containing the
        predicted probabilities and perplexities is returned in "p" and
        "perpl" respectively instead of lists of dicts.
        Sequences are padded to the left or right as required if the
        sequence's prefix or suffix preceding and following the
        start and end index respectively are shorter than the context
        windows required by the model.

        This method only works for character-level models, i.e. the
        tokens of which are characters and the sequences of which are
        strings.
        '''
        if isinstance(sequences, str):
            sequences = [sequences]
        if start_indices is None:
            start_indices = [0] * len(sequences)
        if end_indices is None:
            end_indices = [len(seq) - 1 for seq in sequences]

        contexts = []

        for seq, start, end in zip(sequences, start_indices,
                                   end_indices):
            assert start <= end
            assert end < len(seq)
            padded_seq = self.encoder.pad(seq, pad_before=start,
                                          pad_after=end)

            left_padding_len = max(self.encoder.left_context - start, 0)
            start += left_padding_len
            end += left_padding_len

            # crop each string
            contexts.append(
                padded_seq[start - self.encoder.left_context:
                           end + self.encoder.right_context + 1])

        subseqs = [c[self.encoder.left_context:
                     -self.encoder.right_context] for c in contexts]
        assert len(subseqs) > 0

        num_preds = 0
        for ss in subseqs:
            assert len(ss) > 0
            num_preds += len(ss)

        # predict; result is a combined array of predictions for
        # all substrings
        if num_preds >= MIN_PREDICT_BATCH:
            sequence_object = BiLSTM_Sequence(
                contexts,
                batch_size=min(num_preds, batch_size),
                encoder=self.encoder, padded=False)
            y = self.model.predict(x=sequence_object, verbose=0)
        else:
            y = []
            for s in contexts:
                left_X, right_X, _ = self.encoder.encode(s, padded=False)
                for pred in self.model([left_X, right_X], training=False):
                    y.append(pred.numpy())
            y = np.array(y)

        # encode the true characters in the target substrings
        yhat = []
        yhat_numeric = []
        for token in chain(*subseqs):
            yhat.append(self.encoder.output_encoder.encode(token))
            yhat_numeric.append(self.encoder.output_encoder.to_int(token))
        yhat = np.array(yhat)
        yhat_numeric = np.array(yhat_numeric)
        y_numeric = y.argmax(axis=1)
        y[y < PerElementPerplexity.PROB_FLOOR] =\
            PerElementPerplexity.PROB_FLOOR
        true_probs = np.max(yhat * y, axis=1)

        # split up prediction and target arrays into subarrays
        # that correspond to each string-substring pair

        # at what index does each substring begin?
        subseq_indices = list(accumulate(len(ss)
                                         for ss in subseqs))

        # split, then remove empty split at the end
        ys = np.split(y, subseq_indices)[:-1]
        true_ps = np.split(true_probs, subseq_indices)[:-1]
        y_numerics = np.split(y_numeric, subseq_indices)[:-1]
        yhat_numerics = np.split(yhat_numeric, subseq_indices)[:-1]

        return_dicts = []

        # generate output dicts or matrices and calculate summary
        # metrics for each string-substring pair
        for y_sub, true_p_sub, y_num_sub, yhat_num_sub \
                in zip(ys, true_ps, y_numerics, yhat_numerics):

            if token_dicts:
                p_list = []
                perpl_list = []

                for token in y_sub:
                    p_list.append({self.encoder.code_to_character[i]: p
                                   for i, p in enumerate(token)})
                    perpl_list.append({self.encoder.code_to_character[i]: 1 / p
                                       for i, p in enumerate(token)})
            else:
                p_list = y_sub
                perpl_list = 1 / y_sub

            return_dicts.append(
                {"p": p_list,
                 "perpl": perpl_list,
                 "s": -np.sum(y_sub * np.log2(y_sub), axis=1),
                 "substr-perpl": 2 ** (-np.log2(true_p_sub).sum()
                                       / len(y_sub)),
                 "acc": sum(y_num_sub == yhat_num_sub) / len(y_num_sub)
                 }
            )

        return return_dicts

    def estimate_alternatives(self, sequence, target_index=None):
        '''
        Return a dict containing all possible output tokens as keys
        and the probability of their occurrence at the target index as
        estimated by the model as the corresponding values.
        Regarding possible values of return_metric, see the docstring of
        predict_on_string().
        '''
        target_pred = self.predict_on_string(sequence, target_index,
                                             return_metric='p')
        return {self.encoder.code_to_character[i]: prob
                for i, prob in enumerate(target_pred)}

    def predict_next(self, left_seq, right_seq, m=1):
        '''
        Return a list of the m most likely candidates for the target
        element between the left and the right sequence ordered by
        decreasing probability.
        If either of the two sequences is longer than the left and right
        context parameter of the model, then their final and initial
        subsequences of the appropriate length are used respectively.
        If either of the sequences is shorter than the respective context
        parameter, then the necessary number of padding tokens are
        inserted to the left and to the right respectively.
        The right sequence must be passed without reversing it, i.e. read
        from left to right.
        '''
        y = self.predict_on_string(left_seq + " " + right_seq,
                                   target_index=len(left_seq))

        # Sort possible output tokens by their descending predicted
        # probability.
        sorted_tokens =\
            np.array(self.encoder.code_to_character)[np.argsort(y)[::-1]]

        return list(sorted_tokens)[:m]

    def estimate_prob(self, sequence, target_index=None):
        '''
        Estimate the probability of the element of the input string
        at position target_index (by default, the element after the
        left context window).
        See the docstring of predict_on_string() on the handling of strings
        that are longer or shorter on either side of the target than the
        length of the context expected by the model.
        '''
        if target_index is None:
            target_index = self.encoder.left_context
        target = sequence[target_index]
        y = self.predict_on_string(sequence, target_index)

        return y[self.encoder.output_to_numeric(target)]

    def pred_estimate(self, sequence, target_index=None):
        '''
        Does the same as predict_next plus estimate_prob.
        Returns the *most likely* target element given the left and right
        context along with the estimated probability of the *actual*
        target element.
        See the docstring of predict_on_string() on the handling of strings
        that are longer or shorter on either side of the target than the
        length of the context expected by the model.
        '''
        if target_index is None:
            target_index = self.encoder.left_context
        target = sequence[target_index]
        y = self.predict_on_string(sequence, target_index)

        sorted_tokens =\
            np.array(self.encoder.code_to_character)[np.argsort(y)[::-1]]

        target_prob = y[self.encoder.output_to_numeric(target)]

        return sorted_tokens[0], target_prob

    def metrics_on_string(self, sequence, padded=False, print_string=False,
                          batch_size=DEFAULT_BATCH_SIZE):
        '''
        Calculate overall prediction accuracy and average perplexity
        metrics on input sequence.
        If padded is True, padding is added to the left and right of the
        input sequence, and metrics are calculated for all token positions
        of the sequence. If padded is False, the initial and final tokens
        of the sequence the number of which corresponds to the left and right
        context parameter of the language model respectively are treated as
        context only and are not evaluated.
        Optionally print the predicted next element to standard output
        while processing the string.
        This method should only be used to evaluate relatively short sequences
        (maybe up to a few 10s of elements to be predicted), or if printing
        the predicted next elements is explicitly required for somewhat
        longer texts.
        The evaluate() method should be used to evaluate long texts or entire
        corpora.
        If per-token probability distributions, perplexities and entropies
        are needed instead of summary metrics, the predict_all() or
        predict_substrings() methods should be used.
        '''
        min_len = (self.encoder.left_context
                   + self.encoder.right_context + 1)

        if padded:
            preds = self.predict_subsequences([sequence], token_dicts=False,
                                              batch_size=DEFAULT_BATCH_SIZE)
        elif len(sequence) < min_len:
            raise ValueError(f'Length of input sequence "{sequence}" is only'
                             + f" {len(sequence)}, but must be at least "
                             + f"{min_len}. Either evaluate a longer string, "
                             + "or set padded=True.")
        else:
            preds = self.predict_subsequences(
                [sequence], start_indices=[self.encoder.left_context],
                end_indices=[len(sequence)
                             - (self.encoder.right_context + 1)],
                token_dicts=False, batch_size=DEFAULT_BATCH_SIZE)

        if print_string:
            print(self.encoder.decode(preds[0]['p']))

        return preds[0]['acc'], preds[0]['substr-perpl']

    def string_perplexity(self, sequence, padded=False):
        '''
        Return average perplexity metric on input sequence,
        like metrics_on_string().
        '''
        return self.metrics_on_string(sequence, padded=padded)[1]


class BiLSTM_Encoder:
    '''
    Class for encoder objects which encode input sequences as numeric
    NumPy arrays which can be used in the input and output layers of
    bidirectional recurrent neural language models (e.g. LSTMs).
    The input sequence to be encoded can be a character string for
    character-level language models, or it can be a list of word tokens
    for word-level language models.
    The encoder converts the input sequence into a sequence of windows,
    each window consisting of a left and a right context centered around
    a central target (output) element.

    For encoding the inputs, the encoder can either use the custom codes
    for each input token as supplied by the input_encoder, or it can
    operate in "embedding" mode, which means that it simply converts each
    token into an arbitrary integer identifier (which is also retrieved
    from the input_encoder), which will then be used as lookup key by
    a Keras Embedding layer.
    The central output tokens are always one-hot encoded. The codes are
    retrieved from the output_encoder for each token.

    The core of this class is the encode() method, which returns three
    NumPy arrays: for the left input contexts, the central characters
    and the right input contexts respectively, which have the following
    shapes:

    left_X: (number_of_windows, left_context_width,
             input_encoding_dimension)
    y: (number_of_windows, output_encoding_dimension)
    right_X: (number_of_windows, right_context_width,
              input_encoding_dimension)

    In "embedding" mode, the input_encoding_dimension element is dropped,
    i.e. the shapes are:
    left_X: (number_of_windows, left_context_width)
    right_X: (number_of_windows, right_context_width)

    The 0th dimension corresponds to the windows. Elements of the arrays
    that share the same 0th index make up a set of left context, center
    and right context that belong together.
    The "context width" dimension of the left and right context return
    arrays represents the "time steps" to be processed by the recurrect
    neural network, i.e. each element of the left and right context
    surrounding each target element. The initial time step corresponds
    to the element farthest from the central element (i.e. the left edge
    of the left context and the right edge of the right context
    respectively), and the final time step is the element closest to the
    target element in both the left and the right array.
    '''

    def __init__(self, input_encoder, output_encoder,
                 left_context, right_context, logger=None):
        '''Initialise encoder'''
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.padding_token = input_encoder.PADDING
        self.start_token = input_encoder.START_TEXT
        self.end_token = input_encoder.END_TEXT
        self.mask_token = input_encoder.MASKING

        try:
            self.char_to_int = input_encoder.input_char_to_int
        except AttributeError:
            self.char_to_int = None

        self.left_context = left_context    # width of left context
        self.right_context = right_context  # width of right context
        self.left_padding = ([self.padding_token] * (left_context - 1)
                             + [self.start_token])
        self.right_padding = ([self.end_token]
                              + [self.padding_token] * (right_context - 1))

        self.WHITE_SQUARE = '\u25a1'
        self.code_to_character = ([self.WHITE_SQUARE] +
                                  list(output_encoder.num_code_dict.keys()))

        if logger is not None:
            logger.info(f"Encoder created. {left_context=} {right_context=} "
                        + f"{input_encoder.code_dimension=} "
                        + f"{output_encoder.code_dimension=}")
            if self.char_to_int is not None:
                logger.info("Encoding input in embedding mode, "
                            + f'{len(self.char_to_int)=}')

    def pad(self, sequence, pad_before=0, pad_after=-1,
            left=True, right=True):
        '''
        Add padding tokens on one side or both.
        If pad_before is specified, only the number of padding tokens
        are added which are necessary to complete a context windows of
        the width specified for the language model to the left of the
        pad_before position. For example, if pad_before is 2, and the
        left context width is 10, then padding of length 8 is added
        before the sequence. Same applies for pad_after, which specifies
        the position to the right of which a context window is filled up
        with padding as required.
        Padding is only added if necessary. For example, if pad_before
        is 15, and left context is 10, then the sequence already contains
        the required number of elements to the left of the pad_before
        position to fill a left context window, so no padding is added.
        By default, padding is added to fill a context window to the left
        of the initial and to the right of the final element of the
        sequence respectively.
        '''
        return_seq = list(sequence)
        if pad_after == -1:
            pad_after = len(sequence) - 1
        left_padding_len = max(self.left_context - pad_before, 0)
        right_padding_len = max(self.right_context
                                - (len(sequence) - (pad_after + 1)), 0)

        if left and left_padding_len:
            return_seq = self.left_padding[-left_padding_len:] + return_seq
        if right and right_padding_len:
            return_seq = return_seq + self.right_padding[:right_padding_len]
        return return_seq

    def encode(self, sequence, call_by_reference=None, start=0,
               padded=True):
        '''
        Return an encoded repesentation of an input sequence to be used
        as input to a bidirectional LSTM language model.
        If padded is True, padding is added before and after the sequence
        so that each element of the sequence will serve as a target
        element, i.e. the number of windows in the return arrays becomes
        equal to the length of the input sequence.
        The last left padding token and the first right padding token
        are special, non-zero tokens that only occur as context and
        never as target tokens. These indicate start and end of text
        respectively. The point of these is that they are not ignored if
        Keras masking is used. The "normal" padding tokens have all-zero
        features and are thus skipped by Keras masking layers. Thus when
        predicting the first token of a text while using Keras masking,
        the input in the left context window would be empty if it
        consisted of all-zero padding tokens (since these would all be
        ignored), and the first LSTM layer taking this empty input would
        crash the model with an error. The "start of text" tokens is
        simply a special non-zero padding token that is not skipped by
        Keras masking and thus prevents crashing the model on the first
        token of the input text due to empty input. The same applies
        mutatis mutandis to the "end of text" token and the right
        context of the last token of the input text.
        If no padding is added, then the initial and final elements of
        the sequence are only used for context but not as target
        elements.
        If three array references are passed in the call_by_reference
        argument, the encode function writes the encodings into these
        arrays and returns nothing. This should improve efficiency
        slightly if the data returned by the encoder are combined into a
        larger array containing the encodings of a large number of
        sequences.
        '''

        if padded:
            sequence = self.pad(sequence)
        sequence_length = len(sequence) - (self.left_context +
                                           self.right_context)

        try:  # EAFP style
            input_codes = [self.char_to_int.get(e, len(self.char_to_int))
                           for e in sequence]
        except AttributeError:
            input_codes = [self.input_encoder.encode(e) for e in sequence]

        # The left context window goes up to but not including
        # the final element (e.g. character), which is itself
        # followed by the right padding, if padded.
        # If not padded, the left context window goes up to (not
        # including) the last element before the start of the
        # right context window.
        left_windows = windowed(input_codes[:-(self.right_context + 1)],
                                self.left_context)

        # Right context windows start from the second real element
        # after the left padding, if padded.
        # If not padded, the left context window goes up to the last
        # element before the start of the right context window.
        right_windows = windowed(input_codes[(self.left_context + 1):],
                                 self.right_context)

        if call_by_reference is None:
            left_X = np.array([w for w in left_windows])

            # Reading direction of each right context window is reversed
            # using the go_backwards parameter of the backward LSTM layer,
            # i.e. the right-hand input should not be reversed here.
            # Going backward on the right is intuitively correct and
            # in technical terms essentially entails that the memory of the
            # element closest to the target is the freshest.
            right_X = np.array([w for w in right_windows])

            # Regardless of whether the input sequence was padded or not,
            # all target characters are preceded by a left and a right
            # context.
            y = np.array([self.output_encoder.encode(e) for e in
                          sequence[self.left_context:-self.right_context]])

            return left_X, right_X, y
        else:
            # NumPy arrays passed in call by reference are respectively:
            # left_X, right_X and y
            # The point here is that only the necessary lists are created,
            # but they are neither copied into new Numpy arrays, nor
            # assigned to variables, so processing is somewhat faster.
            call_by_reference[0][start:start + sequence_length] =\
                [w for w in left_windows]
            call_by_reference[1][start:start + sequence_length] =\
                [w for w in right_windows]
            call_by_reference[2][start:start + sequence_length] =\
                [self.output_encoder.encode(c)
                 for c in sequence[self.left_context:-self.right_context]]

    def decode(self, output_sequence, oov_character=None):
        '''
        Decode a y Numpy array returned by a model's predict function
        to text. I.e. for each position of the output, the output
        element (character or word) which was assigned the highest
        probability by the model in that position is output.
        By default an OOV output is rendered as a
        white square (the standard replacement for an unsupported
        Unicode character). A different OOV output character can
        be specified using the oov_character argument.
        '''
        if oov_character is None:
            code_to_character = self.code_to_character
        else:
            code_to_character = self.code_to_character.copy()
            code_to_character[0] = oov_character
        numeric_codes = output_sequence.argmax(axis=1)
        return ''.join(code_to_character[x] for x in numeric_codes)

    def output_to_numeric(self, output_char):
        '''Get numeric output code of a single character.'''
        return self.output_encoder.num_code_dict.get(output_char, 0)


class BiLSTM_Sequence(tf.keras.utils.Sequence):
    '''
    Keras sequence class for feeding encoded text data to a
    bidirectional LSTM model for fitting or evaluating it.
    'Texts' is a list or other iterable of text strings which
    will be each padded optionally before processing. A text can
    be anything from a single sentence to a long document.
    Batch size determines the number of context windows around a
    central token (character or word) the encodings of which are
    stored in memory at a time.
    '''
    def __init__(self, texts, batch_size, encoder, padded=True):
        self.context_length = encoder.left_context + encoder.right_context

        # If a single string is provided in texts, treat it as a
        # one-element list.
        if isinstance(texts, str):
            texts = [texts]

        if padded:
            texts = (encoder.pad(t) for t in texts)

        # Skip inputs that are shorter than a
        # full context window.
        # This means texts of length 0 before padding,
        # or short texts if no padding was added.
        self.texts = [t for t in texts
                      if len(t) >= (self.context_length + 1)]

        # The lengths of the context windows are substracted from the
        # whole character count of a text, i.e. "text length" is defined
        # as the number of full context windows in the text.
        self.text_lengths = [len(t) - self.context_length
                             for t in self.texts]
        self.batch_size = batch_size
        self.encoder = encoder
        self.padded = padded

        if encoder.char_to_int is None:
            self.left_X = np.zeros([batch_size,
                                    encoder.left_context,
                                    encoder.input_encoder.code_dimension])
            self.right_X = np.zeros([batch_size,
                                    encoder.right_context,
                                    encoder.input_encoder.code_dimension])
        else:  # an Embedding layer is used
            self.left_X = np.zeros([batch_size,
                                    encoder.left_context])
            self.right_X = np.zeros([batch_size,
                                    encoder.right_context])

        self.y = np.zeros([batch_size, encoder.output_encoder.code_dimension])

        # Determine starting positions of batches.
        # Note that the positions refer to the index of a full context
        # window in a text, which is equal to the starting position
        # of a full context window.
        self.batch_starts = [(0, 0)]  # (text index, starting character index)
        batch_pos = 0
        current_text = 0
        text_pos = 0
        while True:
            if (self.text_lengths[current_text] - text_pos <=
                    batch_size - batch_pos):
                # If the rest of the current text fits into the current batch
                batch_pos += self.text_lengths[current_text] - text_pos
                current_text += 1
                text_pos = 0
            else:
                # Put the part that fits into the current batch,
                # then start a new batch at that point.
                text_pos += batch_size - batch_pos
                self.batch_starts.append((current_text, text_pos))
                batch_pos = 0
            if current_text == len(self.text_lengths):
                # All texts done
                break

    def __len__(self):
        return math.ceil(sum(self.text_lengths) / self.batch_size)

    def __getitem__(self, idx):
        left_X, right_X, y = self.left_X, self.right_X, self.y
        start_text, start_pos = self.batch_starts[idx]

        if (idx == -1) or (idx == len(self) - 1):
            # create smaller arrays for last batch
            # subtract starting position of first text in the batch
            remaining_elements = -start_pos
            for i in range(start_text, len(self.text_lengths)):
                remaining_elements += self.text_lengths[i]

            if self.encoder.char_to_int is None:
                left_X = np.zeros([
                    remaining_elements,
                    self.encoder.left_context,
                    self.encoder.input_encoder.code_dimension])
                right_X = np.zeros([remaining_elements,
                                    self.encoder.right_context,
                                    self.encoder.input_encoder.code_dimension])
            else:  # an Embedding layer is used
                left_X = np.zeros([remaining_elements,
                                   self.encoder.left_context])
                right_X = np.zeros([remaining_elements,
                                    self.encoder.right_context])
            y = np.zeros([remaining_elements,
                          self.encoder.output_encoder.code_dimension])

            # Set end_text and end_pos so that the last text
            # is processed completely.
            end_text, end_pos = len(self.texts), 0
        else:
            # end_text inclusive, end_pos non-inclusive
            end_text, end_pos = self.batch_starts[idx + 1]

        if start_text == end_text:
            # Encode substring of a single text.
            # Since end_pos refers to the initial character of the
            # final context window in the text, the context length
            # must be added so that the required number of full
            # context windows are fed to the encoder.
            self.encoder.encode(self.texts[start_text][start_pos:end_pos +
                                                       self.context_length],
                                call_by_reference=(left_X, right_X, y),
                                padded=False)
        else:
            # Encode the final substring of the first text,
            # the initial substring of the final text,
            # and any entire texts in between.
            self.encoder.encode(self.texts[start_text][start_pos:],
                                call_by_reference=(left_X, right_X, y),
                                padded=False)
            window_index = self.text_lengths[start_text] - start_pos

            for j in range(start_text + 1, end_text):
                self.encoder.encode(self.texts[j],
                                    call_by_reference=(left_X, right_X, y),
                                    start=window_index,
                                    padded=False)
                window_index += self.text_lengths[j]

            if end_pos != 0:
                # end_pos is 0 if the start of the following batch
                # coincides with the start of the next text, or if there
                # is no next text at all (i.e. this is the end of the last
                # batch, since in that case end_text was set the number of
                # texts and end_pos to 0).
                # The text before end_text has been processed at the end of
                # the previous for loop, so if end_pos is 0, this batch is
                # already complete at this point.
                self.encoder.encode(self.texts[end_text][:end_pos +
                                                         self.context_length],
                                    call_by_reference=(left_X, right_X, y),
                                    start=window_index,
                                    padded=False)

        return ([left_X, right_X], y)


class PerElementPerplexity(tf.keras.metrics.Metric):
    '''
    Custom metric to calculate the per-element (typically per-character
    or per-word) perplexity score of a language model.
    '''
    # If the probability predicted for the true element is less than
    # PROB_FLOOR, it is replaced by PROB_FLOOR.
    # Some such value needs to be specified for elements the probability
    # of which is predicted to be 0, since otherwise no logarithm could
    # be calculated for these results, and thus no perplexity.
    # Note that since the values calculated by the softmax function
    # never add up exactly to 1, but rather to 1 plus or minus a number
    # in the order of 1e-7 or 1e-8 typically (due to floating point
    # calculation inaccuracies), if all values lower than a suitably
    # small PROB_FLOOR are rounded up to this number, the whole
    # set of these values represents a no worse approximation of a
    # probability distribution than the original softmax values.
    PROB_FLOOR = 1e-13

    def __init__(self, name='perplexity', **kwargs):
        super(PerElementPerplexity, self).__init__(name=name, **kwargs)
        self.sum_of_logs = 0
        self.num_predictions = 0

    def update_state(self, y_true, y_pred, sample_weight=None):
        true_probs = tf.reduce_sum(y_true * y_pred, axis=1).numpy()
        true_probs[true_probs < self.PROB_FLOOR] = self.PROB_FLOOR
        self.sum_of_logs -= np.log2(true_probs).sum()
        self.num_predictions += len(true_probs)

    def result(self):
        return 2 ** (self.sum_of_logs / self.num_predictions)

    def reset_state(self):
        self.sum_of_logs = 0
        self.num_predictions = 0
