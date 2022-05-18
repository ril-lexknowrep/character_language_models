import math
import numpy as np
import tensorflow as tf
from more_itertools import windowed

DEFAULT_BATCH_SIZE = 256

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

class BiLSTM_Model:
    '''Class for bidirectional LSTM language models.'''

    def __init__(self, input_encoder, output_encoder,
                 left_context, right_context,
                 lstm_units=64, dense_neurons=64,
                 dropout_ratio=0.5,
                 verbose=True):
        # Configure bidirectional LSTM model
        input_left = tf.keras.Input(shape=(left_context,
                                           input_encoder.code_dimension),
                                    name="input_left")
        input_right = tf.keras.Input(shape=(right_context,
                                            input_encoder.code_dimension),
                                    name="input_right")

        forward_layer = tf.keras.layers.LSTM(lstm_units,
                                             name="lstm_forward")(input_left)
        forward_model = tf.keras.Model(inputs=input_left,
                                       outputs=forward_layer)

        backward_layer = tf.keras.layers.LSTM(lstm_units,
                                              name="lstm_backward",
                                              go_backwards=True)(input_right)
        backward_model = tf.keras.Model(inputs=input_right,
                                        outputs=backward_layer)

        bidirectional = tf.keras.layers.Concatenate()([forward_model.output,
                                                      backward_model.output])
        dense = tf.keras.layers.Dense(dense_neurons,
                                       activation="relu")(bidirectional)
        dropout = tf.keras.layers.Dropout(dropout_ratio)(dense)
        output_classifier = tf.keras.layers.Dense(output_encoder.code_dimension,
                                                  name="output",
                                                  activation="softmax")(dropout)

        self.model = tf.keras.Model(inputs=[forward_model.input,
                                              backward_model.input],
                                      outputs=output_classifier)

        if verbose:
            self.model.summary()

        self.model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy', PerElementPerplexity()],
                             run_eagerly=True)
        # run_eagerly is required for the .numpy() function in the
        # perplexity metric function to work.

        self.model.optimizer.learning_rate = BiLSTM_Model.learning_rate_decay()

        self.encoder = BiLSTM_Encoder(input_encoder, output_encoder,
                                      left_context=left_context,
                                      right_context=right_context)

    def learning_rate_decay():
        # Decay learning rate gradually from 1e-3 to 1e-5 until a total of
        # about STOP_DECAY_AFTER characters have been trained on.
        # Note that in one iteration the optimizer processes a whole batch,
        # so the specified number of characters needs to be divided by
        # the batch size to get the number of iterations in which the
        # learning rate will decay.

        STOP_DECAY_AFTER = 2e9
        steps = STOP_DECAY_AFTER // DEFAULT_BATCH_SIZE
        return tf.keras.optimizers.schedules.PolynomialDecay(
                                    initial_learning_rate=1e-3,
                                    decay_steps=steps,
                                    end_learning_rate=1e-5,
                                    power=2)

    def load(model_file, input_encoder, output_encoder):
        saved_model = tf.keras.models.load_model(model_file,
                                custom_objects={"PerElementPerplexity":
                                                PerElementPerplexity})

        left_context = saved_model.inputs[0].shape[1]
        right_context = saved_model.inputs[1].shape[1]
        input_dim = saved_model.inputs[0].shape[2]
        output_dim = saved_model.outputs[0].shape[1]

        assert input_encoder.code_dimension == input_dim
        assert output_encoder.code_dimension == output_dim

        saved_optimizer = saved_model.optimizer

        saved_model.compile(loss='categorical_crossentropy',
                             optimizer='adam',
                             metrics=['accuracy', PerElementPerplexity()],
                             run_eagerly=True)
        # Compile creates a new optimizer, so data on training progress,
        # particularly the number of iterations so far, is lost.
        saved_model.optimizer = saved_optimizer

        return_model = BiLSTM_Model(input_encoder,
                                    output_encoder,
                                    left_context,
                                    right_context,
                                    verbose=False)
        return_model.model = saved_model
        
        return_model.model.optimizer.learning_rate =\
                                    BiLSTM_Model.learning_rate_decay()

        print("Loaded model:")
        return_model.model.summary()
        print("Trained for", int(return_model.model.optimizer.iterations),
              "iterations")

        return return_model

    def train(self, texts, batch_size=DEFAULT_BATCH_SIZE,
              validation_texts=None, num_epochs=1):
        '''
        Train model on an iterable of texts. Each of these can be
        as short as single sentences or as long as documents,
        as required.
        Validation on a fraction of the training data does not
        seem to be particularly informative, so an iterable of
        validation texts can be optionally passed on which the
        model is validated after each training epoch. If such an
        iterable is not provided, there is no validation after
        training on each epoch.
        '''
        sequence_object = BiLSTM_Sequence(texts, batch_size,
                                              self.encoder)

        validation_sequence = None
        if validation_texts:
            validation_sequence = BiLSTM_Sequence(validation_texts,
                                                  VALIDATION_BATCH_SIZE,
                                                  self.encoder)

        self.model.fit(
            x=sequence_object,
            validation_data=validation_sequence,
            epochs=num_epochs
        )

    def predict_on_string(self, string, target_index=None):
        '''
        Get model's predictions for all possible output characters at position
        target_index in the specified string as a Numpy array.
        If either the part of the string to the left or to the right of the
        target is longer than the left and right context parameter of the model
        respectively, then their final and initial substrings of the appropriate
        length are used respectively.
        If either the part of the string to the left of the target is shorter
        than the model's left context parameter, or the part to the right is
        shorter than the right context parameter, then the necessary number of
        padding characters are added to the string at the beginning and/or
        the end.
        '''
        if target_index == None:
            target_index = self.encoder.left_context

        left_string = string[:target_index]
        left_string = left_string[-(self.encoder.left_context):]
        left_padding_len = max(0, self.encoder.left_context - len(left_string))
        left_string = (left_padding_len * self.encoder.padding_char +
                       left_string)

        right_string = string[target_index + 1:]
        right_string = right_string[:self.encoder.right_context]
        right_padding_len = max(0, self.encoder.right_context - len(right_string))
        right_string = (right_string +
                        right_padding_len * self.encoder.padding_char)

        target = string[target_index]

        string = left_string + target + right_string
        left_X, right_X, _ = self.encoder.encode(string, padded=False)

        preds = self.model.predict(x=[left_X, right_X])[0]

        preds[preds < PerElementPerplexity.PROB_FLOOR] =\
                                            PerElementPerplexity.PROB_FLOOR

        return preds

    def estimate_alternatives(self, string, target_index=None):
        '''
        Return a dict containing all possible output characters as keys
        and the probability of their occurrence at the target index as
        estimated by the model as the corresponding values.
        '''
        target_pred = self.predict_on_string(string, target_index)
        return {self.encoder.code_to_character[i]: prob
                                                for i, prob
                                                in enumerate(target_pred)}

    def predict_next(self, left_string, right_string, m=1):
        '''
        Predict the m most likely candidates for the target element
        between the left and the right string.
        If either of the two strings is longer than the left and right
        context parameter of the model, then their final and initial
        substrings of the appropriate length are used respectively.
        If either of the strings is shorter than the respective context
        parameter, then the necessary number of padding characters are
        inserted to the left and to the right.
        The right string must be passed without reversing it, i.e. read
        from left to right.
        '''
        y = self.predict_on_string(left_string + " " + right_string,
                                   target_index = len(left_string))

        # Sort possible output characters by their descending predicted
        # probability.
        sorted_chars =\
            np.array(self.encoder.code_to_character)[np.argsort(y)[::-1]]

        return list(sorted_chars)[:m]

    def estimate_prob(self, string, target_index=None):
        '''
        Estimate the probability of the element of the input_string at
        position target_index (by default, the element after the left context
        window).
        See the docstring of predict_on_string on the handling of strings
        that are longer or shorter on either side of the target than the
        length of the context expected by the model.
        '''
        if target_index is None:
            target_index = self.encoder.left_context
        target = string[target_index]
        y = self.predict_on_string(string, target_index)

        return y[self.encoder.output_to_numeric(target)]

    def pred_estimate(self, string, target_index=None):
        '''
        Does the same as predict_next plus estimate_prob.
        Returns the *most likely* target element given the left and right
        context along with the estimated probability of the *actual*
        target element.
        See the docstring of predict_on_string on the handling of strings
        that are longer or shorter on either side of the target than the
        length of the context expected by the model.
        '''
        if target_index is None:
            target_index = self.encoder.left_context
        target = string[target_index]
        y = self.predict_on_string(string, target_index)

        sorted_chars =\
            np.array(self.encoder.code_to_character)[np.argsort(y)[::-1]]

        target_prob = y[self.encoder.output_to_numeric(target)]

        return sorted_chars[0], target_prob

    def metrics_on_string(self, string, padded=False, print_string=False):
        '''
        Calculate prediction accuracy and per-element perplexity
        metrics on input string.
        Optionally print the predicted next element to standard output while
        processing the string.
        The input string is assumed to be relatively short. To get 
        '''

        if len(string) < (self.encoder.left_context +
                          self.encoder.right_context - 1):
            raise ValueError('Input string "' + string + '" is too short.')

        left_X, right_X, y = self.encoder.encode(string, padded=padded)
        y_numeric = y.argmax(axis=1)

        num_predictions = len(y)

        preds = self.model.predict(x=[left_X, right_X])
        preds_numeric = preds.argmax(axis=1)

        correct_guesses = sum(y_numeric == preds_numeric)

        true_probs = np.max(y * preds, axis=1)
        true_probs[true_probs < PerElementPerplexity.PROB_FLOOR] =\
                                     PerElementPerplexity.PROB_FLOOR

        sum_of_logs = -np.log2(true_probs).sum()

        if print_string:
            print(self.encoder.decode(preds))

        return (correct_guesses / num_predictions,      # accuracy
                2 ** (sum_of_logs / num_predictions))   # perplexity

    def string_perplexity(self, string):
        if len(string) < (self.encoder.left_context +
                          self.encoder.right_context - 1):
            raise ValueError('Input string "' + string + '" is too short.')

        left_X, right_X, y = self.encoder.encode(string, padded=False)

        num_predictions = len(y)

        preds = self.model.predict(x=[left_X, right_X])

        true_probs = np.max(y * preds, axis=1)
        true_probs[true_probs < PerElementPerplexity.PROB_FLOOR] =\
                                     PerElementPerplexity.PROB_FLOOR

        sum_of_logs = -np.log2(true_probs).sum()

        return (2 ** (sum_of_logs / num_predictions))


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
    It returns three NumPy arrays: for the left input contexts,
    the central characters and the right input contexts respectively,
    which have the following shapes:
    
    left_X: (number_of_windows, left_context_width, input_encoding_dimension)
    y: (number_of_windows, output_encoding_dimension)
    right_X: (number_of_windows, right_context_width, input_encoding_dimension)
    
    The 0th dimension corresponds to the windows. Elements of the arrays that
    have the same 0th index make up a set of left context, center and right
    context that belong together.
    The "middle" dimension of the left and right context return arrays
    represents the "time steps" of the recurrect neural network, i.e. each
    element of the left and right context surrounding each target element.
    The initial time step corresponds to the element farthest from the
    central element (i.e. the left edge of the left context and the
    right edge of the right context respectively), and the final time step
    is the element closest to the target element respectively.
    '''

    def __init__(self, input_encoder, output_encoder,
                 left_context, right_context):
        '''Constructor'''
        self.input_encoder = input_encoder
        self.output_encoder = output_encoder
        self.padding_char = input_encoder.PADDING_CHAR

        self.left_context = left_context    # width of left context
        self.right_context = right_context  # width of right context
        self.left_padding = [self.padding_char] * left_context
        self.right_padding = [self.padding_char] * right_context

        self.WHITE_SQUARE = '\u25a1'
        self.code_to_character = ([self.WHITE_SQUARE] +
                                  list(output_encoder.num_code_dict.keys()))

        print("Encoder created. Left context width:", left_context,
              "Right context width:", right_context)
        print("Input encoding dimension:", input_encoder.code_dimension,
              "Target encoding dimension:", output_encoder.code_dimension)

    def pad(self, sequence, left=True, right=True):
        '''Add padding characters on both sides.'''
        return_seq = list(sequence)
        if left:
            return_seq = self.left_padding + return_seq
        if right:
            return_seq = return_seq + self.right_padding
        return return_seq

    def encode(self, sequence, call_by_reference=None, start=0,
                        padded=True):
        '''
        Return an encoded repesentation of an input sequence to be used
        as input to a bidirectional LSTM language model.
        If padded is True, padding characters are added before and after the
        sequence so that each element of the sequence will serve as a target
        character, i.e. the number of windows in the return arrays becomes
        equal to the length of the input sequence.
        If no padding is added, then the initial and final elements of the
        sequence are only used for context but not as target elements.
        If three array references are passed in the call_by_reference argument,
        the encode function writes the encodings into these arrays and
        returns nothing. This should improve efficiency slightly if the data
        returned by the encoder are combined into a larger array containing
        the encodings of a large number of sequences.
        '''

        if padded:
            sequence = self.pad(sequence)
        sequence_length = len(sequence) - (self.left_context + 
                                           self.right_context)

        input_codes = np.array(list(map(self.input_encoder.encode, sequence)))
        
        # The left context window goes up to but not including
        # the final element (e.g. character) plus right padding if padded.
        # If not padded, the left context window goes up to the last
        # element before the start of the right context window.
        left_windows = windowed(input_codes[:-(self.right_context + 1)],
                                self.left_context)

        # Right context windows start from the second real element
        # after the left padding, if padded.
        # If not padded, the left context window goes up to the last
        # element before the start of the right context window.
        right_windows = windowed(input_codes[(self.left_context + 1):],
                                 self.right_context)

        if call_by_reference is None:
            left_X = np.array(list(map(np.array, left_windows)))
            # Reading direction of each right context window is reversed
            # using the gobackwards parameter of the backward LSTM layer,
            # i.e. the right-hand input should not be reversed.
            # Going backward on the right is intuitively correct and
            # in technical terms essentially entails that the memory of the
            # element closest to the target is the freshest.

            right_X = np.array(list(map(np.array, right_windows)))

            # Regardless of whether the input sequence was padded or not,
            # all target characters are preceded by a left and a right context.

            y = np.array(list(map(self.output_encoder.encode, 
                            sequence[self.left_context:-self.right_context])))

            return left_X, right_X, y
        else:
            # NumPy arrays passed in call by reference are respectively:
            # left_X, right_X and y
            call_by_reference[0][start:start + sequence_length] =\
                                            list(map(np.array, left_windows))
            call_by_reference[1][start:start + sequence_length] =\
                                            list(map(np.array, right_windows))
            call_by_reference[2][start:start + sequence_length] =\
                        list(map(self.output_encoder.encode,
                             sequence[self.left_context:-self.right_context]))

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
        return ''.join(map(lambda x : code_to_character[x], numeric_codes))

    def output_to_numeric(self, output_char):
        '''Get numeric output code of a single character.'''
        return self.output_encoder.num_code_dict.get(output_char, 0)


class BiLSTM_Sequence(tf.keras.utils.Sequence):
    '''
    Keras sequence class for feeding encoded text data to a
    bidirectional LSTM model for fitting or evaluating a model.
    Texts is a list or other iterable of units of texts which
    will be each padded optionally before processing, i.e. anything
    from single sentences to long documents.
    Batch size determines the number of time steps the encodings
    of which are stored in memory at a time.
    '''
    def __init__(self, texts, batch_size, encoder, padded=True):
        self.context_length = encoder.left_context + encoder.right_context
        if padded:
            self.texts = map(encoder.pad, texts)

        # Skip inputs that are shorter than a
        # full context window.
        # This means texts of length 0 before padding,
        # or short texts if no padding was added.
        self.texts = list(filter(lambda x : len(x) >= (self.context_length + 1),
                                 self.texts))

        # The lengths of the context windows are substracted from the
        # whole character count of a text, i.e. "text length" is defined
        # as the number of full context windows in the text.
        self.text_lengths = list(map(lambda x : len(x) - self.context_length,
                                self.texts))
        self.batch_size = batch_size
        self.encoder = encoder
        self.padded = padded
        self.left_X = np.zeros([batch_size,
                                encoder.left_context,
                                encoder.input_encoder.code_dimension])
        self.right_X = np.zeros([batch_size,
                                 encoder.right_context,
                                 encoder.input_encoder.code_dimension])
        self.y = np.zeros([batch_size, encoder.output_encoder.code_dimension])

        # Determine starting positions of batches.
        # Note that the positions refer to the index of a full context
        # window in a text, which is equal to the starting position
        # of a full context window.
        self.batch_starts = [(0,0)] # (text index, starting character index)
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

            left_X = np.zeros([remaining_elements,
                               self.encoder.left_context,
                               self.encoder.input_encoder.code_dimension])
            right_X = np.zeros([remaining_elements,
                                self.encoder.right_context,
                                self.encoder.input_encoder.code_dimension])
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
    # If the probability predicted for the true element is less than,
    # this number, it is replaced by this number.
    # Some such value needs to be specified for elements the probability
    # of which is predicted to be 0 anyway, since otherwise no
    # logarithm could be calculated for these results, and thus no
    # perplexity.
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
