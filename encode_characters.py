'''
Decompose characters into various features: a normalized
letter for each letter (keeping in mind long-short allophony,
as in a-á, long-short confusability as in ü-ű, and
non-confusability of o-ö and u-ü), and various presumpably
useful binary features.
The normalised versions of characters are transformed into
a one-hot vector each.
The original characters are encoded as a combination of
the normalised character plus binary features for
upper/lower case, Hungarian long vowel marking, and
non-Hungarian diacritic. Foreign letters based on the same
Latin letter plus different non-Hungarian diacritics receive
the same feature vector, e.g. è and ę.
Some non-distinctive class features are also added:
characters are classified by the binary features 'letter',
'dash', 'space' and 'digit', as these might be useful
for language modeling.
Finally the one-hot vectors of the normalised characters
are concatenated with the binary features. This represents
the final input encoding of each character.
Output encoding is done by one-hot vectors with no
normalisation or feature decomposition. This is because our
application only involves probability estimation of
substrings in actual texts, and for our purposes it is
sufficient for the language model to know that very rare
(e.g. non-Hungarian) characters are very rare; trying to
predict approximately e.g. whether the very rare character
should be upper or lower case, whether it is a letter or
what its normalised form could be beneficial for
text generation or prediction, but should be irrelevant
to our task.
All output characters (i.e. characters the frequency of
which we are trying to estimate) below a specified
frequency threshold are therefore collapsed into a generic
'OOV' character.
'''

import os
import sys
import unicodedata
import json

import pandas as pd
import numpy as np

from collections import Counter, defaultdict
from abc import ABC, abstractmethod

FREQ_EXPONENT = 6
INPUT_COUNT_THRESHOLD = 1000
FEATURE_NAMES = ['normalised', 'letter', 'upper',
                 'long', 'other_diacritic',
                 'space', 'dash', 'digit']


def main():
    corpus_dir = sys.argv[1]

    charcounter = Counter()

    print("Counting characters in corpus...")

    for fname in os.listdir(corpus_dir):
        if fname.endswith(".txt"):
            with open(corpus_dir + fname, encoding="utf-8") as infile:
                charcounter.update(infile.read())

    print("done\n")

    characters = [i[0] for i in charcounter.most_common()]

    ch_features = [[ch, charcounter[ch]] + list(character_to_features(ch))
                   for ch in characters]
    dframe = pd.DataFrame(ch_features,
                          columns=['raw', 'count'] + FEATURE_NAMES)
    print("*" * 40)
    print("Character feature table")
    print("*" * 40)
    print(dframe.to_string())

    # Some input characters are only distinguished by features
    # but have the same normalised form.
    # Rare normalised forms are represented as an OOV character,
    # but the binary features of the raw (unnormalised)
    # character are still encoded.

    norm_char_freqs = defaultdict(int)

    for feat_list in ch_features:
        character_freq = feat_list[1]
        norm_character = feat_list[2]
        norm_char_freqs[norm_character] += character_freq

    norm_input_chars = list(ch for ch, freq in norm_char_freqs.items()
                            if freq > INPUT_COUNT_THRESHOLD)

    # OOV normalised input will be encoded as 0
    input_numeric_code = {ch: i + 1
                          for i, ch in enumerate(norm_input_chars)}

    input_enc = InputEncoder(input_numeric_code)

    ch_codes = [input_enc.encode(character) for character in characters]
    chartable = np.zeros([len(ch_codes), input_enc.code_dimension + 1], int)
    for i, ch in enumerate(ch_codes):
        character = characters[i]
        chartable[i, 0] = input_numeric_code.get(normal_lower(character), 0)
        chartable[i, 1:] = ch_codes[i]

    print("*" * 40)
    print("INPUT CHARACTER BINARY CODES:")
    dframe = pd.DataFrame(chartable, index=characters)
    print(dframe.to_string())
    print("*" * 40)

    input_enc.save("input_encoder.json")

    # Rare output characters are identified as a generic OOV
    # character with code 0.
    total_char_count = sum(charcounter[ch] for ch in characters)
    count_threshold = total_char_count / 10 ** FREQ_EXPONENT
    output_chars = [ch for ch in characters
                    if charcounter[ch] > count_threshold]

    # OOV output will be encoded as 0
    output_numeric_code = {ch: i + 1
                           for i, ch in enumerate(output_chars)}

    output_enc = OutputEncoder(output_numeric_code)

    output_enc.save("output_encoder.json")

    print("OUTPUT CHARACTER BINARY CODES:")
    chartable = np.zeros([len(characters), output_enc.code_dimension], int)

    for i, character in enumerate(characters):
        chartable[i, :] = output_enc.encode(character)

    dframe = pd.DataFrame(chartable, index=characters)
    print(dframe.to_string())


class CharacterEncoder(ABC):
    '''Superclass for character encoders.'''

    def __init__(self, num_code_dict={}):
        '''
        Initialise object.
        A dictionary object containing a numerical identifier
        for each character to be encoded should be supplied.
        The numerical identifier must have values
        between 0 and the number of classes - 1.
        The initialiser's dictionary object argument should
        only be omitted if the encoder object will be used to load
        a saved encoder object state from file using the 'load'
        method.
        '''
        self.num_code_dict = num_code_dict
        self.one_hot_dimension = len(self.num_code_dict) + 1  # + 1 for OOV
        self.code_dimension = self.one_hot_dimension
        self.ch_encodings = {}

    def keys(self):
        '''Return the characters handled by the encoder'''
        return self.ch_encodings.keys()

    def items(self):
        '''Return pairs each consisting of a character and its encoding'''
        return self.ch_encodings.items()

    def save(self, fname):
        '''Save encoder state to a JSON file'''
        obj_state = [self.num_code_dict,
                     self.code_dimension,
                     list(self.ch_encodings.keys())]
        with open(fname, 'w') as save_file:
            save_file.write(json.dumps(obj_state))

    def load(self, fname):
        '''Load encoder state from a JSON file'''
        with open(fname) as save_file:
            (num_code_dict,
             code_dimension,
             ch_encodings_keys) = json.loads(save_file.read())
        self.num_code_dict = num_code_dict
        self.one_hot_dimension = len(num_code_dict) + 1
        self.code_dimension = code_dimension
        self.ch_encodings_keys = ch_encodings_keys
        self.ch_encodings = {}

    @abstractmethod
    def encode(self, character):
        '''Return encoding for a character'''
        pass


class InputEncoder(CharacterEncoder):
    '''Class to encode input characters as arrays of binary values.'''

    PADDING = '\u0000'  # Unicode null character
    START_TEXT = '\u0002'  # Unicode STX character
    END_TEXT = '\u0003'  # Unicode ETX character
    MASKING = '\u0005'   # Unicode ENQ character

    def __init__(self, num_code_dict={},
                 add_start_char=False, add_end_char=False,
                 add_mask_char=False, file=None, mask_padding=False):
        '''
        Initialise encoder.
        If the path and name of a JSON save file are specified in 'file',
        the encoder is initialised from data found in that file.
        Add character codes for special control characters that will be
        used during model training and prediction, especially padding,
        and optionally characters that indicate start and end of a text
        (at the right and left edge of a padding sequence respectively,
        to be used if the padding characters are masked away using
        Keras masking and thus ignored as timesteps) and a special
        masking character (which is crucially NOT ignored as a timestep,
        but rather serves as an "unknown", to be predicted character).
        '''
        if file is not None:
            self.load(file)
            return
        extended_codes = num_code_dict.copy()
        if add_start_char:
            extended_codes[self.START_TEXT] = len(extended_codes) + 1
        if add_end_char:
            extended_codes[self.END_TEXT] = len(extended_codes) + 1
        if add_mask_char:
            extended_codes[self.MASKING] = len(extended_codes) + 1
        super().__init__(extended_codes)
        len_binary_features = len(character_to_features('a')) - 1  # any char
        self.code_dimension = self.one_hot_dimension + len_binary_features

        # The padding character is encoded as a numpy zeros
        # array of the code dimension.
        self.ch_encodings[self.PADDING] = np.zeros(self.code_dimension)

    def encode(self, character):
        try:
            return self.ch_encodings[character]
        except KeyError:
            char_features = character_to_features(character)
            norm_character = char_features[0]

            # If the character's normalised form is not in the dictionary
            # supplied during instantiation, it's encoded as a generic
            # 'rare' OOV character, which gets 0 as its normalised
            # character code.
            character_code = self.num_code_dict.get(norm_character, 0)
            binary_features = char_features[1:]

            # One-hot encode the normalised character
            oh_array = np.zeros(self.code_dimension)
            oh_array[character_code] = 1

            # Add binary features
            oh_array[-len(binary_features):] = binary_features
            self.ch_encodings[character] = oh_array
            return oh_array

    def load(self, fname):
        super().load(fname)
        for key in self.ch_encodings_keys:
            self.encode(key)
        self.ch_encodings[self.PADDING] = np.zeros(self.code_dimension)


class OutputEncoder(CharacterEncoder):
    '''Class to encode output characters as one-hot vectors.'''

    def __init__(self, num_code_dict={}, file=None):
        '''
        Initialise encoder.
        If the path and name of a JSON save file are specified in 'file',
        the encoder is initialised from data found in that file.
        '''
        if file is not None:
            self.load(file)
            return
        super().__init__(num_code_dict)
        self.diag_matrix = np.diag(np.ones(self.one_hot_dimension))

    def encode(self, character):
        '''Return one-hot vector representation for the character'''
        # Rare output characters are identified as a generic OOV
        # character with code 0.
        return self.diag_matrix[self.num_code_dict.get(character, 0)]

    def to_int(self, character):
        '''Return the integer code for the character, 0 for OOV'''
        return self.num_code_dict.get(character, 0)

    def load(self, fname):
        super().load(fname)
        self.diag_matrix = np.diag(np.ones(self.one_hot_dimension))


def normal_lower(ch):
    '''
    Normalise a letter by lower-casing it and removing diacritics.
    Non-letters are returned unchanged.
    '''

    low = ch.lower()
    if low in 'öő':
        return 'ö'
    if low in 'üű':
        return 'ü'

    # The Unicode standard does not treat letters with strokes,
    # e.g. ø or ł, as combined characters consisting of a base
    # letter and a combining character. Therefore these have to
    # be normalised based on their Unicode name.
    try:
        unicode_stroke_suffix = ' WITH STROKE'
        ch_name = unicodedata.name(ch)
        if ch_name.endswith(unicode_stroke_suffix):
            ch_base = ch_name[-len(unicode_stroke_suffix) - 1]
            return ch_base.lower()
    except:
        pass

    norm_ch = unicodedata.normalize('NFD', low)
    return norm_ch[0]


def has_diacritic(ch):
    '''
    Return True if the character is a letter with a
    non-Hungarian diacritic.
    Return False for all letters of the Hungarian
    alphabet and for non-letters.
    '''

    try:
        ch_name = unicodedata.name(ch)
        if ch_name.endswith('WITH STROKE'):
            return True
    except:
        return False

    if ch.lower() in 'áéíóúöőüű':
        return False

    norm_ch = unicodedata.normalize('NFD', ch)
    if len(norm_ch) > 1:
        # ch_base = norm_ch[0]
        # diacritic = norm_ch[1]
        return True
    else:
        return False


def character_to_features(ch):
    '''Transform a character into a tuple of features'''
    return (
            normal_lower(ch),         # normalised character
            ch.isalpha(),             # is a letter
            ch.isupper(),             # is upper case
            ch.lower() in 'áéíóúőű',  # is long
            has_diacritic(ch),        # foreign letter with diacritic
            ch.isspace(),             # is whitespace
            unicodedata.category(ch) == 'Pd',  # is a dash
            ch.isdigit()              # is a digit
           )


if __name__ == '__main__':
    main()
