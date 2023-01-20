'''
This module contains the data structures to represent a
language model that returns maximum likelihood
estimates of conditional probabilities of all n-grams
in a corpus up to some n specified at model creation.
For n-grams that were not seen during training,
the model tries to approximate the final character's
conditional probability based on the shorter n-grams
that make up the original n-gram.
'''

import os
import sys
import math
import random
import pickle
from io import StringIO
from more_itertools import windowed, split_into, chunked
from encode_characters import OutputEncoder

MAX_SEGMENT_LENGTH = 4
NUMBER_OF_SEGMENTS = 10000
CORPUS_DIR = '/path/to/corpus/'

if not CORPUS_DIR.endswith('/'):
    CORPUS_DIR += '/'


def main():
    '''
    Train a forward and a backward ngram model on a
    corpus consisting of plain text files.
    '''

    n = sys.argv[1]

    FORWARD_OUTPUT_FILE = f'{n}-gram-forward.model'
    BACKWARD_OUTPUT_FILE = f'{n}-gram-backward.model'

    output_enc = OutputEncoder(file='output_encoder.json')

    trie_forward = MultiNgramTrieRoot(output_enc, int(n))
    trie_backward = MultiNgramTrieRoot(output_enc, int(n))

    corpus_files = [fname for fname in os.listdir(CORPUS_DIR)
                    if fname.endswith('.txt')]

    # The corpus file is split into input strings ("segments") so that
    # all input strings end at the end of a line in the document,
    # but not all newlines are treated as the end of an input string,
    # instead up to MAX_SEGMENT_LENGTH lines are randomly combined into
    # one input string, so that the language model can learn where
    # newlines occur in texts.

    segment_lengths = random.choices(range(1, MAX_SEGMENT_LENGTH + 1),
                                     weights=range(1, MAX_SEGMENT_LENGTH + 1),
                                     k=NUMBER_OF_SEGMENTS)
    big_slice_length = sum(segment_lengths)

    for f_name in corpus_files:
        with open(CORPUS_DIR + f_name, encoding="utf-8") as infile:
            print("Working on", f_name)
            big_slices = chunked(infile, big_slice_length)
            for big_slice in big_slices:
                segments = split_into(big_slice, segment_lengths)
                # Each segment is an array of lines. Join these lines,
                # the strip newlines from both ends of these joined
                # strings. Newlines within a segment are
                # kept.
                joined_segments =\
                    (''.join(seg).strip('\n') for seg in segments
                     # discard any empty splits at the end of a "big slice":
                     if len(seg))

                for js in joined_segments:
                    replaced_string = trie_forward.replace_oov(js)
                    trie_forward.add_input(replaced_string)
                    trie_backward.add_input(replaced_string[::-1])

        trie_forward.cache_frequencies()
        trie_backward.cache_frequencies()

        print("Saving models")
        with open(FORWARD_OUTPUT_FILE, 'wb') as savefile:
            pickle.dump(trie_forward, savefile)
        with open(BACKWARD_OUTPUT_FILE, 'wb') as savefile:
            pickle.dump(trie_backward, savefile)


class MultiNgramTrie:
    '''
    Class to represent the nodes of the trie data structure
    that constitutes the language model.
    '''
    def __init__(self):
        '''Constructor'''
        self.children = {}
        self.total = 0

    def add(self, ngram):
        '''Add a single ngram to the model'''
        self.total += 1
        if len(ngram):
            if ngram[-1] not in self.children:
                self.children[ngram[-1]] = MultiNgramTrie()
            self.children[ngram[-1]].add(ngram[:-1])

    def prefix_count(self, prefix):
        '''Return the frequency count of the specified string'''
        if len(prefix):
            if prefix[-1] in self.children:
                return self.children[prefix[-1]].prefix_count(prefix[:-1])
            else:
                return 0
        else:
            return self.total


class MultiNgramTrieRoot(MultiNgramTrie):
    '''
    Class to represent the root of the n-gram model trie.
    This serves as the entry point to the model.
    '''
    PADDING = '\u0000'
    OOV = '\u0001'

    def __init__(self, char_encoder, n=3):
        '''Constructor. n is the order of the language model.'''
        super().__init__()
        self.n = n
        self.unigram_freqs = None
        self.bigram_freqs = None
        self.char_encoder = char_encoder

    def add_input(self, string):
        '''
        Process an input string, dividing it up into n-gram windows
        of the specified order, and adding each n-gram in turn to
        the language model. A single padding character is added
        at the beginning and the end of the string. When the
        initial part of the string is processed, n-grams shorter
        than n (a bigram, then a trigram, etc.) are added to the model
        until a whole n-gram fits.
        '''
        string = self.PADDING + string + self.PADDING
        for i in range(len(string)):
            substring = string[max(0, i - (self.n - 1)):i + 1]
            self.add(substring)

    def cache_frequencies(self):
        '''
        Store unigram and bigram frequencies in a static dictionary
        each. Run this once after populating the model!
        Required for estimating n-gram frequencies.
        '''
        self.bigram_freqs = {}
        self.unigram_freqs = {key: self.children[key].total / self.total
                              for key in self.children.keys()}
        for condition in self.children.keys():
            condition_counts = {}
            for key in self.children.keys():
                count = self.prefix_count(condition + key)
                if count == 0:
                    count = self.unigram_freqs[key]
                condition_counts[key] = count
            count_sum = sum(condition_counts.values())

            for key in condition_counts.keys():
                self.bigram_freqs.update({condition + key:
                                         condition_counts[key] / count_sum})

    def cond_prob(self, ngram, verbose=False):
        '''
        Return maximum likelihood estimate of the conditional probability
        of the last character of the n-gram given that it is preceded
        by the initial n-1 characters.
        '''
        ngram_count = self.prefix_count(ngram)
        if ngram_count == 0:
            if verbose:
                print(ngram, "not seen in corpus")
            return 0.0
        condition_count = sum(v.prefix_count(ngram[:-1])
                              for v in self.children.values())
        if verbose:
            print(ngram, "ngram count", ngram_count, ',',
                  "condition count", condition_count, ',',
                  "conditional probability", ngram_count / condition_count)
        return ngram_count / condition_count

    def estimate_alternatives(self, ngram):
        '''
        Approximate the conditional probabilities of all alternatives of
        the last character of the n-gram by calculating their maximum
        likelyhood estimate, if that alternative ngram occurred in the
        training corpus, or by approximating each alternative n-gram's
        count by the final alternative character's bigram frequency
        given the penultimate character (i.e. a count smaller than zero)
        if that alternative n-gram did not occur.
        The ML estimates and bigram approximations are normalised
        to sum to 1 to yield a proper probability distribution.
        '''
        if len(ngram) > self.n:
            return self.backoff_alternatives(ngram)

        if len(ngram) == 1:
            return self.unigram_freqs

        if len(ngram) == 2:
            return {key: self.bigram_freqs[ngram[0] + key]
                    for key in self.children.keys()}

        condition = ngram[:-1]
        alt_target_counts = {}

        for alt_target in self.children.keys():
            # actual n-gram count
            alt_target_counts[alt_target] =\
                                self.prefix_count(condition + alt_target)
            # approximation based on bigram frequency
            # if the n-gram was not seen in the corpus
            if alt_target_counts[alt_target] == 0:
                alt_target_counts[alt_target] =\
                                self.bigram_freqs[condition[-1] + alt_target]

        # normalisation
        counts_sum = sum(alt_target_counts.values())
        alt_target_counts = {k: v / counts_sum
                             for k, v in alt_target_counts.items()}

        return alt_target_counts

    def estimate_prob(self, ngram, verbose=False):
        '''
        Approximate the conditional probability of the last character
        of the n-gram by returning the maximum likelyhood estimate
        if the ngram occurred in the training corpus, or by approximating
        its count by its bigram frequency given the penultimate character
        otherwise (i.e. a count smaller than zero).
        If the initial n-1-gram was not seen at all in the training corpus,
        this approach would mean backing off directly to the bigram model.
        Instead in this case the estimate backs off to the final n-1-gram,
        which should yield a more reasonable estimate.
        '''
        if len(ngram) > self.n:
            if verbose:
                print(ngram, "is longer than the model's order,",
                      "estimating for", ngram[1:])
        if self.prefix_count(ngram[:-1]) == 0:
            if verbose:
                print(ngram[:-1], "not seen in corpus,",
                      "estimating based on", ngram[1:-1])
            return self.estimate_prob(ngram[1:], verbose=verbose)
        else:
            alternatives = self.estimate_alternatives(ngram)
            return alternatives[ngram[-1]]

    def backoff_alternatives(self, ngram):
        '''
        If none of the alternatives of the ngram with the same prefix
        of length n-1 but all other possible final elements all
        had a count of 0 in the corpus (i.e. the n-1-gram prefix did
        not occur in the corpus at all), probability estimation
        backs off to an n-1-gram model.
        '''
        if len(ngram) > self.n or self.prefix_count(ngram[:-1]) == 0:
            return self.backoff_alternatives(ngram[1:])
        else:
            return self.estimate_alternatives(ngram)

    def predict_next(self, ngram, m=1):
        '''
        Return the m most likely next characters given the
        preceding ngram.
        Accordingly the length of the preceding n-gram should
        be at most n-1 if the order of the n-gram model is n.
        Predicts based on shorter preceding n-grams (in the limit,
        based on the stored bigram frequencies) if the specified n-gram
        did not appear in the training corpus.
        '''

        # return unigram frequencies
        if len(ngram) == 0:
            unigrams_desc = sorted(self.children.keys(),
                                   key=lambda x: self.children[x].total,
                                   reverse=True)
            return unigrams_desc[:m]

        # Add a dummy space character at the end so that
        # its alternatives will be considered.
        alternatives = self.backoff_alternatives(ngram + ' ')
        best_next = sorted(alternatives.keys(),
                           key=lambda x: alternatives[x],
                           reverse=True)[:m]
        return best_next

    def replace_oov(self, string):
        '''
        Replace rare characters (for which the encoder would
        return a 0 code) in a string by an OOV character.
        '''
        return_string = StringIO(newline=None)

        for c in string:
            if c in self.char_encoder.num_code_dict.keys():
                return_string.write(c)
            else:
                return_string.write(self.OOV)
        return_string.seek(0)
        return return_string.read()

    def metrics_on_string(self, string, padded=False):
        '''
        Calculate prediction accuracy and per-character perplexity
        metrics on input string.
        '''
        string = self.replace_oov(string)

        # No prediction is calculated for the first n-1
        # characters unless padded.
        num_predictions = len(string) - (self.n - 1)
        correct_guesses = 0
        sum_of_logs = 0

        if padded:
            string = self.PADDING + string + self.PADDING
            # Process first bigram to (n-1)-gram including
            # the initial padding character.
            for i in range(1, self.n - 1):
                ngram = string[:i + 1]
                target = ngram[-1]

                alternatives = self.backoff_alternatives(ngram)
                best_next = sorted(alternatives.keys(),
                                   key=lambda x: alternatives[x],
                                   reverse=True)[0]

                if best_next == target:
                    correct_guesses += 1
                sum_of_logs -= math.log2(alternatives[target])
            # No prediction is calculated for the
            # initial padding character, but for
            # everything else.
            num_predictions = len(string) - 1

        windows = windowed(string, self.n)

        for ngram in windows:
            ngram = ''.join(ngram)
            target = ngram[-1]

            alternatives = self.backoff_alternatives(ngram)
            best_next = sorted(alternatives.keys(),
                               key=lambda x: alternatives[x],
                               reverse=True)[0]

            if best_next == target:
                correct_guesses += 1
            sum_of_logs -= math.log2(alternatives[target])

        return (correct_guesses / num_predictions,      # accuracy
                2 ** (sum_of_logs / num_predictions))   # perplexity

    def string_perplexity(self, string):
        '''Calculate perplexity score on an input string.'''
        string = self.replace_oov(string)

        num_predictions = len(string) - (self.n - 1)
        sum_of_logs = 0
        windows = windowed(string, self.n)
        for window in windows:
            alternatives = self.backoff_alternatives(''.join(window))
            sum_of_logs -= math.log2(alternatives[window[-1]])

        return (2 ** (sum_of_logs / num_predictions))


class BiMultiNgramModel:
    '''Class for bidirectional multi-n-gram language models.'''

    def __init__(self, forward_model, backward_model):
        self.forward_model = forward_model
        self.backward_model = backward_model

    def combine_alternatives(self, forward_ngram, backward_ngram,
                             combine_function=None):
        '''
        Combine the probability distributions of the alternatives
        of the final element of the forward n-gram and those of the
        backward n-gram.
        Any combining function with a single list argument can be
        passed, obvious choices are max, sum and math.prod.
        The prediction accuracy and perplexity metrics for multiplication
        are generally better than for summing or taking the max,
        so the math.prod function is applied by default.
        '''
        if combine_function is None:
            combine_function = math.prod

        forward_alt =\
            self.forward_model.backoff_alternatives(forward_ngram)
        backward_alt =\
            self.backward_model.backoff_alternatives(backward_ngram)

        combined_alternatives = {}

        for key in forward_alt.keys():
            combined_alternatives[key] =\
                combine_function([forward_alt[key],
                                  backward_alt[key]])

        # normalisation
        total_probs = sum(combined_alternatives.values())
        combined_alternatives = {k: v / total_probs
                                 for k, v in combined_alternatives.items()}

        return combined_alternatives

    def predict_next(self, forward_ngram, backward_ngram,
                     combine_function=None, m=1):
        '''
        Predict the m most likely candidates for the next character
        by combining the predicted conditional probabilities for all
        alternative targets based on a forward and a backward
        n-gram model respectively, and returning the alternative
        with the highest combined probability.

        Note that the forward_ngram and backward_ngram arguments
        are the n-1-grams after which the n-th element (between
        the forward and the backward n-gram) will be predicted.
        The backward n-gram must be passed in reverse direction,
        i.e. right to left.
        If strings shorter than n-1 are passed, the appropriate
        lower-order backoff model is used.

        Any combination function with a single list argument
        can be passed, obvious choices are max, sum and math.prod.
        Of these, math.prod tends to yield best results.
        '''

        # Add an extra space as a dummy target for which
        # the alternatives are calculated.
        combined_alternatives = self.combine_alternatives(forward_ngram + " ",
                                                          backward_ngram + " ",
                                                          combine_function)

        best_next = sorted(combined_alternatives.keys(),
                           key=lambda x: combined_alternatives[x],
                           reverse=True)[:m]

        return best_next

    def estimate_prob(self, string, target_index=None, combine_function=None):
        '''
        Apply the combine_function to the probabilities estimated for each
        alternative target by the forward and the backward model
        respectively, then normalise the combined probabilities to add to 1,
        and return the normalised predicted probability for the actual target.

        By default the string argument must have length (m + n) - 1, where
        m is the order of the forward n-gram model, and n is the order of the
        backward model.
        The estimate is calculated for the final element of the initial m-gram,
        which is also the final element of the final n-gram.
        E.g. for two n-gram models of order 5, a string of length 9 would have
        to be passed, and the estimate is calculated for the middle element.

        Alternatively, if an integer target index is passed, the estimate
        is calculated for the target character at that index, and the input
        for the forward and backward model are the substrings:
        - up to and including the target (forward), and
        - starting from and including the target (backward)
        respectively.
        '''

        if target_index is None:
            target_index = self.forward_model.n - 1

        target = string[target_index]
        forward_ngram = string[:target_index + 1]
        backward_ngram = string[target_index:][::-1]

        combined_alternatives = self.combine_alternatives(forward_ngram,
                                                          backward_ngram,
                                                          combine_function)

        return combined_alternatives[target]

    def pred_estimate(self, string, target_index=None, combine_function=None):
        '''
        Does the same as predict_next plus estimate_prob, but is slightly
        slower than either of the two by themselves and much faster than
        executing one after the other. Used to check prediction accuracy
        along with estimated probability of the true target.
        Returns the *most likely* target element given the left and right
        context along with the estimated probability of the *actual*
        target element.
        '''
        if target_index is None:
            target_index = self.forward_model.n - 1

        target = string[target_index]
        forward_ngram = string[:target_index + 1]
        backward_ngram = string[target_index:][::-1]

        combined_alternatives = self.combine_alternatives(forward_ngram,
                                                          backward_ngram,
                                                          combine_function)
        best_next = sorted(combined_alternatives.keys(),
                           key=lambda x: combined_alternatives[x],
                           reverse=True)[0]

        return (best_next, combined_alternatives[target])

    def metrics_on_string(self, string, padded=False, combine_function=None,
                          print_string=False):
        '''
        Calculate prediction accuracy and per-character perplexity
        metrics on input string.
        Optionally print the predicted next characters to standard output while
        processing the string.
        '''

        if len(string) < ((self.forward_model.n) +
                          (self.backward_model.n) - 1):
            raise ValueError('Input string "' + string + '" is too short.')

        string = self.forward_model.replace_oov(string)

        # No prediction is calculated for the first and last n-1
        # characters unless padded.
        num_predictions = (len(string) -
                           (self.forward_model.n + self.backward_model.n) + 2)
        correct_guesses = 0
        sum_of_logs = 0

        if padded:
            string = (self.forward_model.PADDING +
                      string +
                      self.forward_model.PADDING)
            # Process first bigram to (n-1)-gram including
            # the initial padding character plus its right context.
            print_string_suffix = ''
            for target_index in range(1, self.forward_model.n - 1):
                substring = string[:target_index + self.backward_model.n]
                next_pred, prob_estimate =\
                    self.pred_estimate(substring, target_index,
                                       combine_function)
                if next_pred == string[target_index]:
                    correct_guesses += 1
                if print_string:
                    sys.stdout.write(next_pred)
                sum_of_logs -= math.log2(prob_estimate)

            # Same from the end for the final padding.
            end_backward = string[-(self.forward_model.n +
                                  self.backward_model.n - 2):][::-1]
            for target_index in range(1, self.backward_model.n - 1):
                substring = end_backward[:target_index + self.forward_model.n]
                next_pred, prob_estimate =\
                    self.pred_estimate(substring, target_index,
                                       combine_function)
                if next_pred == substring[target_index]:
                    correct_guesses += 1
                print_string_suffix += next_pred
                sum_of_logs -= math.log2(prob_estimate)

            # No prediction is calculated for the first
            # padding character in each direction, but for
            # everything else.
            num_predictions = len(string) - 2

        windows = windowed(string,
                           self.forward_model.n + self.backward_model.n - 1)
        target_index = self.forward_model.n - 1

        for window in windows:
            window = ''.join(window)
            next_pred, prob_estimate =\
                self.pred_estimate(window, combine_function=combine_function)

            if next_pred == window[target_index]:
                correct_guesses += 1
            if print_string:
                sys.stdout.write(next_pred)
            sum_of_logs -= math.log2(prob_estimate)

        if print_string:
            if padded:
                sys.stdout.write(print_string_suffix[::-1])
            print()

        return (correct_guesses / num_predictions,      # accuracy
                2 ** (sum_of_logs / num_predictions))   # perplexity

    def string_perplexity(self, string):
        if len(string) < ((self.forward_model.n) +
                          (self.backward_model.n) - 1):
            raise ValueError('Input string "' + string + '" is too short.')

        string = self.forward_model.replace_oov(string)

        num_predictions = (len(string) -
                           (self.forward_model.n + self.backward_model.n) + 2)
        sum_of_logs = 0
        windows = windowed(string,
                           self.forward_model.n + self.backward_model.n - 1)

        for window in windows:
            prob_estimate = self.estimate_prob(''.join(window))
            sum_of_logs -= math.log2(prob_estimate)

        return (2 ** (sum_of_logs / num_predictions))


class MultiNgramModelNonrec:
    '''
    Non-recursive version of a multi-ngram model.
    Should run about 5 times faster if properly optimised.
    '''
    PADDING = '\u0000'

    def __init__(self, n=3):
        '''Constructor. n is the order of the language model.'''
        self.ngrams = {'': 0}
        self.n = n

    def add(self, ngram):
        '''Add a single ngram to the model'''
        if len(ngram):
            if ngram not in self.ngrams:
                self.ngrams[ngram] = 1
            else:
                self.ngrams[ngram] += 1
            self.add(ngram[1:])
        else:
            self.ngrams[''] += 1

    def prefix_count(self, ngram):
        '''Return the frequency count of the specified string'''
        return self.ngrams.get(ngram, 0)

    def add_input(self, string):
        '''
        Process an input string, dividing it up into n-gram windows
        of the specified order, and adding each n-gram in turn to
        the language model. A single padding character is added
        at the beginning and the end of the string. When the
        initial part of the string is processed, n-grams shorter
        than n (a bigram, then a trigram, etc.) are added to the model
        until a whole n-gram fits.
        '''
        string = self.PADDING + string + self.PADDING

        # Process prefixes of up to n-1 characters
        for i in range(self.n - 1, 1, -1):
            self.add(string[0:i])

        for i in range(len(string)):
            substring = string[max(0, i - (self.n - 1)):i + 1]
            self.add(substring)


if __name__ == '__main__':
    main()
