'''
The point of this script is to sample many contiguous short
passages from a corpus randomly and shuffle them, so that
the resulting excerpt from the corpus contains short
contiguous, typically relatively coherent sections from the
corpus, but these sections follow one another in random
order.

The intended use case is to shuffle a relatively diverse
training corpus so that the genres, topics, etc. are
relatively evenly shuffled all over the corpus, but at
the same time each short section remains coherent. If this
is used in training a language model, this means that
different parts of longer subcorpora (e.g. specific
novels) are seen at many different times during training
instead of the whole subcorpus in one go, a single time
during each training epoch. Since shuffling training data
is generally useful for any machine learning model, this
is the intended goal. At the same time, shuffling a
training corpus e.g. sentence by sentence would be
clearly detrimental, since this would make it difficult
to learn inter-sentence or more generally long-distance
dependencies, thematic coherence between words, etc. Thus
it makes sense to shuffle sections that are longer than
single sentences, especially ones that are about as long
as the processing window size of the language model.

This script randomly samples lines from the input stream
at SAMPLE_PROB. If a sampled line is shorter than
MIN_SEG_LEN, then the next line is also read and
concatenated with it, and so on, until the combined length
of the concatenated lines exceeds MIN_SEG_LEN.
The sampled strings (i.e. lines or sequences of lines) are
stored as elements of a list. 
After the entire input stream has been processed, this list
is shuffled, then output as a stream.
If SAMPLE_PROB is 1.0, the entire corpus is split up into
shorter sections of at least MIN_SEG_LEN characters which
consist of sequences of whole lines (i.e. no lines are
truncated, and then these sections are shuffled and
combined into a single file.
'''

import random
from sys import stdin, stdout, argv

random.seed = 222
SAMPLE_PROB = float(argv[1])
MIN_SEG_LEN = 300

subset = []
segment = ''
for line in stdin:
    if random.random() < SAMPLE_PROB:
        segment += line
        while len(segment) < MIN_SEG_LEN:
            try:
                segment += next(stdin)
            except StopIteration:
                pass
        subset.append(segment)
        segment = ''
random.shuffle(subset)

for line in subset:
    stdout.write(line)
