import lstm_model
from encode_characters import InputEncoder, OutputEncoder
from lstm_model import PerElementPerplexity
from more_itertools import chunked
import numpy as np
from sys import stdin, stderr
from contextlib import redirect_stdout
import gc

import tensorflow as tf

PROB_FLOOR = 1e-13
LINES_AT_A_TIME = 10000
BATCH_SIZE = 4000

# new version
input_enc = InputEncoder(file='input_encoder.json')
output_enc = OutputEncoder(file='output_encoder.json')

# old version
# input_enc = InputEncoder()
# output_enc = OutputEncoder()
# input_enc.load("input_encoder.pickle")
# output_enc.load("output_encoder.pickle")

MODEL_FILE = 'bilstm_model_512_2_epoch.h5'

model = tf.keras.models.load_model(MODEL_FILE,
    custom_objects={"PerElementPerplexity": PerElementPerplexity})

with redirect_stdout(stderr):
    model.summary()

left_context = model.input[0].shape[1]
right_context = model.input[1].shape[1]

encoder = lstm_model.BiLSTM_Encoder(input_enc, output_enc,
                                    left_context, right_context)

chunks = chunked(stdin, LINES_AT_A_TIME)
for c in chunks:
    sequence_object = lstm_model.BiLSTM_Sequence([''.join(c)],
                                                 BATCH_SIZE,
                                                 encoder)
    with redirect_stdout(stderr):
        predictions = model.predict(x=sequence_object)
    predictions[predictions < PROB_FLOOR] = PROB_FLOOR
    entropy = -np.sum(predictions * np.log2(predictions), axis=1)
    for e in entropy:
        print(e)
    del entropy
    del predictions
    tf.keras.backend.clear_session()
    gc.collect()
