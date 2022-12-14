from sys import argv
import lstm_model
from encode_characters import InputEncoder, OutputEncoder

input_enc = InputEncoder(file="input_encoder_with_mask_chars.json")
output_enc = OutputEncoder(file="output_encoder.json")
bilstm_model = lstm_model.BiLSTM_Model.load(argv[1],
                                            input_enc, output_enc)
result = bilstm_model.evaluate(['test_files/1.press_hu_nem_007.txt',
                                'test_files/1.press_hu_nem_009.txt',
                                'test_files/2.press_hu_nem_004_1998.txt',
                                'test_files/2.press_hu_promenad_003_2010.txt',
                                'test_files/2.press_hu_promenad_003_2011.txt'
                                ], text_files=True, batch_size=6000)
print(result)
