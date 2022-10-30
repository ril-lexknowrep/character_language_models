
SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

I=test_files
O=outputs
A_TEXT = $I/mh14_comp.txt
B_TEXT = $I/mh15_comp.txt
MERGED_TEXT = $O/correct_ocr_output.txt

# ===== MAIN STUFF

uncompress_data:
	7z e character_language_models.7z
	7z e $I/test_files.7z -o$I

train_lstm:
	python3 train_lstm_object.py

correct_ocr:
	time python3 correct_ocr_diffs.py $(A_TEXT) $(B_TEXT) $(MERGED_TEXT)

filter_log:
	cat correct_ocr_diffs.log | python3 filter_best.py > filtered_log.txt

confusion_matrix:
	python3 calculate_confusion_matrix.py

# -----

use_lm:
	clear ; time python3 use_lm.py | grep -v "1/1" > use_lm.log

