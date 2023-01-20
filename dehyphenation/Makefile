
SHELL:=/bin/bash

all:
	@echo "choose explicit target = type 'make ' and press TAB"

S=scripts
I=tests/inputs
O=tests/outputs

TEST_FILE=all_hvg

# ===== MAIN STUFF

### change to local directory containing a large unhyphenated corpus
UNHYPHENATED_CORPUS_DIR=path/to/corpus

HUNSPELL_URL=https://cgit.freedesktop.org/libreoffice/dictionaries/plain/hu_HU/hu_HU

prepare: get_hunspell_data generate_corpus_word_count

get_hunspell_data:
	cd $S; rm -f hu_HU.aff hu_HU.dic ; wget $(HUNSPELL_URL).aff; wget $(HUNSPELL_URL).dic

generate_corpus_word_count:
	python3 $S/extract_words_from_corpus.py $(UNHYPHENATED_CORPUS_DIR)

# ----- test dehyph_perpl_dev.py (a version with development comments)

dehyph-perpl: dehyph-perpl-plain dehyph-perpl-verbose dehyph-perpl-eval

dehyph-perpl-plain:
	clear ; time cat dehyph_test_input.txt | python3 $S/dehyph_perpl_dev.py > dehyph_test_output.txt
dehyph-perpl-verbose:
	clear ; time cat dehyph_test_input.txt | python3 $S/dehyph_perpl_dev.py -v > dehyph_perpl.log
dehyph-perpl-eval:
	clear ; time cat dehyph_test_input.txt | python3 $S/dehyph_perpl_dev.py -e > dehyph_perpl.eval

# ----- obsolete

dehyph:
	clear ; time python3 dehyph.py | grep -v "[0-9]/[0-9].*ETA:" > dehyph.log

