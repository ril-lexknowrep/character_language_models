#!/bin/bash

HEAD=$1

GOLDDIR="tests/inputs"
GOLD="2.press_hu_promenad_003_2010_hyph.txt"

# eval only
#RUN=
# run + eval
RUN=--run

if [ -z "$RUN" ]; then
  echo
  echo "warning: doing only eval on prepared results! to run scripts set RUN=--run"
  echo
fi

PYTHON=python3
SCRIPTDIR=scripts

EVALSCRIPT=$SCRIPTDIR/eval.py

# functions

run() {
    time python3 $EVALSCRIPT --name "$1" --golddir $GOLDDIR --gold $GOLD \
    --python $PYTHON --scriptdir $SCRIPTDIR --script "$2" \
    --head $HEAD $RUN
}

run_files_as_param() {
    time python3 $EVALSCRIPT --name "$1" --golddir $GOLDDIR --gold $GOLD \
    --python $PYTHON --scriptdir $SCRIPTDIR --script "$2" \
    --head $HEAD --files-as-param $RUN
}

# --- trivial baseline
run trivial dehyph_trivial_baseline.py

# --- rule based solution
run_files_as_param rulebased "dehyphenate_rule_based.py -e"

# --- n-gram character language model
# running time on CPU on 100.000 lines: ~30m
run ngram "dehyph_perpl_ngram.py -e"

# --- lstm character language model
# running time on CPU on 100.000 lines: ~2h
run lstm "dehyph_perpl.py -e"

