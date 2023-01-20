"""
General evaluation script.
"""

import argparse
import contextlib
import os
import sys

from sklearn import metrics


def main():
    """Main."""
    args = get_args()


    # --- Set parameters.

    # name of the experiment
    NAME = args.name

    # gold annotated material: 2 TAB-separated columns = item + label
    GOLDDIR = args.golddir
    GOLD = args.gold

    # python to use -- maybe in a venv!
    PYTHON = args.python

    # annotator script to test
    SCRIPTDIR = args.scriptdir
    ANNOTATOR_SCRIPT = args.script

    # use to take only part of gold data
    HEAD = args.head
    try:
        SELECT = f'head -{HEAD}' if int(HEAD) > 0 else 'cat'
    except ValueError:
        SELECT = 'cat'

    FILES_AS_PARAM = args.files_as_param
    RUN = args.run

    THISDIR = os.getcwd()

    # output dir
    OUTDIR = f'{THISDIR}/eval/{NAME}.h{HEAD}__{GOLD}'
    with contextlib.suppress(FileExistsError): os.mkdir(OUTDIR)
    # basic files
    OUTGOLD = f'{OUTDIR}/gold'
    OUTINPUT = f'{OUTDIR}/input'
    OUTANNO = f'{OUTDIR}/anno'
    # helper files
    OUTEVALDATA = f'{OUTDIR}/eval_data'
    # result
    OUTMETRICS = f'{OUTDIR}/eval.txt'


    if RUN:
        # --- Run script to be evaluated on data.

        # mark which gold file was used
        os.system(f"echo \'{GOLDDIR}/{GOLD}\' > {OUTDIR}/INFO")

        # take some gold data
        os.system(f"cat {GOLDDIR}/{GOLD} | {SELECT} > {OUTGOLD}")

        # create input: get text, omit labels (by TAB)
        os.system(f"cat {OUTGOLD} | cut -d '	' -f 1 > {OUTINPUT}")

        # run from script's dir, to be safe
        os.chdir(SCRIPTDIR)
        if FILES_AS_PARAM:
            command = f"{PYTHON} {ANNOTATOR_SCRIPT} {OUTINPUT} {OUTANNO}"
        else:
            command = f"cat {OUTINPUT} | {PYTHON} {ANNOTATOR_SCRIPT} > {OUTANNO}"
        print(command)
        os.system(command)
        os.chdir(THISDIR)


    # --- Do evaluation itself.

    # get labels for evaluation == 2nd col from each file
    # XXX special step: ignore {0} (=NO_HYPHEN_LABEL)
    os.system(f"paste {OUTGOLD} {OUTANNO} | grep -v '{{0}}' | grep '{{[1-9]}}' | cut -d '	' -f 2,4 > {OUTEVALDATA}")

    # not relevant classes in gold
    NOT_RELEVANT_CLASSES = ['{5}', '{6}', '{7}', '{8}']

    # maga a kiértékelés eszerint:
    # https://towardsdatascience.com/multi-class-metrics-made-simple-part-i-precision-and-recall-9250280bddc2 szerint!

    y_true = []
    y_pred = []

    NA = 'n/a'

    # read data
    with open(OUTEVALDATA) as f:
        for line in f:
            values = line.rstrip('\n').split('\t')

            if len(values) == 1:
                t, p = values[0], NA
            elif len(values) == 2:
                t, p = values
            else:
                print(f'Bad value pair: [{values}]', f=sys.stderr)
                exit(1)

            # filter not relevant gold classes
            if t in NOT_RELEVANT_CLASSES:
                continue

            y_true.append(t)
            y_pred.append(p)

    # write results
    with open(OUTMETRICS, 'w') as f:
        print(metrics.confusion_matrix(y_true, y_pred), file=f)
        print(file=f)
        print(metrics.classification_report(y_true, y_pred, digits=3), file=f)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--name',
        help='name of the experiment',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--golddir',
        help='directory of the gold corpus',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--gold',
        help='gold corpus file name',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--python',
        help='a python interpreter to use, e.g. `python3` or a version in a venv',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--scriptdir',
        help='directory of the annotator script',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--script',
        help='annotator script (with switches)',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--head',
        help='run only on first HEAD lines from gold data, `--head all` means all data',
        required=True,
        type=str,
        default=argparse.SUPPRESS
    )
    parser.add_argument(
        '--files-as-param',
        help='`SCRIPT i o` instead of `cat i | SCRIPT > o`',
        action='store_true'
    )
    parser.add_argument(
        '--run',
        help='run before eval, if omitted just eval on prepared results',
        action='store_true'
    )

    return parser.parse_args()


if __name__ == '__main__':
    main()
