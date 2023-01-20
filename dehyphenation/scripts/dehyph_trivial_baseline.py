"""
Trivial baseline for dehyphenation: delete all hyphen+\\n pairs.
"""

import sys


BREAKING_HYPHEN_LABEL = 1
# XXX should be imported from consts.py


def main():
    """Main."""
    for line in sys.stdin:
        line = line.rstrip('\n')
        code = f'\t{{{BREAKING_HYPHEN_LABEL}}}' if line.endswith('-') else ''
        print(f'{line}{code}')


if __name__ == '__main__':
    main()
