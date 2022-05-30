import sys

best_flag = False
for line in sys.stdin:
    if best_flag:
        if line == '\n':
            best_flag = False
        else:
            sys.stdout.write('\t' + line)
    elif line.startswith('Best:'):
        sys.stdout.write(line[6:])
        best_flag = True