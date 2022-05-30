results = {
    'aa': 0,
    'ab': 0,
    'ac': 0,
    'ba': 0,
    'bb': 0,
    'bc': 0,
    'ca': 0,
    'cb': 0,
    'cc': 0,
    'cd': 0
}

with open('filtered_log_annot.txt', encoding='utf-8') as infile:
    infile.readline()
    for line in infile:
        cols = line.rstrip('\n').split('\t')
        if cols[0] != '':
            if len(cols[2]) != len(cols[0]):
                print("Rosszul annotálva:", line)
                continue
            preds = cols[0]
            targets = cols[2]
            for i, pred in enumerate(preds):
                target = targets[i]
                if pred in 'ab':
                    if target in 'ab':
                        results[pred + target] += 1
                    else:
                        results[pred + 'c'] += 1
                else:
                    if target in 'ab':
                        results['c' + target] += 1
                    elif pred == target:
                        results['cc'] += 1
                    else:
                        results['cd'] += 1

print(results)