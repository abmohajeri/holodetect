import random
import subprocess

clean = "ac"
raw = "ar"
random_samples = 3000

lines = subprocess.Popen(
    'diff ' + clean + '.csv ' + raw + '.csv --unchanged-line-format="" --old-line-format="" --new-line-format="%dn,"',
    shell=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT).stdout.read().decode("utf-8")

num_lines = sum(1 for line in open(clean + '.csv'))
raw_lines = str(lines).strip(',').split(',')
clean_lines = [str(k) for k in range(1, num_lines) if k not in raw_lines]
clean_lines = random.sample(clean_lines, random_samples)
raw_lines += clean_lines
random.shuffle(raw_lines)
specified_lines = ['0'] + raw_lines

file = open(clean + '.csv')
lines = []
for pos, l_num in enumerate(file):
    if str(pos) in specified_lines:
        lines.append(l_num)
with open(clean + '_new.csv', 'w') as f:
    for item in lines:
        f.write("%s" % item)

file = open(raw + '.csv')
lines = []
for pos, l_num in enumerate(file):
    if str(pos) in specified_lines:
        lines.append(l_num)
with open(raw + '_new.csv', 'w') as f:
    for item in lines:
        f.write("%s" % item)
