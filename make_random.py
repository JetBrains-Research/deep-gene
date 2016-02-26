__author__ = 'atsky'

import random


f = open('negative_rnd.txt', 'w')
for i in xrange(30000):
    f.write(''.join(random.choice("atgc") for _ in range(4000)) + "\n")

f.close()
