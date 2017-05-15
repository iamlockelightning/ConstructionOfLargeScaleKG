# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle

'''
taxonomyTree = pickle.load(file('taxonomy.pkl', 'rb'))['Tree']
taxonomyTypes = pickle.load(file('taxonomy.pkl', 'rb'))['Types']
maxl = -1
minl = 10000
def adepth(node, lev):
    global maxl
    global minl
    if len(taxonomyTree[node]['son']) == 0:
        if lev > maxl:
            maxl = lev
        if lev < minl:
            minl = lev
    for son in taxonomyTree[node]['son']:
        adepth(son, lev+1)

adepth('Root', 0)
print minl, maxl
'''

mentionDict = pickle.load(file('mentionDict.pkl', 'rb'))
#
# with open('mentionDict.dat', 'w') as fw:
#     for k in mentionDict:
#         fw.write(k + '\t' + mentionDict[k]['url'] + '\t' + '::;'.join(mentionDict[k]['mentions']) + '\n')

a = [0] * 10000
for k in mentionDict:
    a[len(mentionDict[k]['mentions'])] += 1

for i in range(100):
    print a[i],

# import pdb;
# pdb.set_trace()
