# -*- coding:utf-8 -*-
'''
# STEP1: get title
import json

with open('new_baidu_title.json', 'w') as fw:
    with open('key_url_title_final.result.json', 'r') as fr:
        for line in fr:
            page = json.loads(line.strip());
            fw.write(page['title']['h1'].encode('utf-8') + '#####' + page['title']['h2'].encode('utf-8') + '\n')
print 'Done'
'''

'''
# STEP2: match title
baidu = {}
with open('baidu-instance-concept.dat', 'r') as baidu_fr:
    for line in baidu_fr:
        words = line.strip().split('\t')
        baidu[words[0]] = words[min(1, len(words)-1)]

print 'len(baidu)', len(baidu)

hudong = {}
with open('hudong-instance-concept.dat', 'r') as hudong_fr:
    for line in hudong_fr:
        words = line.strip().split('\t')
        hudong[line.strip().split('\t')[0]] = words[min(1, len(words)-1)]

print 'len(hudong)', len(hudong)

new_baidu = {}
with open('new_baidu_title.json', 'r') as new_baidu_fr:
    for line in new_baidu_fr:
        new_baidu[line.strip().replace('#####', '')] = line.strip()

print 'len(new_baidu)', len(new_baidu)

baidu_U_hudong = list(set(baidu.keys()).union(set(hudong.keys())))
print 'len(baidu_U_hudong)', len(baidu_U_hudong)

nBaidu_N_2 = list(set(baidu_U_hudong).intersection(set(new_baidu.keys())))
print 'len(nBaidu_N_2)', len(nBaidu_N_2)

cnt = 0
with open('new_baidu-instance-concept.dat', 'w') as fw:
    for key in nBaidu_N_2:
        if key in baidu:
            if len(baidu[key]) != 0:
                fw.write(new_baidu[key] + '\t' + baidu[key] + '\n')
                cnt += 1
        elif key in hudong:
            if len(hudong[key]) != 0:
                fw.write(new_baidu[key] + '\t' + hudong[key] + '\n')
                cnt += 1
        if cnt%100000==0:
            print '___', cnt

print 'cnt', cnt
'''

'''
# STEP3: get sub_key_url_title_final.json
import json
import pdb

new_baidu = {}
with open('new_baidu-instance-concept.dat', 'r') as fr:
    for line in fr:
        words = line.strip().split('\t')
        new_baidu[words[0]] = words[1].split(';')

print 'len(new_baidu)', len(new_baidu)

cnt = 0
with open('sub_key_url_title_final.result.json', 'w') as fw:
    with open('key_url_title_final.result.json', 'r') as fr:
        for line in fr:
            page = json.loads(line.strip())
            title = page['title']['h1'].encode('utf-8') + '#####' + page['title']['h2'].encode('utf-8')
            if title in new_baidu:
                cnt += 1
                page['category'] = json.loads(json.dumps(new_baidu[title]))
                fw.write(json.dumps(page, ensure_ascii=False).encode("utf-8")+'\n')
                if cnt%10000 == 0:
                    print '___', cnt

print 'cnt:', cnt
'''
