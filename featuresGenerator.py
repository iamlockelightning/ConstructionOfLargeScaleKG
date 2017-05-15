# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle
import json
import copy
import threading

# http://stackoverflow.com/questions/956867/how-to-get-string-objects-instead-of-unicode-ones-from-json-in-python
def json_load_byteified(file_handle):
    return _byteify(
        json.load(file_handle, object_hook=_byteify),
        ignore_dicts=True
    )

def json_loads_byteified(json_text):
    return _byteify(
        json.loads(json_text, object_hook=_byteify),
        ignore_dicts=True
    )

def _byteify(data, ignore_dicts = False):
    # if this is a unicode string, return its string representation
    if isinstance(data, unicode):
        return data.encode('utf-8')
    # if this is a list of values, return list of byteified values
    if isinstance(data, list):
        return [ _byteify(item, ignore_dicts=True) for item in data ]
    # if this is a dictionary, return dictionary of byteified keys and values
    # but only if we haven't already byteified it
    if isinstance(data, dict) and not ignore_dicts:
        return {
            _byteify(key, ignore_dicts=True): _byteify(value, ignore_dicts=True)
            for key, value in data.iteritems()
        }
    # if it's anything else, return it in its original form
    return data

# http://blog.csdn.net/fred1653/article/details/51255530
class TrieNode(object):
    def __init__(self):
        """
        Initialize your data structure here.
        """
        self.data = {}
        self.extra_data = []
        self.is_word = False

class Trie(object):
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word, title):
        """
        Inserts a word into the trie.
        :type word: str
        :rtype: void
        """
        node = self.root
        for letter in word:
            child = node.data.get(letter)
            if not child:
                node.data[letter] = TrieNode()
            node = node.data[letter]
        node.is_word = True
        node.extra_data.append(title)

    def search(self, word):
        """
        Returns if the word is in the trie.
        :type word: str
        :rtype: bool
        """
        node = self.root
        for letter in word:
            node = node.data.get(letter)
            if not node:
                return False
        return node

    def starts_with(self, prefix):
        """
        Returns if there is any word in the trie
        that starts with the given prefix.
        :type prefix: str
        :rtype: bool
        """
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
            if not node:
                return False
        return True

    def get_start(self, prefix):
        """
        Returns words started with prefix
        :param prefix:
        :return: words (list)
        """
        def _get_key(pre, pre_node):
            words_list = []
            if pre_node.is_word:
                words_list.append(pre)
            for x in pre_node.data.keys():
                words_list.extend(_get_key(pre + str(x), pre_node.data.get(x)))
            return words_list

        words = []
        if not self.starts_with(prefix):
            return words
        if self.search(prefix):
            words.append(prefix)
            return words
        node = self.root
        for letter in prefix:
            node = node.data.get(letter)
        return _get_key(prefix, node)

# self.VM = pickle.load(file(self.dataStructureDir+ 'VM.pkl', 'rb'))
#
# pickle.dump(self.VM, file(self.dataStructureDir+'VM.pkl', 'wb'), True)
#
# self.VM = np.zeros((self.Cn, self.Cn), dtype=np.float32)
#
# pickle.dump({'W_AVG_D': self.W_AVG_D, 'W_Nij': self.W_Nij}, file(self.dataStructureDir+'W_AVGD_N.pkl', 'wb'), True)

class PklMaker:
    def __init__(self):
        print('I\'m a PklMaker~')

    def makeTaxonomyForCategory(self):
        print 'makeTaxonomy(self): Start.'
        taxonomyTree = {}
        taxonomyTypes = []
        with open('../taxonomy/baidu-taxonomy-20160720.dat', 'r') as fr:
            for line in fr:
                words = line.strip().split('\t')
                if words[0] in taxonomyTree and len(taxonomyTree[words[0]]['son'])>=len(words[1].split(';')):
                    continue
                taxonomyTree[words[0]] = {'son': words[1].split(';'), 'parent': None}
                taxonomyTypes.append(words[0])
                for w in words[1].split(';'):
                    taxonomyTypes.append(w)

        taxonomyTypes = list(set(taxonomyTypes))
        print '__len(taxonomyTree):', len(taxonomyTree)
        print '__len(taxonomyTypes):', len(taxonomyTypes)

        for node in taxonomyTypes:
            if node in taxonomyTree:
                for son in taxonomyTree[node]['son']:
                    if son in taxonomyTree:
                        taxonomyTree[son]['parent'] = node
                    else:
                        taxonomyTree[son] = {'son': [], 'parent': node}

        taxonomyTypes.remove('Root')
        a = []
        for (key, val) in taxonomyTree.items():
            a.append(key)
            for x in val['son']:
                a.append(x)
            if val['parent']==None:
                print '__', key, 'has parent:', val['parent']
            cnt = 0
            for (k, v) in taxonomyTree.items():
                if key in v['son']:
                    cnt += 1
            # print cnt
            if cnt >=2:
                print '__Over__', key

        print '__len(taxonomyTree):', len(taxonomyTree)
        print '__len(taxonomyTypes):', len(taxonomyTypes)
        print '__nodes num(check):', len(set(a))

        pickle.dump({'Tree': taxonomyTree, 'Types': taxonomyTypes}, file('taxonomy.pkl', 'wb'), True)
        print 'Done.'

    def splitSubPagesForCategory(self, num):
        print 'splitSubPages(self, num): Start...'
        taxonomyTypes = pickle.load(file('taxonomy.pkl', 'rb'))['Types']
        cnt = 0
        with open('sub_'+str(num)+'.result.json', 'w') as fw:
            with open('sub_key_url_title_final.result.json', 'r') as fr:
                for line in fr:
                    page = json_loads_byteified(line.strip())
                    category = page['category']
                    infobox = page['infobox']
                    use = True
                    for cate in category:
                        if (cate in taxonomyTypes) == False:
                            use = False
                    if use and len(infobox)>0:
                        fw.write(line)
                        cnt += 1
                        if cnt%1000 == 0:
                            print '__', cnt
                        if cnt == num:
                            break
        print 'Done.'

    def makeFeatureForCategory(self):
        beta = 2
        ratio = 5

        masses = {}
        # categoryCount = {}
        infoboxKeyCount = {}
        tagsCount = {}
        cnt = 0
        with open('sub_50000.result.json', 'r') as fr:
            for line in fr:
                cnt += 1
                if cnt%1000 == 0:
                    print '__', cnt
                page = json_loads_byteified(line.strip())
                title = page['title']['h1'] + '#####' + page['title']['h2']
                category = page['category']
                infobox = page['infobox']
                for key in infobox:
                    if (key in infoboxKeyCount) == False:
                        infoboxKeyCount[key] = 1
                    else:
                        infoboxKeyCount[key] += 1

                tags = []
                for tag in page['tags']:
                    if tag.find('[[') != -1:
                        tag = tag.split('||')[0].split('[[')[1]
                    tags.append(tag)
                    if (tag in tagsCount) == False:
                        tagsCount[tag] = 1
                    else:
                        tagsCount[tag] += 1

                masses[title] = {'category': category, 'infobox': infobox, 'tags': tags}

        infoboxKeyCount = sorted(infoboxKeyCount.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
        tagsCount = sorted(tagsCount.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

        infoboxKeyFeatures = []
        for (k, v) in infoboxKeyCount:
            if v >= beta:
                infoboxKeyFeatures.append(k)

        print 'len(infoboxKeyFeatures):', len(infoboxKeyFeatures), 'total:', len(infoboxKeyCount)

        tagsFeatures = []
        for (k ,v) in tagsCount:
            if v >= beta:
                tagsFeatures.append(k)

        print 'len(tagsFeatures):', len(tagsFeatures), 'total:', len(tagsCount)

        dataSet = {}
        for title in masses:
            features = np.zeros(len(tagsFeatures)+len(infoboxKeyFeatures), dtype=np.float32)
            for tag in masses[title]['tags']:
                if tag in tagsFeatures:
                    features[tagsFeatures.index(tag)] = 1.0

            for k in masses[title]['infobox']:
                if k in infoboxKeyFeatures:
                    features[len(tagsFeatures) + infoboxKeyFeatures.index(k)] = 1.0

            dataSet[title] = {'features': features, 'label': masses[title]['category']}

        trainSet = dict(dataSet.items()[len(dataSet)/ratio:])
        testSet = dict(dataSet.items()[:len(dataSet)/ratio])
        pickle.dump({'Train': self.splitDataSetForCategory(trainSet), 'Test': testSet}, file('dataSet.pkl', 'wb'), True)

    def splitDataSetForCategory(self, data_set):
        nodeDataSet = {}
        taxonomyTypes = pickle.load(file('taxonomy.pkl', 'rb'))['Types']
        taxonomyTree = pickle.load(file('taxonomy.pkl', 'rb'))['Tree']
        typesList = {}
        pos = 0
        neg = 0
        for title in data_set:
            for cate in data_set[title]['label']:
                if (cate in typesList)==False:
                    typesList[cate] = []
                typesList[cate].append(title)

        for t in taxonomyTypes:
            # Pos
            nodeDataSet[t] = {'pos': [], 'neg': []}
            if t in typesList:
                for e in typesList[t]:
                    pos += 1
                    nodeDataSet[t]['pos'].append(data_set[e])

            # Neg
            t_parent = taxonomyTree[t]['parent']
            for sib in taxonomyTree[t_parent]['son']:
                if sib != t and sib in typesList:
                    for e in typesList[sib]:
                        if (t in data_set[e]['label']) == False:
                            neg += 1
                            nodeDataSet[t]['neg'].append(data_set[e])

            temp_t = t
            while taxonomyTree[temp_t]['parent'] != None:
                if taxonomyTree[temp_t]['parent'] in typesList:
                    for e in typesList[ taxonomyTree[temp_t]['parent'] ]:
                        if (t in data_set[e]['label']) == False:
                            neg += 1
                            nodeDataSet[t]['neg'].append(data_set[e])
                temp_t = taxonomyTree[temp_t]['parent']

        print '__total pos:', pos
        print '__total neg:', neg
        return nodeDataSet

    def makeMentionDictForMissingLink(self):
        print 'makeMentionDictForMissingLink(self): Start...'
        mentionDict = {}
        links_pool = set()
        links_pool_count = {}
        pages = []
        cnt = 0
        with open('../key_url_title_final.result.json', 'r') as fr:
            for line in fr:
                page = json_loads_byteified(line.strip())
                pages.append(page)
                title = page['title']['h1'] + '#####' + page['title']['h2']
                url = page['url']
                mentionDict[title] = {'url': url, 'mentions': []}
                if len(page['synonym']) != 0:
                    for synonym in page['synonym']['to'].split('||'):
                        links_pool.add((synonym, url))
                        if synonym+'*****'+url not in links_pool_count:
                            links_pool_count[synonym+'*****'+url] = 0
                        links_pool_count[synonym+'*****'+url] += 1

                for w in page['links']:
                    k = w.split('||')[0].split('[[')[1]
                    v = 'http://baike.baidu.com' + w.split('||')[1].split(']]')[0]
                    links_pool.add((k, v))
                    if k+'*****'+v not in links_pool_count:
                        links_pool_count[k+'*****'+v] = 0
                    links_pool_count[k+'*****'+v] += 1

                links_pool.add((title.replace('#####', ''), url))
                if title.replace('#####', '')+'*****'+url not in links_pool_count:
                    links_pool_count[title.replace('#####', '')+'*****'+url] = 0
                links_pool_count[title.replace('#####', '')+'*****'+url] += 1

                cnt += 1
                if cnt%100000==0:
                    print '__pro__', cnt

        links_pool = list(links_pool)
        print links_pool[0:10]
        print '__load done.'

        trie = Trie()
        cnt = 0
        for title in mentionDict:
            cnt += 1
            trie.insert(mentionDict[title]['url'], title)
            if cnt % 100000 == 0:
                print "__bui__", cnt

        print '__build done.'

        cnt = 0
        c = 0
        for (k, v) in links_pool:
            c += 1
            node = trie.search(v)
            if node != False and node.is_word == True:
                for w in node.extra_data:
                    mentionDict[w]['mentions'].append( k+':::'+str(links_pool_count[k+'*****'+v]) )
                cnt += 1
            if c % 100000 == 0:
                print "__fin__", c
        print '__Total:', cnt

        for title in mentionDict:
            mentionDict[title]['mentions'] = list(set(mentionDict[title]['mentions']))
        pickle.dump(mentionDict, file('2_mentionDict.pkl', 'wb'), True)
        print '__makeMentionDictForMissingLink(self): Done! :D'


    def splitSubPagesForMissingLink(self, num):
        print 'splitSubPagesForMissingLink(self, num): Start...'
        mentionDict = pickle.load(file('mentionDict.pkl', 'rb'))
        m2e = {}
        for k in mentionDict:
            # print mentionDict[k]
            for m in mentionDict[k]['mentions']:
                mk = m.split(':::')[0]
                if mk != '':
                    if mk not in m2e:
                        m2e[mk] = []
                    m2e[mk].append(k)

        k3 = 0
        k3m = []
        k3e = []
        for m in m2e:
            m2e[m] = list(set(m2e[m]))
            if len(m2e[m]) == 3:
                k3m.append(m)
                for i in m2e[m]:
                    k3e.append(i)
                k3 += 1

        print '# k3 mention:', k3
        k3e = list(set(k3e))
        print '# k3 entity:', len(k3e)

        cnt = 0
        with open('sub_mlk3.result.json', 'w') as fw:
            with open('../key_url_title_final.result.json', 'r') as fr:
                for line in fr:
                    cnt += 1
                    if cnt%500000 == 0:
                        print '___', cnt
                    page = json_loads_byteified(line.strip())
                    title = page['h1'] + '#####' + page['h2']
                    if title in k3e:
                        fw.write(line)
        print 'Done.'


if __name__ == "__main__":
    pklMaker = PklMaker()

    # pklMaker.makeTaxonomyForCategory()
    # pklMaker.splitSubPagesForCategory(50000)
    # pklMaker.makeFeatureForCategory()

    # pklMaker.makeMentionDictForMissingLink()
    pklMaker.splitSubPagesForMissingLink(50000)
