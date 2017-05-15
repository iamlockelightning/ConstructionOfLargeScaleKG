# -*- coding:utf-8 -*-

import numpy as np
import cPickle as pickle
import json
import sys
import time
import Queue
import pdb

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, NuSVC

class HierarchicalClassifier:
    def __init__(self, classifier_type):
        self.classifier_type = classifier_type
        self.taxonomyTree = pickle.load(file('taxonomy.pkl', 'rb'))['Tree']
        self.taxonomyTypes = pickle.load(file('taxonomy.pkl', 'rb'))['Types']
        self.nodeDataSet = {}
        self.treeClassifier = {}
        for k in self.taxonomyTypes:
            if self.classifier_type == 'LR': self.treeClassifier[k] = LogisticRegression(n_jobs=-1)
            elif self.classifier_type == 'RF': self.treeClassifier[k] = RandomForestClassifier(n_jobs=-1)
            elif self.classifier_type == 'SVM': self.treeClassifier[k] = LinearSVC(max_iter=3000)

    def loadData(self):
        print 'def loadData(self): ...'
        dataSet = pickle.load(file('dataSet.pkl', 'rb'))
        for t in self.taxonomyTypes:
            self.nodeDataSet[t] = {'Xtrain': [], 'Ytrain': []}#, 'Xtest': [], 'Ytest': []}
            trainPool = []
            for v in dataSet['Train'][t]['pos']:
                trainPool.append((v['features'], 1))
            for v in dataSet['Train'][t]['neg']:
                trainPool.append((v['features'], 0))
            np.random.shuffle(trainPool)
            for v in trainPool:
                self.nodeDataSet[t]['Xtrain'].append(v[0])
                self.nodeDataSet[t]['Ytrain'].append(v[1])
            # testPool = []
            # for v in dataSet['Test'][t]['pos']:
            #     testPool.append((v['features'], v['label']))
            # for v in dataSet['Test'][t]['neg']:
            #     testPool.append((v['features'], v['label']))
            # np.random.shuffle(testPool)
            # for v in testPool:
            #     self.nodeDataSet[t]['Xtest'].append(v[0])
            #     self.nodeDataSet[t]['Ytest'].append(v[1])

        self.dataSet = dataSet

    def train(self):
        print 'def train(self): ...'
        cnt = 0
        f, t = 0, 0
        for node in self.taxonomyTypes:
            cnt += 1
            if cnt%10 == 0:
                print '__', cnt,'(/', len(self.taxonomyTypes), ')'

            if np.sum(self.nodeDataSet[node]['Ytrain']) == 0:
                self.treeClassifier[node] = False
                f += 1
            elif np.sum(self.nodeDataSet[node]['Ytrain']) == len(self.nodeDataSet[node]['Ytrain']):
                self.treeClassifier[node] = True
                t += 1
            else:
                self.treeClassifier[node].fit(self.nodeDataSet[node]['Xtrain'], self.nodeDataSet[node]['Ytrain'])

        print 'f:', f, 't:', t
        # TODO: Hierarchical P, R, F1 Calculation
        sumTe = 0.0
        sumPe = 0.0
        sumTe_Pe = 0.0
        cnt = 0
        for title in self.dataSet['Test']:
            page = self.dataSet['Test'][title]
            cnt += 1
            if cnt%100 == 0:
                print '__', cnt

            pe = self.predict(page['features'])
            te = page['label']
            # print '[',
            # for s in pe:
            #     print s.decode('utf-8'),
            # print '], [',
            # for s in te:
            #     print s.decode('utf-8'),
            # print ']'
            te_pe = list(set(pe).intersection(set(te)))
            sumTe_Pe += len(te_pe)
            sumTe += len(te)
            sumPe += len(pe)
            # ypred = self.treeClassifier[node].predict(self.nodeDataSet[node]['Xtest'])
            # ACC = metrics.accuracy_score(self.nodeDataSet[node]['Ytest'], ypred)
            # P = metrics.precision_score(self.nodeDataSet[node]['Ytest'], ypred)
            # R = metrics.recall_score(self.nodeDataSet[node]['Ytest'], ypred)
            # F1 = metrics.f1_score(self.nodeDataSet[node]['Ytest'], ypred)
            # print metrics.classification_report(self.nodeDataSet[node]['Ytest'], ypred, target_names=['Non', 'Pre'])
        hP = sumTe_Pe / sumPe
        hR = sumTe_Pe / sumTe
        hF1 = 2.0* hP*hR / (hP + hR)
        print "Hierarchical Precision:", hP
        print "Hierarchical Recall:", hR
        print "Hierarchical F1:", hF1

    def predict(self, vec):
        Te = []
        Q = Queue.Queue()
        for son in self.taxonomyTree['Root']['son']:
            Q.put(son)
        # pdb.set_trace()
        historyList = []
        while Q.empty()==False:
            # print Q.qsize()
            t = Q.get()
            historyList.append(t)
            if self.treeClassifier[t] == False:
                continue
            elif self.treeClassifier[t] == True or self.treeClassifier[t].predict(np.array(vec).reshape((1, -1)))[0] == 1:
                Te.append(t)
                if t in self.taxonomyTree:
                    for st in self.taxonomyTree[t]['son']:
                        if (st in historyList)==False and (st in Q.queue)==False:
                            Q.put(st)

        return Te

    def save(self):
        pickle.dump({'Model': self.treeClassifier, 'Message': ''}, file('model_'+ time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())  +'.pkl', 'wb'), True)

    def load(self, model_name):
        self.treeClassifier = pickle.load(file(model_name, 'rb'))['Model']


if __name__ == "__main__":
    if len(sys.argv) != 2:
        exit(0)

    classifier_type = sys.argv[1]
    print 'argv', sys.argv[1]
    hierarchicalClassifier = HierarchicalClassifier(classifier_type)
    hierarchicalClassifier.loadData()
    hierarchicalClassifier.train()
    # hierarchicalClassifier.predict(page)
    print 'Done! :D'
