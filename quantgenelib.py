
from __future__ import division

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder, Imputer, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.model_selection import train_test_split

import itertools

class Base():
    '''
    Defined to import the dataset.
    '''
    def __init__(self, file):
        self.file = file
        self.nameA = "Dataset A 16"
        self.nameB = "Dataset B 24"
        self.nA = 16
        self.nB = 24
        
        self.getData()
        
    def getData(self):
        self.dataA = pd.read_excel(self.file, sheet_name=self.nameA, header=None)
        print('Data A Loaded')
        self.dataB = pd.read_excel(self.file, sheet_name=self.nameB, header=None)
        print('Data B Loaded')

class Import(Base):
    '''
    This class is created to reuse the general preprocessing steps.
    
    *parameter
    
    file: enter the name of the excel file that stores the DNA mutation ratio and disease information of the subjects 
    '''
    def __init__(self, file):
        super().__init__(file)
    
    def organize(self):
        '''
        Transfrom the unorganized dataset to be model inputs.
        '''
        rowStart = 2
        colStart = 4
        DNAlistA = set(self.dataA.iloc[rowStart:, 2].unique())
        DNAlistB = set(self.dataB.iloc[rowStart:, 2].unique())
        self.DNAinter = list(DNAlistA.intersection(DNAlistB))
        
        def new_col(data):
            colNames = data.iloc[rowStart:, 2].apply(str) + "-" + \
                        data.iloc[rowStart:, 0].apply(lambda x: str(x).split('_')[1] if not pd.isnull(x) else '00') +\
                        "-" + data.iloc[rowStart:, 1].apply(str)
            return colNames
            
        def get_trans(data, n):
            colNames = new_col(data)
            data_T = data.iloc[rowStart:, colStart:n+colStart].T
            data_T.columns = colNames
            return data_T
        
        def get_target(data, n):
            target = data.iloc[0, colStart:n+colStart].apply(lambda x: x.split(' ')[1] if not pd.isnull(x) else 'N').reset_index(drop=True)
            return target
        
        self.targetA = get_target(self.dataA, self.nA) 
        self.targetB = get_target(self.dataB, self.nB)
        
        self.dataA = get_trans(self.dataA, self.nA).reset_index().drop('index', axis=1)
        self.dataB = get_trans(self.dataB, self.nB).reset_index().drop('index', axis=1)
        
        self.dataA['isC'] = (self.targetA == 'C').apply(int)
        self.dataB['isC'] = (self.targetB == 'C').apply(int)
        
        def encode(target):
            nClass = len(np.unique(self.targetB))
            codeDic = dict(zip(np.unique(self.targetB), range(nClass)))
            codeDic['C'] = codeDic['N']
            
            y = np.zeros(len(target)*nClass)
            for i, target in enumerate(target):
                y[nClass*i + codeDic[target]] = 1
            return y.reshape(-1, nClass)
        
        self.yA, self.yB = map(encode, [self.targetA, self.targetB])
        self.organized = 1
        
        # missing values
        imputer = Imputer(strategy='median')
        self.columnsA = self.dataA.columns
        self.columnsB = self.dataB.columns
        self.dataA = pd.DataFrame(imputer.fit_transform(self.dataA), columns=self.columnsA)
        self.dataB = pd.DataFrame(imputer.fit_transform(self.dataB), columns=self.columnsB)
        
        return self.dataA, self.dataB, self.DNAinter
      
      
class Modeling():
    '''
    This is a helper class to test classifiers based on f1 scores.
    '''
    def __init__(self):
        self.score = []
    
    def fit(self, X_train, y_train):
        self.X_train = X_train.copy()
        self.y_train = y_train.copy()
        self.best_f1 = -1

    def test(self, clf, X_test, y_test):
        clf.fit(X_test, y_test)
        pred = clf.predict(X_test)
        self.score.append(f1_score(y_test, pred, average='weighted'))
        
        f1 = f1_score(y_test, pred, average='weighted')
        if f1 == max(self.score) and f1 > self.best_f1:
            self.best_clf = clf
            self.best_f1 = f1

    def plot(self, name, param):
        sns.set_style("darkgrid")
        plt.figure(figsize=(7,7))
        plt.plot(param, self.score)
        plt.xlabel(name)
        plt.ylabel('f1-score')
        plt.title('f1-score vs. {0}: {1}'.format(name, param))
        self.score = []
        self.best_f1 = -1
        
        
class Preprocess():
    '''
    This class is to preprocess the imported dataset for classifiers.
    '''
    def __init__(self, aug_ratio=100000):
        assert isinstance(aug_ratio, int) and aug_ratio > 1, Exception('Enter a valid integer for the augment ratio')
        self.aug_ratio = aug_ratio
        self.n_row_A = 16
        
    def fit(self, dataA, dataB, y_train, y_test, DNAlist):
        self.dataA = dataA.copy()
        self.dataB = dataB.copy()
        self.DNAlist = DNAlist.copy()
        self.y_train = y_train.copy()
        self.y_test = y_test.copy()
        
    def get_total(self, kind='max'):
        assert kind in set(['mean', 'median', 'max']), Exception('Enter a valid option for the kind')
        self.kind = kind

        total = []
        for gene in [self.dataA, self.dataB]:
            new_data = []
            
            for n in self.DNAlist:
                geneTemp = [x for x in list(gene.columns) if x.split('-')[0] == str(n)]
            
                if self.kind == 'mean':
                    new_data.append(np.mean(gene[geneTemp], axis=1))
                
                elif self.kind == 'median':
                    new_data.append(np.median(gene[geneTemp], axis=1))
                
                elif self.kind == 'max':
                    new_data.append(np.max(gene[geneTemp], axis=1))
            
            if gene.shape[0] == self.n_row_A:
                new_data.append(self.dataA['isC'])
            else:
                new_data.append([0]*gene.shape[0])
            
            scaler = MinMaxScaler()
            this = scaler.fit_transform(np.stack(new_data, axis=1))
            total.append(this)
            
        return tuple(total)
    
    def get_intersection(self):
        total = []
        for gene in [self.dataA, self.dataB]:
            new_data = []
            for n in self.DNAlist:
                geneTemp = [x for x in list(gene.columns) if x.split('-')[0] == str(n)]
                new_data.append(gene[geneTemp])
            
            if gene.shape[0] == self.n_row_A:
                new_data.append(pd.DataFrame(self.dataA['isC']))
            else:
                new_data.append(pd.DataFrame([0]*gene.shape[0]))
            
            total.append(np.concatenate(new_data, axis=1))
            
        return tuple(total)
      
    def augument(self, new_DNAlist):
        total = []
        new_y = []
        target_label = self.y_train.argmax(axis=1)
        
        for label in [0, 1, 2]:
            new_data = []
            
            for n in new_DNAlist:
                geneTemp = [x for x in list(self.dataA.columns) if x.split('-')[0] == str(n)]
                
                this_rows = np.where(target_label==label)[0]
                    
                s = np.std(self.dataA.loc[this_rows, geneTemp].values)
                mu = np.mean(self.dataA.loc[this_rows, geneTemp].values)

                existing_row = np.max(self.dataA.loc[this_rows, geneTemp], axis=1).values.reshape(-1, 1)
                created_row = (np.random.randn(self.aug_ratio*sum(target_label==label)) * s + mu).reshape(-1, 1)
                
                new_data.append(np.concatenate([existing_row, created_row]))
            
            total.append(np.concatenate(new_data, axis=1))
            new_y.extend([label for _ in range((self.aug_ratio+1)*sum(target_label==label))])
            
        def encode(target):
            nClass = len(np.unique(target_label))
            
            y = np.zeros(len(target)*nClass)
            
            for i, target in enumerate(target):
                y[nClass*i + target] = 1
            
            return y.reshape(-1, nClass)
          
        return (np.concatenate(total), encode(new_y))
    
    
### The definitions of two functions below are not original entirely. 
### Some parts were forked from a Kaggle kernel.
### I have lost the link to the original work.
def plot_confusion_mat(cm, classes, cmap=plt.cm.Blues):
    title='Confusion matrix'

    plt.figure(figsize=(7,7))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
def full_report(model, x, y_true, classes, clf):
    
    one_hot = (clf == 'rf')
        
    # 2. Predict classes and stores in y_pred
    y_pred = model.predict(x).argmax(axis=1) if one_hot else model.predict(x)
    y_true = y_true.argmax(axis=1) if one_hot else y_true

    
    # 3. Print accuracy score
    print("Accuracy : "+ str(accuracy_score(y_true, y_pred)))
    
    # 4. Print classification report
    print("Classification Report")
    print(classification_report(y_true, y_pred, digits=4))    
    
    # 5. Plot confusion matrix
    
    cnf_matrix = confusion_matrix(y_true, y_pred)
    #print(cnf_matrix)
    plot_confusion_mat(cnf_matrix, classes=classes)