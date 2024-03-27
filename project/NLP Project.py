# !pip install nltk
# !pip install emoji
# !pip install autocorrec\t
# !pip install xgboost
# !pip install -U spacy
# nltk.download('all-corpora')
# nltk.download('punkt')  # for using word_tokenizer
# nltk.download('wordnet')  # for using Lemmatizer
# nltk.download('averaged_perceptron_tagger') # for language processing i.e tagging words with their parts of speech (POS)
# !python -m spacy download en_core_web_lg

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import time
import spacy
import nltk
import re   # regular expression
import string
import multiprocessing
from multiprocessing import Pool
from pandarallel import pandarallel
import scipy.sparse as sp
import joblib
import emoji
from autocorrect import Speller   # for correcting spelling
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize    # for tokenizing string into words
from nltk.stem import WordNetLemmatizer    # for lemmatizing words
from nltk.tag import pos_tag # for tagging words with their parts of speech (POS)
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest,chi2, mutual_info_classif
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# import xgboost
# from xgboost import XGBClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, f1_score

print('Reading Training Data')
df_train = pd.read_csv('data/project_training_data_with_class_labels.csv',dtype=str,delimiter=',',quoting=3)
print(f'train data: {df_train.shape}')
print(f'{df_train.info()}')
print(f'check for null value{df_train.isna().sum()}')
df_train.rename(columns={' Class Labels ' : 'Class Labels'},inplace=True) # changing col name
df_train= df_train.replace(to_replace = np.nan, value = '', regex = True)
# Inserting the column before Class Labels col.
df_train.insert(loc = 3,
        column = 'Text',
        value = df_train['Parent Comments'] + " " +df_train['Comments'])

# df_train.head()
# we are combining Parent Comment and Comment cols. into one cols.
df_train['Text'] = df_train['Parent Comments'] + " " +df_train['Comments']
# df_train.head()
print('Reading test Data')
df_test = pd.read_csv('data/project_test_data.csv',dtype=str,delimiter=',',quoting=3)
print(df_test.head())
print(df_test.shape)
#preprocessing of data
df_test= df_test.replace(to_replace = np.nan, value = '', regex = True)
df_test.rename(columns={'Parent Comments ': 'Parent Comments'},inplace = True)
# we are combining Parent Comment and Comment cols. into one cols.
df_test['Text'] = df_test['Parent Comments'] + " " +df_test['Comments']
# df_test.head()
# df_train['Class Labels'].value_counts().plot(kind = 'bar',rot = .3)

##########################################################
# Pre Processing
#########################################################
# english Stopwords
stopwords_list = stopwords.words('english')
# Emoticons
with open('EMOTICONS.pkl', 'rb') as f1:
    EMOTICONS = pickle.load(f1)
# Contractions
with open('CONTRACTIONS.pkl', 'rb') as f2:
    CONTRACTIONS = pickle.load(f2)


def preprocess_text(tweet):
    # import numpy as np
    # import pandas as pd
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # import pickle
    # import time
    # import spacy
    # import nltk
    # import re  # regular expression
    # import string
    # import multiprocessing
    # from multiprocessing import Pool
    # import scipy.sparse as sp
    # import joblib
    # import emoji
    # from autocorrect import Speller  # for correcting spelling
    # from nltk.corpus import stopwords
    # from nltk.tokenize import word_tokenize  # for tokenizing string into words
    # from nltk.stem import WordNetLemmatizer  # for lemmatizing words
    # from nltk.tag import pos_tag  # for tagging words with their parts of speech (POS)
    #
    # # english Stopwords
    # stopwords_list = stopwords.words('english')
    # with open('EMOTICONS.pkl', 'rb') as f1:
    #     EMOTICONS = pickle.load(f1)
    # with open('CONTRACTIONS.pkl', 'rb') as f2:
    #     CONTRACTIONS = pickle.load(f2)

    nan_tweet = 'NaN'
    # this code is to short unnecessary sentence, bec. some rows has unnecessary long repeated characters
    # like 'HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH'
    # we manually decide len = 10 any with len >10 is discarded
    # convert all text lowercase
    tweet = tweet.lower()
    tweet = tweet.split()
    tw = []
    for t in tweet:
        # removing digits only
        if t.isnumeric():
            continue
        else:
            pass
        if len(t) <= 20:
            if len(set(t)) <= 1:
                continue
            if sum(c.isdigit() for c in t) > sum(c.isalpha() for c in t):
                continue
            tw.append(t)
    tweet = ' '.join(tw)
    # remove any urls
    tweet = re.sub(r"www\S+|http\S+|", "", tweet, flags=re.MULTILINE)
    # remove square bracket including its content if
    tweet = re.sub(r'\[|\]', " ", tweet)
    # to remove new line character
    tweet = re.sub(r'\n', " ", tweet)
    # remove user @ references and '#' from tweet
    tweet = re.sub(r"\@\w+|\#", "", tweet)
    # replace emojis with its meaning
    tweet = (emoji.demojize(tweet, delimiters=(" ", ""))).replace('_', ' ')
    # expand contractions
    splitted_string = tweet.split()
    for index, text in enumerate(splitted_string):
        if text in CONTRACTIONS.keys():
            splitted_string[index] = CONTRACTIONS[text]
    tweet = ' '.join(splitted_string)
    # replace emoticons with its meaning
    splitted_tweet = tweet.split()
    for index, word in enumerate(splitted_tweet):
        if word in EMOTICONS.keys():
            splitted_tweet[index] = EMOTICONS[word]
    tweet = ' '.join(splitted_tweet)
    # remove tags
    tweet = re.sub(re.compile('<.*?>'), '', tweet)
    # remove punctuations
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))
    # remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stopwords_list]
    # spelling correction
    correct_words = []
    # initialize Speller object for english language
    spell_corrector = Speller(lang='en')
    for word in filtered_words:
        correct_word = spell_corrector(word)
        correct_words.append(correct_word)
    # lemmatizing
    wnl = WordNetLemmatizer()
    lemma_words = []
    for word, tag in pos_tag(correct_words):  # Part-of-speech constants for ADJ,VERB,ADV = 'a', 'r', 'v'
        if tag.startswith('JJ'):  # for adjectives
            lemma_word = wnl.lemmatize(word, pos='a')
            lemma_words.append(lemma_word)
        elif tag.startswith('VB'):  # for verbs
            lemma_word = wnl.lemmatize(word, pos='v')
            lemma_words.append(lemma_word)
        elif tag.startswith('RB'):  # for adverbs
            lemma_word = wnl.lemmatize(word, pos='r')
            lemma_words.append(lemma_word)
        else:
            lemma_word = word
            lemma_words.append(lemma_word)

    tweet = " ".join(lemma_words)
    if len(tweet) == 0:  # if after pre-processing sent. has no letter
        tweet = nan_tweet
    # f1.close(),f2.close()
    return tweet
pandarallel.initialize(progress_bar=True,nb_workers=64)
################################################
# pre processing on traning data
print('Preprocess on training data')
t = time.time()
xx = df_train['Text'].parallel_apply(preprocess_text)
df_train.insert(loc = 4,
        column = 'Pre Processed Text',
        value = xx)
# saving
df_train.to_csv('processed_train.csv',index=False) # on train data,
tt = time.time() - t
print(f'preprocess time {tt} sec')
########################
# on Test data
print('Preprocess on Test data')
t = time.time()
xtt = df_test['Text'].parallel_apply(preprocess_text) # on test data
df_test.insert(loc = 4,
        column = 'Pre Processed Text',
        value = xtt)
df_test.to_csv('processed_test.csv',index=False)
tt = time.time() - t
print(f'preprocess time {tt} sec')
del df_train, df_test, xx, xtt
####################################
# Embeddings
#######################################
# train data
df_prepro_train = pd.read_csv('processed_train.csv',na_filter=False)
# test data
df_prepro_test = pd.read_csv('processed_test.csv')
print(df_prepro_test.shape)
df_prepro_test.head()
# Train data
X_train = df_prepro_train['Pre Processed Text']
y_trn = df_prepro_train['Class Labels']

# Test data
X_test = df_prepro_test['Pre Processed Text']
# converting Labels coloumn into numerical form of train data
encoder = LabelEncoder()
y = encoder.fit_transform(y_trn)

####################################
# Bag of Words
#####################################

CV = CountVectorizer(ngram_range=(2,2),max_features=100000) # (1,3) means Unigrams, Bigrams and Trigrams
X_train_bow = sp.csr_matrix(CV.fit_transform(X_train))  # compressing the data to use less memory


def Classification(X, file_name: str, method: str, Gridsearch=False,
                   model: str = 'all', cv=5):#fs: str = None
    # try:
    #     y_train = y_train.values.ravel()
    #     y_test = y_test.values.ravel()
    # except:
    #     pass
    with open(file_name + '.txt', 'a') as f:
        file_size = os.stat(f'{file_name}.txt').st_size
        if file_size == 0:
            print(f"New File Created: {file_name}.txt")
            f.write("method, Model, Accuracy, f1_micro, f1_macro,other_info \n")
        else:
            print(f"Existing file found: Appending to File: {file_name}.txt")
        f.close()

    # assert fs in ['pca', 'chi2', 'mi', None], f'feature selection should be {{"pca","chi2","mi",None}}. got:{fs}'
    mdls = ['knn', 'lr', 'dtree', 'rf', 'svm', 'mnb', 'all']
    assert model in mdls, f"model should be {{'all','lr','svm','knn','dtree','rf','mnb'}} but got : {model}"
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=3, stratify=y)
    # n_comp = 'fsp__k'
    # if fs == 'mi':
    #     fsl = SelectKBest(score_func=mutual_info_classif)
    #     fslk = [10000]
    # elif fs == 'chi2':
    #     fsl = SelectKBest(chi2)
    #     fslk = [10000]
    # elif fs == 'pca':
    #     n_comp = 'fsp__n_components'
    #     if isinstance(X_train, sp.csr_matrix):
    #         print('using TruncatedSVD as input is sparse')
    #         fsl = TruncatedSVD()
    #         fslk = [5000, 10000]
    #
    #     else:
    #         fsl = PCA(svd_solver='full')
    #         fslk = [0.95]
    #
    # elif fs == None:
    #     fsl = None
    fs = 'chi2'
    classifiers = [
        KNeighborsClassifier(),
        SGDClassifier(n_jobs=-1, loss='log'),
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        # AdaBoostClassifier(),
        #         GradientBoostingClassifier(),
        SGDClassifier(n_jobs=-1, early_stopping=True, loss='hinge'),
        MultinomialNB()

    ]
    if Gridsearch == True:
        clf_parameters = [
            {
                "clf__n_neighbors": [500, 1000],  # np.arange(2,25 ,10),
                "clf__metric": ["l1", "cosine"],
                "clf__weights": ["uniform", "distance"],
            },
            {
                'clf__penalty': ['l1', 'l2'],
                'clf__alpha': [0.001, 0.01, 0.1]
            },
            {
                'clf__criterion': ["gini", "entropy"],
                # 'clf__max_features':['sqrt', 'log2'],
                'clf__max_depth': [50, 100],
                # 'clf__ccp_alpha':np.logspace(-3,-2,20),#np.logspace(-2.32,-2.3,20),
                "clf__max_leaf_nodes": [100],

                # "clf__splitter" : ["best", "random"],
                # "clf__min_samples_split":np.arange(2,50,10)
            },
            {
                'clf__n_estimators': [50, 80],
                # 'clf__max_features': ['sqrt', 'log2'],
                'clf__max_depth': [20, 50],  # np.arange(4,15,2).tolist(),
                'clf__criterion': ['gini', 'entropy'],
                #         'clf__ccp_alpha':np.logspace(-2,1,10)
            },
            #             {
            #                 'clf__base_estimator':[RandomForestClassifier(),DecisionTreeClassifier(criterion='entropy'),SVC(),LogisticRegression(**{'C': 0.017433288221999882, 'penalty': 'l2', 'solver': 'newton-cg'})],
            #                 'clf__algorithm' : ['SAMME', 'SAMME.R'],
            #                 'clf__n_estimators': [50,100]
            #             },
            #             {
            #               'clf__loss' :['deviance', 'exponential'],
            #                 'clf__criterion' : ['friedman_mse', 'squared_error'],
            #                 'clf__max_features' : [ 'sqrt', 'log2'],
            #                 'clf__learning_rate': np.logspace(-1,1,5),
            #                 'clf__n_estimators':np.arange(100,1000,200)

            #             },
            {
                'clf__loss': ['hinge'],
                'clf__alpha': [0.0001,0.001, 0.01, 0.1]
                # 'clf__early_stopping':[True],

            },
            {
                'clf__alpha': [0] + np.logspace(-2, 5, 5).tolist(),

            }
        ]
    else:
        clf_parameters = [{}] * (len(mdls) - 1)
    # data[name] = {'Model':[],'Accuracy' :[],'f1_micro' :[],'f1_macro' :[]}
    # dataint = {'Model':[],'Accuracy' :[],'f1_micro' :[],'f1_macro' :[]}

    i = 1

    def fit(i, file_name, X_train, X_valid, y_train, y_valid):
        # pipeline = Pipeline([
        #     ('vect', CountVectorizer()),
        #     ('feature_selector', SelectKBest(chi2)),
        #     ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
        #     ('clf', classifier), ])
        #
        # feature_parameters = {
        #     'vect__min_df': (2, 3),
        #     'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (2, 3)),  # Unigrams, Bigrams or Trigrams
        # }
        # if fs != None:
        #     clf_params[n_comp] = fslk
        pipe = Pipeline([
            ('vect', CountVectorizer()),
            ('feature_selector', SelectKBest(chi2)),
            ('tfidf', TfidfTransformer(use_idf=True, smooth_idf=True)),
            ('clf', classifier), ])
        feature_parameters = {
            'vect__min_df': (2, 3),
            'vect__ngram_range': ((1, 1), (1, 2), (1, 3), (2, 3)),
            'feature_selector__k': [1000,10000]# Unigrams, Bigrams or Trigrams
        }
            # pipe = Pipeline(steps=[('',),('fsp', fsl), ('clf', classifier)])
        # else:
        #     pipe = Pipeline(steps=[('clf', classifier)])
        parameters = {**feature_parameters, **clf_params}
        # grid = GridSearchCV(pipeline, parameters, scoring='f1_micro', cv=10)
        grid = GridSearchCV(pipe, parameters, scoring='f1_micro', cv=cv, n_jobs=-1,
                            verbose=1)  # early_stopping=False,use_gpu=True)
        try:
            print("_" * 32)
            print(f'{i}.', classifier)
            print("_" * 32)
            grid.fit(X_train, y_train)
            pred = grid.predict(X_test)
            best_clf = grid.best_params_
            print(best_clf)
            print(classification_report(y_test, pred))
            i1 = classifier.__class__.__name__
            i2 = sklearn.metrics.accuracy_score(y_test, pred)
            i3 = sklearn.metrics.f1_score(y_test, pred, average='micro')
            i4 = sklearn.metrics.f1_score(y_test, pred, average='macro')
            with open(f'{file_name}.txt', 'a') as f:
                f.writelines(
                    f'{method},{i1},{i2},{i3},{i4},{f"model:{model};Gridsearch:{Gridsearch};cv:{cv};fs:{fs};best_param:{grid.best_params_}"} \n')
                f.close()
            # dataint['Model'].append(i1)
            # dataint['Accuracy'].append(i2)
            # dataint['f1_micro'].append(i3)
            # dataint['f1_macro'].append(i4)
            print("-" * 80)
            print("-" * 80)
            # data[name] = dataint

        except Exception as e:
            print(e)
        return grid

    if model == 'all':
        for classifier, clf_params in zip(classifiers, clf_parameters):
            fit(i, file_name, X_train, X_test, y_train, y_test)
            i += 1
    else:
        _ = mdls.index(model)
        # print(_,len(clf_parameters))
        classifier, clf_params = classifiers[_], clf_parameters[_]
        # print(classifier.get_params())
        fit(i, file_name, X_train, X_test, y_train, y_test)


