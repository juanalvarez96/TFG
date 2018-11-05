
# coding: utf-8

# ## Classifier and preprocessing
# 
# In this notebook, the noironicos dataset will be treated, since ironicos's tweets are all ironic and we want a mixture of ironic and non ironic.

# In[3]:


# General import and load data
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import re

# Needed for running
nltk.download('punkt')
nltk.download('stopwords')

# Import database
df=pd.read_csv('final_dataset.csv', encoding='utf-8', delimiter=",", header=0)
df.groupby('ironic').size()

# Delete rows containing nan
df=df.dropna(subset=['tweet'])


# In[6]:


# Before splitting database, a shuffling action will be performed since data is not randomized.
# That way the train and test splitting will be more balanced

df = df.sample(frac=1).reset_index(drop=True)

# Define X and Y
X = df['tweet'].values
y = df['ironic'].values.astype(int)


# ### Train and test splitting

# In[3]:



# Splitting
# Test set will be the 25% taken randomly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)


# ## Lexical features
# The lexical features analysis will be performed by using the twitter tokenizer provided by nltk library.
# Important: This feature extractor is NOT used since tweets are considered to contain only one sentence
# 

# # Sample statistics using NLTK
# # A transformer will be implemented
# 
# from nltk.tokenize import sent_tokenize, word_tokenize
# 
# 
# class LexicalStats (BaseEstimator, TransformerMixin):
#     """Extract lexical features from each document"""
#     
#     def number_sentences(self, doc):
#         sentences = sent_tokenize(doc, language='spanish')
#         return len(sentences)
# 
#     def fit(self, x, y=None):
#         return self
# 
#     def transform(self, docs):
#        
#         return [{'length': len(doc),
#                  'num_sentences': self.number_sentences(doc)}
#                 
#                 for doc in docs]

# In[4]:


# A tokenizer will be defined
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer
from nltk import word_tokenize
from nltk.corpus import stopwords
import string

def custom_tokenizer(words):
    tokens = word_tokenize(words.lower())
    stemmer = SnowballStemmer('spanish')
    lemmas = [stemmer.stem(t) for t in tokens]
    stoplist = stopwords.words('spanish')
    lemmas_clean = [w for w in lemmas if w not in stoplist]
    punctuation = set(string.punctuation)
    lemmas_punct = [w for w in lemmas_clean if  w not in punctuation]
    return lemmas_punct


# ## Syntactic features
# 
# ALOMEJOR HAY QUE QUITARLO

# In[5]:


# We will use NLTK's tag set
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import pos_tag, word_tokenize
import collections

# We can extract particular chunks (trozos, pedazos) from the sentence
# if we use a RegExpParser. See Syntactic Processing
def PosStats(BaseEstimator, TransformerMixin):
    
    def stats(self, doc):
        tokens = custom_tokenizer(doc)
        
        tagged = pos_tag(tokens, tagset = 'universal' )
        counts = collections.Counter(tag for word, tag in tagged)
        total = sum(counts.values())
        #copy tags so that we return always the same number of features
        pos_features = {'NOUN': 0, 'ADJ': 0, 'VERB': 0, 'ADV': 0, 'CONJ': 0, 
                        'ADP': 0, 'PRON':0, 'NUM': 0}
        
        pos_dic = dict((tag, float(count)/total) for tag,count in counts.items())
        for k in pos_dic:
            if k in pos_features:
                pos_features[k] = pos_dic[k]
        return pos_features
    
    def transform(self, docs, y=None):
        return [self.stats(doc) for doc in docs]
    
    def fit(self, docs, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
        


# ## Feature extraction Pipeline
# The feature extraction will be carried out by using pipelines. The defined pipelines are selected in order to extract the desired features

# In[6]:


from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer


ngrams_featurizer = Pipeline([
  ('count_vectorizer',  CountVectorizer(ngram_range = (1, 2), encoding = 'ISO-8859-1', 
                                        tokenizer=custom_tokenizer)),
  ('tfidf_transformer', TfidfTransformer())
])


# ## Feature Union Pipeline
# Now we define which features we want to extract, how to combine them and later apple machine learning in the resulting feature set.

# In[7]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import classification_report
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.svm import SVC

def my_pipeline(clf):
    pipeline = Pipeline([
       ('features', FeatureUnion([
                    
                    ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),
                    ('ngrams', ngrams_featurizer),
                    #('pos_stats', Pipeline([
                                #('pos_stats', PosStats()),
                                #('vectors', DictVectorizer())
                            #])),
                    ('lda', Pipeline([ 
                                ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                                ('lda',  LatentDirichletAllocation(n_components=45, max_iter=5, # Change ntopics
                                                       learning_method='online', 
                                                       learning_offset=50.,
                                                       random_state=0))
                            ])),
                ])),
       
        ('clf', clf)  # classifier
    ])
    return pipeline
    


# ## Multinomial NaiveBayes

# In[8]:


from sklearn.naive_bayes import  MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
print("Size of training set: {}   size of test set: {}".format(X_train.shape[0], X_test.shape[0]))
model = MultinomialNB(alpha=.01)
modelNB = my_pipeline(model)
modelNB.fit(X_train, y_train)


# In[9]:


predicted1 = modelNB.predict(X_test)
expected = y_test


# In[10]:


from sklearn import metrics
# Accuracy
metrics.accuracy_score(expected, predicted1)


# In[11]:


print(classification_report(expected, predicted1, digits=5))


# ### SVC

# In[12]:


from sklearn.svm import SVC

types_of_kernels = ['linear', 'rbf', 'poly']

kernel = types_of_kernels[0]
gamma = 3.0

# Create kNN model
model = SVC(kernel=kernel, probability=True, gamma=gamma)
modelSVC = my_pipeline(model)
modelSVC.fit(X_train, y_train)


# In[13]:


predicted2 = modelSVC.predict(X_test)
expected = y_test
metrics.accuracy_score(expected, predicted2)
print(classification_report(expected, predicted2, digits=5))


# ### Kneighbors Classifier

# In[14]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=7, algorithm='ball_tree')
modelKnn = my_pipeline(model)
modelKnn.fit(X_train, y_train)


# In[15]:


predicted3 = modelKnn.predict(X_test)
expected = y_test
metrics.accuracy_score(expected, predicted3)
print(classification_report(expected, predicted3, digits=5))


# ### Logistic Regression classifier

# In[16]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(n_jobs = -1)
modelLR = my_pipeline(model)
modelLR.fit(X_train, y_train)


# In[17]:


predicted4 = modelLR.predict(X_test)
expected = y_test
metrics.accuracy_score(expected, predicted4)
print(classification_report(expected, predicted4, digits=5))


# ## Optimize models
# Tune parameters of previously defined models using Grid Search

# ### Multinomial NaiveBayes

# In[28]:


from sklearn.model_selection import GridSearchCV
# Used alpha = .01
parametersNB = {'clf__alpha': [.0001,.001,.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]}
scoresNB = ['precision', 'recall']
for score in scoresNB:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    gs_NB = GridSearchCV(modelNB,parametersNB, n_jobs=-1, scoring='%s_macro' % score)
    gs_NB.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs_NB.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gs_NB.cv_results_['mean_test_score']
    stds = gs_NB.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs_NB.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, gs_NB.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    


# gs_NB=gs_NB.fit(X_train,y_train)

# print("Best Score with MultinomialNB: %s" % gs_NB.best_score_)
# for param_name in sorted(parametersNB.keys()):
#     print("%s: %r" % (param_name, gs_NB.best_params_[param_name]))

# ### SVC

# #Optimize SVC
# 
# 
# parametersSVC = {'clf__C':range(1,15),'clf__gamma': np.logspace(-6, -1, 10), 'clf__kernel': ('linear','rbf'),
#                  'clf__probability':(True,False),}
# 
# gs_SVC = GridSearchCV(modelSVC, parametersSVC, n_jobs=-1)

# gs_SVC = gs_SVC.fit(X_train, y_train)

#  print("Best Score with SVC: %s" % gs_SVC.best_score_)
# for param_name in sorted(parametersSVC.keys()):
#     print("%s: %r" % (param_name, gs_SVC.best_params_[param_name]))

# In[18]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = {'C': [1, 10, 100, 1000]}

scoresSVC = ['precision', 'recall']

for score in scoresSVC:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    gs_SVC = GridSearchCV(modelSVC, tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    gs_SVC.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs_SVC.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gs_SVC.cv_results_['mean_test_score']
    stds = gs_SVC.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs_SVC.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, gs_SVC.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# ### KNeighbors Classifier

# parametersKN = {'clf__n_neighbors': range(1,15), 'clf__p':(1,2),'clf__algorithm':('ball_tree', 'kd_tree', 'brute')}
# 
# gs_KN = GridSearchCV(modelKnn, parametersKN, n_jobs=-1)

# gs_KN = gs_KN.fit(X_train, y_train)

#  print("Best Score with KN: %s" % gs_KN.best_score_)
# for param_name in sorted(parametersKN.keys()):
#     print("%s: %r" % (param_name, gs_KN.best_params_[param_name]))

# In[ ]:


from sklearn.model_selection import GridSearchCV
tuned_parameters = {'clf__n_neighbors': range(1,15), 'clf__p':[1,2]}

scoresKNN = ['precision', 'recall']

for score in scoresKNN:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    gs_KNN = GridSearchCV(modelKnn, tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    gs_KNN.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs_KNN.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gs_KNN.cv_results_['mean_test_score']
    stds = gs_KNN.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs_KNN.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, gs_KNN.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()


# ### LogisticRgression

# parametersLR = {'clf__penalty': ['l1','l2'], 'clf__tol': [0.0001,0.001,0.01,0.1], 'clf__C': range(1,15)}
# 
# gs_LR = GridSearchCV(modelLR, parametersLR, n_jobs=-1)

# gs_LR = gs_LR.fit(X_train, y_train)

#  print("Best Score with LogisticRegression: %s" % gs_LR.best_score_)
# for param_name in sorted(parametersLR.keys()):
#     print("%s: %r" % (param_name, gs_LR.best_params_[param_name]))

# In[ ]:


tuned_parameters = [{'clf__penalty': ['l1','l2'], 'clf__tol': [0.0001,0.001,0.01,0.1], 'clf__C': range(1,15)}]

scoresLR = ['precision', 'recall']

for score in scoresLR:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    gs_LR = GridSearchCV(modelLR, tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    gs_LR.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(gs_KNN.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = gs_LR.cv_results_['mean_test_score']
    stds = gs_LR.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, gs_KNN.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, gs_LR.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()

