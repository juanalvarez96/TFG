
import senpy
from senpy.plugins import AnalysisPlugin, ShelfMixin
from senpy.models import Response, Entry, Sentiment, Results
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
from pandas import Series, DataFrame
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.svm import SVC, LinearSVC
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import FunctionTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.svm import SVC
import string
import re
import os
import pickle
import pdb


def custom_tokenizer(words):
    tokens = word_tokenize(words.lower())
    stemmer = SnowballStemmer('spanish')
    lemmas = [stemmer.stem(t) for t in tokens]
    stoplist = stopwords.words('spanish')
    lemmas_clean = [w for w in lemmas if w not in stoplist]
    punctuation = set(string.punctuation)
    lemmas_punct = [w for w in lemmas_clean if w not in punctuation]
    return lemmas_punct

# This is a necessary import for importing the classifier. The code belongs to the classifier.ipynb notebook.
#UTIL
#https://senpy.readthedocs.io/en/latest/plugins.html#docker-image
class MyPLugin(ShelfMixin, AnalysisPlugin):
    '''Plugin to detetct sarcasm'''
    author = "Juan Álvarez Fernández del Vallado"
    version = 1
    

    def train(self):
        ''' Classifier will be defined and trained here'''
        import nltk
        # Prueba a quitar estos dos (solo hacen falta una vez)
        nltk.download('punkt')
        nltk.download('stopwords')



        # Loading the datasets
        path = os.path.dirname(os.path.abspath(__file__))
        self.dataset = 'final_dataset.csv'
        df = pd.read_csv('final_dataset.csv', encoding='utf-8', delimiter=",", header=0)
        # Import database
        
        df.groupby('ironic').size()

        # Delete rows containing nan
        df = df.dropna(subset=['tweet'])
        # Hasta aquí llego

        # Before splitting database, a shuffling action will be performed since data is not randomized.
        # That way the train and test splitting will be more balanced

        df = df.sample(frac=1).reset_index(drop=True)

        # Define X and Y
        X = df['tweet'].values
        y = df['ironic'].values.astype(int)

        # Since optimum clasifier's parameters are already known, no need to split dataset.

        # Define model

        

        ngrams_featurizer = Pipeline([
            ('count_vectorizer',  CountVectorizer(ngram_range=(1, 2), encoding='ISO-8859-1',
                                                  tokenizer=custom_tokenizer)),
            ('tfidf_transformer', TfidfTransformer())
        ])

        modelSVC = Pipeline([
            ('features', FeatureUnion([

                ('words', TfidfVectorizer(tokenizer=custom_tokenizer)),
                ('ngrams', ngrams_featurizer),
                #('pos_stats', Pipeline([
                #('pos_stats', PosStats()),
                #('vectors', DictVectorizer())
                #])),
                ('lda', Pipeline([
                    ('count', CountVectorizer(tokenizer=custom_tokenizer)),
                    ('lda',  LatentDirichletAllocation(n_components=10, max_iter=5,  # Change ntopics (optimized = 45)
                                                       learning_method='online',
                                                       learning_offset=50.,
                                                       random_state=0))
                ])),
            ])),

            # classifier with optimum parameters
            ('clf', LinearSVC(C=1, ))
            # Usar clasificador linearsvc
        ])

        # Fit Model
        modelSVC.fit(X,y)

        return modelSVC

    def activate(self):
        
        #pdb.set_trace()
        if 'classifier' not in self.sh:
            classifier = self.train()
            self.sh['classifier'] = classifier
        self.classifier = self.sh['classifier']
        self.save()

    def analyse_entry(self, entry, params):
        pdb.set_trace()
        text = entry["nif:isString"]
        value = self.classifier.predict([text])
        pdb.set_trace()
        prediction = value[0]
        pdb.set_trace()
        if (prediction == 1):
            is_ironic = True
        else:
            is_ironic = False
        entity = {'text': text, 'is_ironic': is_ironic}

        #No poner en foaf:predator por que foaf predator no existe como etiqueda dentro de la ontologia foaf
        #entity = {'@id':'Entity0','text':text,'is_insomniac':is_insomniac}

        #,'foaf:accountName':params['input'],'prov:wasGeneratedBy':self.id}

        #entry.entities = []
        #entry.entities.append(entity)
        #yield entry
        yield entity

    def deactivate(self):
        self.close()
