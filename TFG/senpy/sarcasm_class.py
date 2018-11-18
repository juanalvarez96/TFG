import pandas as pd
import string
import os
from sklearn import svm, naive_bayes
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from nltk.tokenize import sent_tokenize, word_tokenize
from senpy.models import Sentiment, Response, Entry, Results
from senpy.plugins import SentimentPlugin, ShelfMixin, AnalysisPlugin
import nltk
import pdb

# Tokenizer customly defined. Will be used later


def custom_tokenizer(words):
    tokens = word_tokenize(words.lower())
    stemmer = SnowballStemmer('spanish')
    lemmas = [stemmer.stem(t) for t in tokens]
    stoplist = stopwords.words('spanish')
    lemmas_clean = [w for w in lemmas if w not in stoplist]
    punctuation = set(string.punctuation)
    lemmas_punct = [w for w in lemmas_clean if w not in punctuation]
    return lemmas_punct


class Plugin(AnalysisPlugin, ShelfMixin):
    """Plugin to detect sarcasm in Tweets"""
    author = "Juan Alvarez"
    version = '1'

    # Train the classifier
    def train(self):
        self.dataset = os.path.dirname(
            os.path.abspath(__file__)) + self.dataset
        print(self.dataset)

        #print(os.path.join(path, self.dataset))
        df = pd.read_csv(self.dataset)

        # Cleansing
        df = df.dropna(subset=['tweet'])

        # Before splitting database, a shuffling action will be performed since data is not randomized.
        # That way the train and test splitting will be more balanced
        df = df.sample(frac=1).reset_index(drop=True)

        # Define X and Y
        X = df['tweet'].values
        y = df['ironic'].values.astype(int)

        # Feature extraction pipeline
        ngrams_featurizer = Pipeline([
            ('count_vectorizer',  CountVectorizer(ngram_range=(1, 2), encoding='ISO-8859-1',
                                                  tokenizer=custom_tokenizer)),
            ('tfidf_transformer', TfidfTransformer())
        ])

        #Feature Union pipeline
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
                    ('lda',  LatentDirichletAllocation(n_components=45, max_iter=5,  # Change ntopics
                                                       learning_method='online',
                                                       learning_offset=50.,
                                                       random_state=0))
                ])),
            ])),

            ('clf', svm.LinearSVC())  # classifier
        ])
        pdb.set_trace()
        modelSVC.fit(X, y)

        return modelSVC

    def activate(self, *args, **kwargs):
        if 'classifier' not in self.sh:
            classifier = self.train()
            #pdb.set_trace()
            self.sh['classifier'] = classifier

        self._classifier = self.sh['classifier']
        self.save()

    def analyse_entry(self, entry, params):
        text = entry["nif:isString"]
        #pdb.set_trace()
        value = self._classifier.predict([text])
        ironic = value[0]

        if ironic == 1:
            outputText="Ironic"
        elif ironic == 0:
            outputText="Non Ironic"
        else:
            outputText="Error classifying text"
        entity = {'@id':'Entity0','text':text, 'is_ironic': outputText}


        yield entity

    def deactivate(self):
        self.close()
