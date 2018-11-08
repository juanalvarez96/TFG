
import senpy
from senpy.plugins import AnalysisPlugin, ShelfMixin
from senpy.models import Response, Entry, Sentiment, Results
from sklearn.model_selection import train_test_split
import nltk
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
from nltk.tokenize import sent_tokenize, word_tokenize
import re
import os


# This is a necessary import for importing the classifier. The code belongs to the classifier.ipynb notebook.
#UTIL
#https://senpy.readthedocs.io/en/latest/plugins.html#docker-image
class MyPLugin(ShelfMixin ,AnalysisPlugin):
    '''Plugin to detetct sarcasm'''
    author = "Juan Álvarez Fernández del Vallado"
    version = 1

    def train(self):
        ''' Classifier will be defined and trained here'''
        # Loading the datasets


        # Import database
        df=pd.read_csv('final_dataset.csv', encoding='utf-8', delimiter=",", header=0)
        df.groupby('ironic').size()

        # Delete rows containing nan
        df=df.dropna(subset=['tweet'])
        print(df)
                #return classifier
            # Load classifier
    def activate(self):
        train()
        
    
    def analyse_entry(self, entry, params):
        
        text = entry["nif:isString"]
        
        print(text+"Traza")
        
        value = modelSVM.predict(text[0])
        
        print("PREDICTION {}".format(prediction))
        if (prediction == 1):
            is_ironic = True
        else:
            is_ironic = False
        entity = {'text':text,'is_ironic':is_ironic}


        #No poner en foaf:predator por que foaf predator no existe como etiqueda dentro de la ontologia foaf
        #entity = {'@id':'Entity0','text':text,'is_insomniac':is_insomniac}

        #,'foaf:accountName':params['input'],'prov:wasGeneratedBy':self.id}

        #entry.entities = []
        #entry.entities.append(entity)        
        #yield entry
        yield entity
    
    def deactivate(self):
        self.close()
        
        

