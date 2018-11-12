
import senpy
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from senpy.plugins import AnalysisPlugin, ShelfMixin, SentimentPlugin
from senpy.models import Response, Entry, Sentiment, Results
from nltk.parse.malt import train_from_file
import os
import pickle
import string

# This is a necessary import for importing the classifier. The code belongs to the classifier.ipynb notebook.
def custom_tokenizer(words):
    tokens = word_tokenize(words.lower())
    stemmer = SnowballStemmer('spanish')
    lemmas = [stemmer.stem(t) for t in tokens]
    stoplist = stopwords.words('spanish')
    lemmas_clean = [w for w in lemmas if w not in stoplist]
    punctuation = set(string.punctuation)
    lemmas_punct = [w for w in lemmas_clean if  w not in punctuation]
    return lemmas_punct
#UTIL
#https://senpy.readthedocs.io/en/latest/plugins.html#docker-image
class SarcasmPlugin(AnalysisPlugin, ShelfMixin):
    '''Plugin to detetct sarcasm'''
    author = "Juan Álvarez Fernández del Vallado"
    version = 1
    file_in_data='optimized_classifier.pkl'
    file_in_sources = 'senpy_data'
    # Load classifier
    def activate(self):
        with self.open(self.file_in_data) as f:
            self.classifier = train_from_file(f)
        file_in_source = os.path.join(self.get_folder(), self.file_in_sources)
        with self.open(file_in_source) as f:
            pass
    
    def analyse_entry(self, entry, params):
        
        text = entry["nif:isString"]
        
        print(text+"Traza")
        modelSVM = self.classifier 
        prediction = modelSVM.predict(text[0])
        
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
        
        

