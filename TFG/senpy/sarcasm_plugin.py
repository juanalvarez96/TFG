
# coding: utf-8

# In[41]:


# This is a necessary import for importing the classifier. The code belongs to the classifier.ipynb notebook.
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import pickle


# In[42]:


#UTIL
#https://senpy.readthedocs.io/en/latest/plugins.html#docker-image
class SentimentPlugin(AnalysisPlugin):
    '''Plugin to detetct sarcasm'''
    author = "Juan Álvarez Fernández del Vallado"
    version = 1
    
    # Load classifier
    def activate(self):
        def custom_tokenizer(words):
            tokens = word_tokenize(words.lower())
            stemmer = SnowballStemmer('spanish')
            lemmas = [stemmer.stem(t) for t in tokens]
            stoplist = stopwords.words('spanish')
            lemmas_clean = [w for w in lemmas if w not in stoplist]
            punctuation = set(string.punctuation)
            lemmas_punct = [w for w in lemmas_clean if  w not in punctuation]
            return lemmas_punct
        
        # Import classifier
        filename = 'optimized_classifier.pkl'
        svm_model_pkl = open(filename, 'rb')
        self.classifier = pickle.load(svm_model_pkl)
    
    def analyse_entry(entry, params):
        
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
        
        

