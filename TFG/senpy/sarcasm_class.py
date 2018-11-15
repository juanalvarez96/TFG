from senpy.plugins import SentimentPlugin, ShelfMixin, AnalysisPlugin
from senpy.models import Sentiment, Response, Entry, Results
import os
import pandas as pd
import pdb





class Plugin(SentimentPlugin, ShelfMixin):
    """Plugin to detect sarcasm in Tweets"""
    author = "Juan Alvarez"
    version = '1'



    def activate(self, *args, **kwargs):
        path = os.path.dirname(os.path.abspath(__file__))
        pdb.set_trace()
        print(self.dataset)
        print(os.path.join(path, self.dataset))
        #pd.read_csv(os.path.join(path, self.dataset))
        #self.train()