import utils
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class MemoryTagger(BaseEstimator, TransformerMixin):
    def fit(self, X, y):
        '''
        Find tag which most assigned to each word
        :param X: list of words
        :param y: list of tags
        :return:
        '''

        

    def predict(self, X, y=None):
        pass
