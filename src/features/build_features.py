import os
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.abspath("../.."))
# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)

from src.code_snippets.dataprep.stanford_corenlp.utils import serverConnection
from src.code_snippets.utils.abstract_classes import DataProcessor
from src.code_snippets.dataprep.embeddings_preprocessing.glove.twitter_preprocessing import preprocessing
from src.code_snippets.dataprep.embeddings_preprocessing.glove.reader import read_glove_file,get_word_index_dicts
from src.code_snippets.dataprep.cleaner import (count_characters_in_tokenized_sentence,
                                                count_punctuations_in_tokenized_sentece,
                                                count_ner_tags_in_tokenized_sentence)
from src.code_snippets.dataprep.embeddings_preprocessing.data_preparation import sentences_to_indices, pretrained_embedding_layer

import modin.pandas as pd
import multiprocessing as mp
from nltk.parse.corenlp import CoreNLPParser
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords 
import re
import gensim
import string
from typing import Tuple


class textProcessor(DataProcessor):
    
    def __init__(self,data_dir: str,nrows = None,output_dir = None, embedding_dir = None)->None:
        
        self.data_dir = data_dir
        self.embedding_dir = embedding_dir
        self.output_dir = output_dir
        self.nrows = nrows
        self.server = serverConnection()
        self.server.start_server()
        
    def read(self)->None:
        self.df = pd.read_csv(self.data_dir,nrows = self.nrows)
        if self.embedding_dir is not None:
            self.gensim_model = read_glove_file(self.embedding_dir)
            self.word_to_index,self.index_to_words = get_word_index_dicts(self.gensim_model)
    
    def clean(self,text_col: str,new_col_name: str)->None:
        st = CoreNLPParser()
        self.df[new_col_name] = self.df[text_col].apply(preprocessing,self.df[text_col])
        self.df[new_col_name] = self.df[new_col_name].apply(lambda text: list(st.tokenize(text)),self.df[text_col])
        
    def feature_engineer(self,text_col: str)-> None:
        temp = self.df[text_col].apply(lambda x: re.sub("\\n"," ",x))
        self.df['num_characters'] = temp.apply(len)
        tokenized_col = temp.apply(word_tokenize)
        self.df['num_words'] = tokenized_col.apply(len)
        self.df['clean'] = (self.df[['toxic',
                                     'severe_toxic',
                                     'obscene','threat',
                                     'insult',
                                     'identity_hate']].sum(axis=1)==0)*1
        

        #features relative to number of words
        self.df['uppercase_pct'] = tokenized_col.apply(lambda x: len([i for i in x if i.isupper()])*100/len(x) )
        self.df['lowercase_pct'] = tokenized_col.apply(lambda x: len([i for i in x if i.islower()])*100/len(x) )
        self.df['exclamation_pct'] = temp.apply(lambda x: x.count('!')+x.count('¡'))\
            *100/self.df['num_words']
        self.df['punctuation_pct'] = tokenized_col.apply(count_punctuations_in_tokenized_sentece)\
            *100/self.df['num_words']
        self.df['nouns_pct'] = tokenized_col.\
            apply(lambda x: count_ner_tags_in_tokenized_sentence(x,'NOUN'))\
            *100/self.df['num_words']
        self.df['adjetives_pct'] = tokenized_col.\
            apply(lambda x: count_ner_tags_in_tokenized_sentence(x,'ADJ'))\
            *100/self.df['num_words']
        self.df['pronouns_pct'] = tokenized_col.\
            apply(lambda x: count_ner_tags_in_tokenized_sentence(x,'PRON'))\
            *100/self.df['num_words']
        self.df['verbs_pct'] = tokenized_col.\
            apply(lambda x: count_ner_tags_in_tokenized_sentence(x,'VERB'))\
            *100/self.df['num_words']
        
    def pre_process(self,text_col: str,max_len: int, target_cols: list,aux_cols  = None, filter_long_texts = True):
        
        if filter_long_texts:
            self.df = self.df[self.df['num_words']<=max_len]
        X = self.df[text_col].values
        X_indices = sentences_to_indices(X,self.word_to_index,max_len)
        y = self.df[target_cols].values
        if aux_cols is not None:
            X_aux = self.df[aux_cols].values
            return X_indices, X_aux, y
        else:
            return X_indices, y
    
    def save(self):
        pass
        
        
    