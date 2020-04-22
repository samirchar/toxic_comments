import os
import sys
from dotenv import load_dotenv, find_dotenv
import numpy as np
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
from src.code_snippets.utils.data_handler import read_pickle,save_to_pickle

#TODO: Change this import to modin
import pandas as pd
import multiprocessing as mp
from nltk.parse.corenlp import CoreNLPParser
from nltk.corpus import stopwords 
import re
import gensim
import string
from typing import Tuple
from sklearn.preprocessing import StandardScaler


class textProcessor(DataProcessor):
    
    def __init__(self,data_dir: str,nrows = None,output_dir = None, embedding_dir = None)->None:
        
        self.data_dir = data_dir
        self.embedding_dir = embedding_dir
        self.output_dir = output_dir
        self.nrows = nrows
        self.server = serverConnection()
        self.server.start_server()
        self.st = CoreNLPParser()
        
    def read(self)->None:
        self.df = pd.read_csv(self.data_dir,nrows = self.nrows)
        if self.embedding_dir is not None:
            self.gensim_model = read_glove_file(self.embedding_dir)
            self.word_to_index,self.index_to_words = get_word_index_dicts(self.gensim_model)

    def filter_by_text_length(self,processed_text_col: str,max_len: int):
        temp = self.df[processed_text_col].apply(len)
        self.df = self.df[temp<=max_len]

    def clean(self,text_col: str,new_col_name: str)->None:
        self.df[new_col_name] = self.df[text_col].apply(preprocessing)
        self.df[new_col_name] = self.df[new_col_name].apply(lambda text: list(self.st.tokenize(text)))

    def feature_engineer(self,text_col: str)-> None:
        temp = self.df[text_col].apply(lambda x: re.sub("\\n"," ",x))
        self.df['num_characters'] = temp.apply(len)
        tokenized_col = temp.apply(lambda text: list(self.st.tokenize(text)))
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
        
    def pre_process(self,text_col: str,max_len: int, target_cols: list,aux_cols  = None, standard_scaler = None, filter_by_max_length = False):
        
        if filter_by_max_length:
            print("FILTERING TEXTS BY MAX LENGTH")
            self.filter_by_text_length(text_col,max_len)

        X = self.df[text_col].values
        X_indices = sentences_to_indices(X,self.word_to_index,max_len)
        y = self.df[target_cols].values
        self.processed_data = { 'X_indices':X_indices, 'y':y}
        if aux_cols is not None:
            X_aux = self.df[aux_cols].values

            #Standarize aux features. If given a standard scaler in params, this is used to transformed data.
            #For this to work the standard scaler given as parameter must be previously trained. If no standard scaler is given,
            #It is assumed that this is training data and we need to fit for the first time.
            if standard_scaler is not None:
                self.ss = standard_scaler
                X_aux = self.ss.transform(X_aux)c
            else:
                self.ss = StandardScaler()
                self.ss.fit(X_aux)
                X_aux = self.ss.transform(X_aux)

            #Store X_aux to precessed data dictionary
            self.processed_data['X_aux'] = X_aux
        
    
    def save(self,suffix,directory = "../../data/processed/"):
        save_to_pickle(self.processed_data,os.path.join(directory,f"processed_data_{suffix}.pickle"))
        save_to_pickle(self.ss,os.path.join(directory,f"StandardScaler_{suffix}.pickle"))
            
        
    