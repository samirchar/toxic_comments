import os
from re import X
import sys
from dotenv import load_dotenv, find_dotenv

sys.path.append(os.path.abspath("../.."))
# find .env automagically by walking up directories until it's found
dotenv_path = find_dotenv()
# load up the entries as environment variables
load_dotenv(dotenv_path)
# import pandas as pd

from src.code_snippets.dataprep.embeddings_preprocessing.data_preparation import (
    sentences_to_indices,
    pretrained_embedding_layer,
)

from src.code_snippets.utils.abstract_classes import Trainer
from src.code_snippets.dataprep.embeddings_preprocessing.glove.reader import read_glove_file,get_word_index_dicts
from src.code_snippets.dataprep.embeddings_preprocessing.data_preparation import pretrained_embedding_layer

class ManyToOneSeqModel(Trainer):

    def __init__(self,X,y,embedding_dir,X_aux = None):
        self.X = X
        self.X_aux = X_aux
        self.y = y
        self.embedding_dir = embedding_dir
        self.gensim_model = read_glove_file(self.embedding_dir)
        self.word_to_index,self.index_to_words = get_word_index_dicts(self.gensim_model)

    
    def set_model(self):
        pretrained_embedding_layer(self.gensim_model, self.word_to_index)
        