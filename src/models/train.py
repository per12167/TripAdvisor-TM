import pickle
import logging

import numpy as np
import pandas as pd

from src.features.dictionary import create_dictionary
from src.features.tokenize import (tokenize, read_sample)
import gensim.corpora as corpora
import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def train():

    data = read_sample()
    logging.info('Source data file read succesfully')

    logging.info('Tokenization started. This will for sure take some time.')
    data = tokenize(data.review)
    logging.info('Tokenization completed for all documents.')

    id2word = corpora.Dictionary(data)

    corpus = [id2word.doc2bow(text) for text in data]

    logging.info('Model training started.')
    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,
                                                random_state=100,
                                                update_every=1,
                                                chunksize=100,
                                                passes=10,
                                                alpha='auto',
                                                per_word_topics=True)

    lda_model.save('models/lda_model')
    lda_model.print_topics()
    logging.info('Model saved to artifact lda_model :)')