from typing import List

from src.features.tokenize import tokenize
from gensim.models import LdaModel
from gensim import corpora
import pandas as pd


def predict(documents: List[str]):

    tokens = tokenize(documents)

    lda_model = LdaModel.load('models/lda_model')

    topics = ['Cool Music Hostel', 'Romantic Stay', 'Great to Eat', 'Gorgeous for Business Trip',
              'Great Staff and Location', 'Top and Cozy', 'Old with Garden Patio', 'Dissapointed with Hotel',
              'Good Place to Stay', 'Like a Museum']

    id2word = corpora.Dictionary(tokens)
    corpus = [id2word.doc2bow(text) for text in tokens]

    lda_prediction = pd.DataFrame(lda_model[corpus][0][0])
    lda_prediction.columns = ['Index', 'Prob']
    index = int(lda_prediction['Index'][lda_prediction[['Prob']].idxmax()])
    topic = topics[index]

    return topic
