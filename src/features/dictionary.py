from typing import List

from gensim.corpora import Dictionary


def create_dictionary(documents: List[List[str]]):
    return Dictionary(documents)


def term_document_matrix(documents, dictionary: Dictionary):
    return [dictionary.doc2bow(text) for text in documents]