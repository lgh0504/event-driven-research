from __future__ import print_function
from __future__ import print_function

import re
import time

import nltk
import numpy as np
from gensim.models import KeyedVectors
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize


class TextEmbedding:
    """
    Methods about text embedding, initializing this class is expensive
    """
    DIMENSION = 300
    zero_vector = [0] * DIMENSION
    one_vector = [1.0] * 3 * DIMENSION

    def __init__(self, model_path):
        """
        doing Initialization work
        :param model_path: the path to the google pre-trained language model
        """
        print("Initializing, may take several minutes...")

        # check necessary resources
        start_time = time.time()
        try:
            nltk.data.find('tokenizers/punkt.zip')
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find('corpora/stopwords.zip')
        except LookupError:
            nltk.download("stopwords")

        # load word2vec model
        self.pattern = '((http|ftp|https)://)(([a-zA-Z0-9\._-]+\.[a-zA-Z]{2,6})|([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,' \
                       '3}\.[0-9]{1,3}))(:[0-9]{1,4})*(/[a-zA-Z0-9\&%_\./-~-]*)? '
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.english_stopwords = stopwords.words("english")
        end_time = time.time()
        print("Initialization Done! Using %d seconds" % (end_time - start_time))

    @staticmethod
    def _vectors_mean(word_vectors):
        """
        get the mean of a bunch of vectors
        :param word_vectors: a bag / list of vectors
        :return:
        """
        text_vector = np.array(word_vectors)
        text_vector = np.mean(text_vector, axis=0)
        return text_vector.tolist()

    @staticmethod
    def _vectors_mix(text_vectors):
        """
        a naive way to merge text vectors
        :param text_vectors: a bag of text vectors
        :return: a single event vector
        """
        event_vector = []
        temp_array = np.array(text_vectors)
        event_vector += np.max(temp_array, axis=0).tolist()
        event_vector += np.min(temp_array, axis=0).tolist()
        event_vector += np.mean(temp_array, axis=0).tolist()
        return event_vector

    def _text_tokenize(self, text):
        """
        tokenize and clean the text
        :param text: a string
        :return: cleaned [words]
        """

        # remove urls
        text = re.sub(self.pattern, "", text)

        # tokenize, turn string to [[words]]
        words = [word_tokenize(t) for t in sent_tokenize(text)]

        # remove punctuation and stop words
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*', '&']
        clean_words = []
        for sent in words:
            for word in sent:
                word = word.lower()
                if word not in english_punctuations and word not in self.english_stopwords:
                    clean_words.append(word)

        # return cleaned words
        return clean_words

    def text_embedding(self, text):
        """
        turn string to a vector
        :param text: string
        :return: a text vector
        """
        word_bag = self._text_tokenize(text)
        word_vector_bag = []
        for word in word_bag:
            try:
                vector = self.model[word]
            except KeyError:
                vector = TextEmbedding.zero_vector
            word_vector_bag.append(vector)
        return self._vectors_mean(word_vector_bag)

    def event_embedding(self, texts, weights):
        """
        turn [string] to a vector
        :param texts: [string]
        :parm weights: [weight]
        :return: a single event vector
        """
        if len(texts) == 0:
            return self.one_vector
        text_vectors = []
        for (text, weight) in zip(texts, weights):
            weighted_text_vector = [ weight * x for x in self.text_embedding(text)]
            text_vectors.append(weighted_text_vector)
        event_vector = self._vectors_mix(text_vectors)
        return event_vector

