from __future__ import print_function
from __future__ import print_function

import nltk
import numpy as np
from gensim.models import KeyedVectors
from matplotlib import pyplot
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA


class TextEmbedding:
    """
    Methods about text embedding, initialize this class is expensive
    """
    DIMENSION = 300
    zero_vector = [0] * DIMENSION

    def __init__(self, model_path):
        """
        doing Initialization work
        :param model_path: the path to the google pre-trained language model
        """
        print("Initializing, may take several minutes...")
        # check resources
        try:
            nltk.data.find('tokenizers/punkt.zip')
        except LookupError:
            nltk.download("punkt")
        try:
            nltk.data.find('corpora/stopwords.zip')
        except LookupError:
            nltk.download("stopwords")

        # load word2vec model
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)
        self.english_stopwords = stopwords.words("english")
        print("Initialization Done!")

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
        :param text: a string format of a text / document containing many sentences
        :return: a bag of words
        """
        # tokenize
        words = [word_tokenize(t) for t in sent_tokenize(text)]

        # remove punctuation and stop words
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '!', '@', '#', '%', '$', '*']
        clean_words = []
        for sent in words:
            for word in sent:
                if word not in english_punctuations and word.lower() not in self.english_stopwords:
                    clean_words.append(word)
        return clean_words

    def text_embedding(self, text):
        """
        turn a word bag to a word vector bag
        :param text: a string of a text / document
        :return: a bag / list of word vector
        """
        word_bag = self._text_tokenize(text)
        word_vector_bag = []
        for word in word_bag:
            try:
                vector = self.model[word]
            except KeyError:
                vector = TextEmbedding.zero_vector
            word_vector_bag.append(vector)
        return word_vector_bag

    def texts_embedding(self, texts):
        """
        convert a bunch of text to a single event vector
        :param texts: a list of text(string)
        :return: a single event vector
        """
        text_vectors = []
        for text in texts:
            text_vector = self._vectors_mean(self.text_embedding(text))
            text_vectors.append(text_vector)
        event_vector = self._vectors_mix(text_vectors)
        return event_vector

    def visualize_words(self, words):
        """
        visualize words
        :param words: a list of word
        :return: nothing, just show a 2-D image
        """
        # generate vectors
        word_vectors = []
        zero_vector = [0] * TextEmbedding.DIMENSION
        for word in words:
            try:
                vector = self.model[word]
            except KeyError:
                vector = zero_vector
            word_vectors.append(vector)

        # fit a 2d PCA model to the vectors
        pca = PCA(n_components=2)
        result = pca.fit_transform(word_vectors)

        # create a scatter plot of the projection
        pyplot.scatter(result[:, 0], result[:, 1])
        for i, word in enumerate(words):
            pyplot.annotate(word, xy=(result[i, 0], result[i, 1]))
        pyplot.show()
