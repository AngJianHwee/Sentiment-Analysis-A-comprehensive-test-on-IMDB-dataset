from gensim.models import KeyedVectors
import os
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def simple_vocab_vectorizer(texts):
    # built a vocab
    vocab = set()
    for review in texts:
        for word in review.split():
            vocab.add(word)

    # create a dict to map word to index
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # create a dict to map index to word
    index_to_word = {i: word for i, word in enumerate(vocab)}

    # transform review to vector
    def review_to_vector(review):
        vector = [0] * len(vocab)
        for word in review.split():
            if word in vocab:
                vector[word_to_index[word]] += 1
        return vector

    return [review_to_vector(review) for review in texts]


def tfidf_vectorizer(texts):
    tokenized_texts = [word_tokenize(text) for text in texts]
    count_vect = CountVectorizer()
    X_counts = count_vect.fit_transform(texts)

    # tfidf
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    return [vector.toarray()[0] for vector in X_tfidf]


def word2vec_vectorizer(texts):
    tokenized_texts = [word_tokenize(text) for text in texts]

    # train word2vec
    model = Word2Vec(tokenized_texts, min_count=1)
    return [sum([model.wv[word] for word in x]) for x in tokenized_texts]


def init_glove(glove_folder_path=None):
    if glove_folder_path:
        glove_folder_path = glove_folder_path
    else:
        glove_folder_path = '.'

    glove_folder_path = os.path.abspath(glove_folder_path)
    # skip if exist already
    if not os.path.exists(os.path.join(glove_folder_path, 'glove.6B.100d.txt.word2vec')):
        # download glove to glove_folder_path
        os.system(
            f"wget https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip -P {glove_folder_path}"
        )
        os.system('unzip glove.6B.zip -d {}'.format(glove_folder_path))

        # remove downloaded zip
        os.system('rm glove.6B.zip')

        # convert glove to word2vec
        glove_input_file = 'glove.6B.100d.txt'
        word2vec_output_file = 'glove.6B.100d.txt.word2vec'
        glove2word2vec(
            os.path.join(glove_folder_path, glove_input_file),
            os.path.join(glove_folder_path, word2vec_output_file)
        )

    # load glove, convert text to vec,
    glove_model = KeyedVectors.load_word2vec_format(
        os.path.join(glove_folder_path, 'glove.6B.100d.txt.word2vec'),
        binary=False
    )
    return glove_model


def glove_vectorizer(texts):
    glove_model = init_glove()

    def text_to_vec(text):
        vec = []
        for word in text:
            try:
                vec.append(glove_model[word])
            except KeyError:
                vec.append(glove_model['unk'])
        return sum(vec) / len(vec)

    return [text_to_vec(text) for text in texts]
