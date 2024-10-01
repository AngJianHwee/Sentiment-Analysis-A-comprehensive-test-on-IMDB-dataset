import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re


def nltk_init():
    nltk.download('stopwords')
    nltk.download('punkt_tab')
    nltk.download('punkt')
    nltk.download('wordnet')


def _remove_punctuation(text):
    return re.sub('[^\w\s]', '', text)


def _remove_numbers(text):
    return re.sub('[0-9]+', '', text)


def _lower_case(text):
    return text.lower()


def _remove_extra_spaces(text):
    return ' '.join(text.split())


def _remove_stopwords(text, stopwords):
    return ' '.join([word for word in text.split() if word not in stopwords])


def _stemming(text, stemmer):
    return ' '.join([stemmer.stem(word) for word in text.split()])


def _lemmatization(text, lemmatizer):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])


def _remove_html_tags(text):
    return re.sub('<.*?>', '', text)


def preprocess_texts(texts, stopwords, stemmer, lemmatizer, logger=None):
    def lambda_logger(logger, info): return logger.info(
        info) if logger else print(info)

    lambda_logger(logger, '[Preprocessing] Start removing html tags.')
    texts = [_remove_html_tags(t) for t in texts]
    lambda_logger(logger, '[Preprocessing] Remove html tags done.')

    lambda_logger(logger, '[Preprocessing] Start removing punctuation.')
    texts = [_remove_punctuation(t) for t in texts]
    lambda_logger(logger, '[Preprocessing] Remove punctuation done.')

    lambda_logger(logger, '[Preprocessing] Start removing numbers.')
    texts = [_remove_numbers(t) for t in texts]
    lambda_logger(logger, '[Preprocessing] Remove numbers done.')

    lambda_logger(logger, '[Preprocessing] Start lower casing.')
    texts = [_lower_case(t) for t in texts]
    lambda_logger(logger, '[Preprocessing] Lower casing done.')

    lambda_logger(logger, '[Preprocessing] Start removing extra spaces.')
    texts = [_remove_extra_spaces(t) for t in texts]
    lambda_logger(logger, '[Preprocessing] Remove extra spaces done.')

    lambda_logger(logger, '[Preprocessing] Start removing stopwords.')
    texts = [_remove_stopwords(t, stopwords) for t in texts]
    lambda_logger(logger, '[Preprocessing] Remove stopwords done.')

    lambda_logger(logger, '[Preprocessing] Start stemming.')
    texts = [_stemming(t, stemmer) for t in texts]
    lambda_logger(logger, '[Preprocessing] Stemming done.')

    lambda_logger(logger, '[Preprocessing] Start lemmatization.')
    texts = [_lemmatization(t, lemmatizer) for t in texts]
    lambda_logger(logger, '[Preprocessing] Lemmatization done.')

    return texts


def generate_stopwords(language='english'):
    return set(stopwords.words(language))


def generate_stemmer(language='english'):
    if language == 'english':
        return PorterStemmer()
    else:
        raise ValueError('Language not supported')


def generate_lemmatizer(language='english'):
    if language == 'english':
        return nltk.WordNetLemmatizer()
    else:
        raise ValueError('Language not supported')
