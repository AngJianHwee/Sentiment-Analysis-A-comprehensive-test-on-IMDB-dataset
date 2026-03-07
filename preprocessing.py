import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from multiprocessing import Pool, cpu_count
from functools import partial
import os


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


def preprocess_texts(texts, stopwords, stemmer, lemmatizer, logger=None, n_jobs=None):
    """
    Preprocess text data using parallel processing.
    
    Args:
        texts: List of text strings to preprocess
        stopwords: Set of stopwords to remove
        stemmer: PorterStemmer instance
        lemmatizer: WordNetLemmatizer instance
        logger: Optional logger instance
        n_jobs: Number of parallel jobs (-1 for all CPUs)
    
    Returns:
        List of preprocessed text strings
    """
    # Determine number of parallel jobs
    if n_jobs is None:
        n_jobs = cpu_count()
    
    logger.info(f'[Preprocessing] Start removing html tags (n_jobs={n_jobs}).') if logger else print(f'[Preprocessing] Start removing html tags (n_jobs={n_jobs}).')
    texts = _parallel_process(texts, _remove_html_tags, n_jobs, logger)
    logger.info('[Preprocessing] Remove html tags done.') if logger else print('[Preprocessing] Remove html tags done.')

    logger.info('[Preprocessing] Start removing punctuation.') if logger else print('[Preprocessing] Start removing punctuation.')
    texts = _parallel_process(texts, _remove_punctuation, n_jobs, logger)
    logger.info('[Preprocessing] Remove punctuation done.') if logger else print('[Preprocessing] Remove punctuation done.')

    logger.info('[Preprocessing] Start removing numbers.') if logger else print('[Preprocessing] Start removing numbers.')
    texts = _parallel_process(texts, _remove_numbers, n_jobs, logger)
    logger.info('[Preprocessing] Remove numbers done.') if logger else print('[Preprocessing] Remove numbers done.')

    logger.info('[Preprocessing] Start lower casing.') if logger else print('[Preprocessing] Start lower casing.')
    texts = _parallel_process(texts, _lower_case, n_jobs, logger)
    logger.info('[Preprocessing] Lower casing done.') if logger else print('[Preprocessing] Lower casing done.')

    logger.info('[Preprocessing] Start removing extra spaces.') if logger else print('[Preprocessing] Start removing extra spaces.')
    texts = _parallel_process(texts, _remove_extra_spaces, n_jobs, logger)
    logger.info('[Preprocessing] Remove extra spaces done.') if logger else print('[Preprocessing] Remove extra spaces done.')

    logger.info('[Preprocessing] Start removing stopwords.') if logger else print('[Preprocessing] Start removing stopwords.')
    texts = _parallel_process(texts, partial(_remove_stopwords, stopwords=stopwords), n_jobs, logger)
    logger.info('[Preprocessing] Remove stopwords done.') if logger else print('[Preprocessing] Remove stopwords done.')

    logger.info('[Preprocessing] Start stemming.') if logger else print('[Preprocessing] Start stemming.')
    texts = _parallel_process(texts, partial(_stemming, stemmer=stemmer), n_jobs, logger)
    logger.info('[Preprocessing] Stemming done.') if logger else print('[Preprocessing] Stemming done.')

    logger.info('[Preprocessing] Start lemmatization.') if logger else print('[Preprocessing] Start lemmatization.')
    texts = _parallel_process(texts, partial(_lemmatization, lemmatizer=lemmatizer), n_jobs, logger)
    logger.info('[Preprocessing] Lemmatization done.') if logger else print('[Preprocessing] Lemmatization done.')

    return texts


def _parallel_process(texts, func, n_jobs, logger=None):
    """
    Process texts in parallel using multiprocessing Pool.
    
    Args:
        texts: List of text strings
        func: Function to apply to each text
        n_jobs: Number of parallel jobs
        logger: Optional logger instance
    
    Returns:
        List of processed texts
    """
    if n_jobs == 1 or len(texts) < 100:
        # For small datasets, sequential processing is faster
        return [func(t) for t in texts]
    
    try:
        with Pool(processes=n_jobs) as pool:
            results = list(pool.imap(func, texts, chunksize=100))
        return results
    except Exception as e:
        # Fallback to sequential if multiprocessing fails
        logger.info(f'[Preprocessing] Multiprocessing failed: {e}. Using sequential processing.') if logger else print(f'[Preprocessing] Multiprocessing failed: {e}. Using sequential processing.')
        return [func(t) for t in texts]


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
