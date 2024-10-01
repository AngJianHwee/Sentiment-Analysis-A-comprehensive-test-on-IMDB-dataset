from classifier import VECTORIZATION_APPROACHES, CLASSIFICATION_MODELS, run_model
import json
from tqdm import tqdm
import random
from vectorizer import simple_vocab_vectorizer, tfidf_vectorizer, word2vec_vectorizer, glove_vectorizer
from preprocessing import nltk_init, preprocess_texts, generate_stopwords, generate_stemmer, generate_lemmatizer
from utils import plot_figure, plot_training_results

import matplotlib.pyplot as plt
import pandas as pd
import logging

formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler(
    'nlp.log', mode='w', encoding='utf-8', delay=False)
file_handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
logger.info('Logging started')
logging.basicConfig(level=logging.INFO)

logger.info('[Data] Start loading data')
data = pd.read_csv(
    'https://raw.githubusercontent.com/Ankit152/IMDB-sentiment-analysis/master/IMDB-Dataset.csv')
logger.info('[Data] Data loaded')

data = data.reset_index(drop=True)  # drop index

logger.info('[Data] Start plotting data')
plot_figure(data)
logger.info('[Data] Data plotted')


nltk_init()

stemmer = generate_stemmer()
lemmatizer = generate_lemmatizer()
stop_words = generate_stopwords()

logger.info('[Preprocessing] Start preprocessing data')

data['review_cleaned'] = preprocess_texts(
    data['review'], stop_words, stemmer, lemmatizer, logger)

logger.info('[Preprocessing] Data preprocessing completed')

# randomly show 10 reviews
for i in range(10):
    row = data.sample(1).iloc[0]
    print('Review:', row['review'])
    print('Sentiment:', row['sentiment'])
    print('Cleaned Review:', row['review_cleaned'])
    print()


# Approach 1: Simple vectorizer
logger.info('[Vectorization] [Simple] Start vectorization')
data['review_vector__simple'] = simple_vocab_vectorizer(data['review_cleaned'])
logger.info('[Vectorization] [Simple] Vectorization completed')

# Approach 2: tf-idf and save to review_vector__tfidf
logger.info('[Vectorization] [tfidf] Start vectorization')
data['review_vector__tfidf'] = tfidf_vectorizer(data['review_cleaned'])
logger.info('[Vectorization] [tfidf] Vectorization completed')

# approach 3: word2vec
logger.info('[Vectorization] [word2vec] Start vectorization')
data['review_vector__word2vec'] = word2vec_vectorizer(data['review_cleaned'])
logger.info('[Vectorization] [word2vec] Vectorization completed')

# Approach 4: GloVe
logger.info('[Vectorization] [GloVe] Prepare GloVe start')
data['review_vector__glove'] = glove_vectorizer(data['review_cleaned'])
logger.info('[Vectorization] [GloVe] Vectorization completed')

# view first 5 rows
print(data.head())


# random split by 8:2, add train_test_flag
index = ['train'] * int(len(data) * 0.8) + ['test'] * \
    (len(data) - int(len(data) * 0.8))
random.shuffle(index)
data['train_test_flag'] = index

# start training
train_results = {}
for approach in tqdm(VECTORIZATION_APPROACHES, desc='Approach', position=0):
    train_results[approach] = {}

    # get train and test data from previous step
    X_train = data[data['train_test_flag'] == 'train'][approach].values
    X_test = data[data['train_test_flag'] == 'test'][approach].values
    y_train = data[data['train_test_flag'] == 'train']['sentiment'].values
    y_test = data[data['train_test_flag'] == 'test']['sentiment'].values

    # convert to int for y
    y_train = [1 if x == 'positive' else 0 for x in y_train]
    y_test = [1 if x == 'positive' else 0 for x in y_test]

    for cur_model in tqdm(CLASSIFICATION_MODELS, desc='Model', position=1):

        train_results[approach][cur_model[1]] = run_model(
            cur_model, approach, X_train, y_train, X_test, y_test, logger)

        acc, f1, precision, recall = train_results[approach][cur_model[1]].values()
        logger.info(f'[Predict] [approach: {approach}] [Model: {cur_model[1]}] Completed, acc: {acc:.4f}, f1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}')

logger.info('[Train] Completed')

# dump to json
with open('train_results.json', 'w') as f:
    json.dump(train_results, f)
logger.info('[Dump] Dumped train results to ./train_results.json')

logger.info('[Plot] Start plotting for training results')
plot_training_results(train_results)
logger.info('[Plot] Training results plotted')
