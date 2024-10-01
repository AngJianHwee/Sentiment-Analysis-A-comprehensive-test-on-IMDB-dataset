from matplotlib import pyplot as plt

APPROACH_NAME_DICT = {
        'review_vector__simple': 'Simple Vectorizer',
        'review_vector__tfidf': 'TF-IDF',
        'review_vector__word2vec': 'Word2Vec',
        'review_vector__glove': 'GloVe',
    }

def plot_figure(data):
    plt.figure()
    data['sentiment'].value_counts().plot(
        kind='bar', title='Distribution of Sentiment', figsize=(5, 3))
    plt.xticks(rotation=0)
    plt.savefig('sentiment_distribution.png')

    plt.figure()
    data['review'].apply(len).plot(kind='hist', title='Review Length Distribution',
                                   bins=50, figsize=(5, 3))
    plt.savefig('review_length_distribution.png')


def plot_training_results(train_results):
    plt.figure()
    approaches = list(train_results.keys())

    models = list(train_results[approaches[0]].keys())

    

    # get accuracy for each model, then plot with .plot, legend is different approach
    # plot together on same plot
    for approach in approaches:
        accs = [train_results[approach][model]['accuracy'] for model in models]
        plt.plot(models, accs, label=APPROACH_NAME_DICT[approach], marker='x')

    plt.gcf().set_size_inches(10, 5)
    plt.legend()
    plt.xticks(rotation=45)
    plt.ylim(0.5, 1)
    plt.grid()
    plt.title('Accuracy of Different Models')
    plt.savefig('accuracy_of_different_models.png')
