from matplotlib import pyplot as plt


def plot_figure(data):
    data['sentiment'].value_counts().plot(
        kind='bar', title='Distribution of Sentiment', figsize=(5, 3))
    plt.xticks(rotation=0)
    plt.savefig('sentiment_distribution.png')

    data['review'].apply(len).plot(kind='hist', title='Review Length Distribution',
                                   bins=50, figsize=(5, 3))
    plt.savefig('review_length_distribution.png')
