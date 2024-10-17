import pickle
from sklearn.feature_extraction.text import TfidfVectorizer


def get_stopwords(documents, threshold=3):
    """
    :param documents: list of documents
    :param threshold:
    :return: list of words has idf <= threshold
    """
    tfidf = TfidfVectorizer(min_df=100)
    tfidf_matrix = tfidf.fit_transform(documents)
    features = tfidf.get_feature_names_out()
    stopwords = []
    print(min(tfidf.idf_), max(tfidf.idf_), len(features))
    for index, feature in enumerate(features):
        if tfidf.idf_[index] <= threshold:
            stopwords.append(feature)
    return stopwords


if __name__ == '__main__':
    docs = pickle.load(open('../dataset/processed-data/X.pkl', 'rb'))
    stopwords = get_stopwords(docs, threshold=3)
    with open('stopwords.txt', 'w', encoding='utf-8') as fp:
        for word in stopwords:
            fp.write(word + '\n')
