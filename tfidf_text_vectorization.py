from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_representation(train_data, test_data):
  
    tfidf_vectorizer = TfidfVectorizer()

    # fit the vectorizer on the training data and transform training data
    trainTfidf = tfidf_vectorizer.fit_transform(train_data)

    # transform the test data
    testTfidf = tfidf_vectorizer.transform(test_data)

    return trainTfidf, testTfidf