from sklearn.feature_extraction.text import CountVectorizer

def create_bow_representation(train_data, test_data):

    vectorizer = CountVectorizer()

    # fit the vectorizer on the training data and transform training data
    trainBow = vectorizer.fit_transform(train_data)

    # transform the test data
    testBow = vectorizer.transform(test_data)

    return trainBow, testBow
