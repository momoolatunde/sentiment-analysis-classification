from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

def train_nb_classifier(train_data, train_labels):
    # create a Multinomial Naive Bayes classifier
    nb_classifier = MultinomialNB()
    nb_classifier.fit(train_data, train_labels)
    joblib.dump(nb_classifier, 'nb_classifier.pkl')
    return nb_classifier

def train_svm_classifier(train_data, train_labels):
    # create a Support Vector Machine classifier with a linear kernel
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(train_data, train_labels)
    joblib.dump(svm_classifier, 'svm_classifier.pkl')
    return svm_classifier

def train_lr_classifier(train_data, train_labels):
    # create a Logistic Regression classifier
    lr_classifier = LogisticRegression()
    lr_classifier.fit(train_data, train_labels)
    joblib.dump(lr_classifier, 'lr_classifier.pkl')
    return lr_classifier
    
def evaluate_model(model, test_data, test_labels):
    # evaluates the model using the test data and labels, returning the F1 score
    test_predictions = model.predict(test_data)
    f1 = f1_score(test_labels, test_predictions, average='binary')
    return f1
