import pandas as pd

def generate_dl_results_df(f1_cnn_word2vec, f1_cnn_glove, f1_bilstm_word2vec, f1_bilstm_glove, f1_cnn_lstm_word2vec, f1_cnn_lstm_glove):
    # generates a DataFrame for deep learning model results with the F1-scores
    results = {
        'Model': ['CNN', 'CNN', 'BILSTM', 'BILSTM', 'CNN-LSTM', 'CNN-LSTM'],
        'Embedding': ['Word2Vec', 'GloVe', 'Word2Vec', 'GloVe', 'Word2Vec', 'GloVe'],
        'F1-score': [f1_cnn_word2vec, f1_cnn_glove, f1_bilstm_word2vec, f1_bilstm_glove, f1_cnn_lstm_word2vec, f1_cnn_lstm_glove]
    }

    # create DataFrame
    results_df = pd.DataFrame(results)
    return results_df
   
def generate_ml_results_df(f1_nb_bow, f1_nb_tfidf, f1_svm_bow, f1_svm_tfidf, f1_lr_bow, f1_lr_tfidf):
    # generates a DataFrame for machine learning model results with the F1-scores
    results = {
        "Model": ["Naive Bayes (BOW)", "Naive Bayes (TFIDF)", "SVM (BOW)", "SVM (TFIDF)", "LR (BOW)", "LR (TFIDF)"],
        "F1 Score": [f1_nb_bow, f1_nb_tfidf, f1_svm_bow, f1_svm_tfidf, f1_lr_bow, f1_lr_tfidf]
    }
    
     # create DataFrame
    results_df = pd.DataFrame(results)
    return results_df
