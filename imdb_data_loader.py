import pandas as pd
from datasets import load_dataset

def load_and_prepare_imdb_dataset():
    # load the IMDB dataset
    dataset = load_dataset('imdb')

    # extract train and test datasets
    train_dataset = dataset['train']
    test_dataset = dataset['test']

    # convert to pandas DataFrames
    trainData = pd.DataFrame(train_dataset)
    testData = pd.DataFrame(test_dataset)

    return trainData, testData