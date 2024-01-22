import numpy as np
from gensim.models import KeyedVectors

def load_gensim_model(file_path):

    # load a pre-trained model from the given file path using Gensim KeyedVectors
    return KeyedVectors.load(file_path, mmap='r')