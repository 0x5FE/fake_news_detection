import pickle
import numpy as np
from os import path
from bs4 import BeautifulSoup as bs
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Masking
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

def load_data(data_path='data/'):
    if not path.exists(data_path + 'final_data.pkl'):
        print('No saved data found; generating from scratch...')
        print('Loading data')
        with open(data_path + 'train_val_data.pkl', 'rb') as f:
            train_data, val_data = pickle.load(f)
        with open(data_path + 'test_data.pkl', 'rb') as f:
            test_data = pickle.load(f)
        print('Making Tokenizer')
        tokenizer = Tokenizer(num_words=12_000,
                              filters='!"#$%&()*+,-./…‘’“”—–:;<=>?@[\\]^_`{|}~\\t\\n©®™',
                              lower=True,
                              split=" ")

        indices_to_remove = [232, 301, 620, 1362, 1656, 1738]

        for idx in indices_to_remove:
            train_data.pop(idx)

        if not path.exists(data_path + 'text_data.pkl'):
            print('No saved text found; converting HTML to text')
            train_texts = [bs(page[1], 'html.parser').get_text() for page in train_data]
            valid_texts = [bs(page[1], 'html.parser').get_text() for page in val_data]
            test_texts = [bs(page[1], 'html.parser').get_text() for page in test_data]
            with open(data_path + 'text_data.pkl', 'wb') as f:
                pickle.dump((train_texts, valid_texts, test_texts), f)

        else:
            print('Using preconverted text')
            with open(data_path + 'text_data.pkl', 'rb') as f:
                train_texts, valid_texts, test_texts = pickle.load(f)

        print('Fitting Tokenizer')
        tokenizer.fit_on_texts(train_texts)

        total_words = len(tokenizer.word_index)
        print('Generating sequences and labels from data/text')
        X_train = tokenizer.texts_to_sequences(train_texts)
        X_valid = tokenizer.texts_to_sequences(valid_texts)
        X_test = tokenizer.texts_to_sequences(test_texts)
        
        y_train = [page[2] for page in train_data]
        y_valid = [page[2] for page in val_data]
        y_test = [page[2] for page in test_data]

        print('Pruning bad data')
        prune_data(X_train, train_texts)
        prune_data(X_valid, valid_texts)
        prune_data(X_test, test_texts)

        word_idx = tokenizer.word_index
        with open(data_path + 'final_data.pkl', 'wb') as f:
            pickle.dump((X_train, y_train, X_valid, y_valid, X_test, y_test, total_words, word_idx), f)
    else:
        print('Using saved data')
        with open(data_path + 'final_data.pkl', 'rb') as f:
            X_train, y_train, X_valid, y_valid, X_test, y_test, total_words, word_idx = pickle.load(f)
