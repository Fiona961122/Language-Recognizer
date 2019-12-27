"""recognizer.py
Author: fiona961122@hotmail.com
Data: 12/20/2019
Version: 1.0
Desc: build a classifier to recognize languages of spanish, french, english and italian"""

import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix as cm
import pandas as pd
import pickle
import numpy as np
import argparse

File_PATH = 'train_languages.csv'
TOKENIZER = 'tokenizer.pkl'
LABEL_ENCODER = 'label_encoder.pkl'
RECOGNIZER = 'recognizer.h5'
MAX_FEATURES = 5000
MAXLEN = 400
EMBEDDING_DIM = 50


def load_data():
    """load and preprocess the data"""
    data_df = pd.read_csv(File_PATH)  # in total 3633 data
    X, Y = data_df['sentence'], data_df['language']

    # encode the categories
    encoder = LabelEncoder()
    Y = encoder.fit_transform(Y)
    pickle.dump(encoder, open(LABEL_ENCODER, 'wb'))
    Y = tf.keras.utils.to_categorical(Y, num_classes=4)

    # preprocess the text
    data_df['sentence_lower_no_punc'] = data_df['sentence'].str.lower().str.replace('[^\w\s]', '').fillna('fillna')
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=MAX_FEATURES)
    tokenizer.fit_on_texts(list(data_df['sentence_lower_no_punc']))
    pickle.dump(tokenizer, open(TOKENIZER, 'wb'))
    X = tokenizer.texts_to_sequences(list(data_df['sentence_lower_no_punc']))
    X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=MAXLEN)

    return train_test_split(X, Y, test_size=0.1, random_state=42)


def train():
    """build up the recognizer"""
    X_train, X_test, Y_train, Y_test = load_data()
    tokenizer = pickle.load(open(TOKENIZER, 'rb'))

    # build up the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index)+1,
                                  output_dim=EMBEDDING_DIM,
                                  input_length=MAXLEN),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(np.array(X_train), np.array(Y_train), epochs=3)

    # evaluate the model
    loss, accuracy = model.evaluate(np.array(X_test), np.array(Y_test))
    print("Test Loss: " + str(loss))
    print("Test Accuracy: " + str(accuracy))
    predictions = model.predict(X_test)
    confusion = cm(predictions.argmax(axis=1), Y_test.argmax(axis=1))
    print(confusion)

    model.save(RECOGNIZER)

def test_wiki():

    import wikipedia

    tokenizer = pickle.load(open(TOKENIZER, 'rb'))
    model = tf.keras.models.load_model(RECOGNIZER)

    new_wiki_text = []
    wikipedia.set_lang('es')
    for i in range(0, 5):
        random = wikipedia.random(1)

        try:
            new_wiki_text.append([wikipedia.page(random).summary])
        except wikipedia.exceptions.DisambiguationError as e:
            random = wikipedia.random(1)

    new_wiki_text = pd.DataFrame(new_wiki_text)
    new_wiki_text.columns = ['sentence']

    new_wiki_text['sentence_lower'] = new_wiki_text["sentence"].str.lower()
    new_wiki_text['sentence_no_punctuation'] = new_wiki_text['sentence_lower'].str.replace('[^\w\s]', '')
    new_wiki_text['sentence_no_punctuation'] = new_wiki_text["sentence_no_punctuation"].fillna("fillna")

    np.set_printoptions(suppress=True)
    test_wiki_text = tokenizer.texts_to_sequences(
        list(new_wiki_text['sentence_no_punctuation']))  # this is how we create sequences
    test_wiki_text = tf.keras.preprocessing.sequence.pad_sequences(test_wiki_text,
                                                                   maxlen=MAXLEN)  # let's execute pad step

    predictions = model.predict(test_wiki_text)
    print(predictions)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.test:
        test_wiki()


if __name__ == '__main__':
    main()