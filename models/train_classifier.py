import sys
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')
from sqlalchemy import create_engine
import pandas as pd
import sqlite3
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, validation_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import re
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report, multilabel_confusion_matrix
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    """
    function that will create the engine to read an sqlite table and convert it into a pandas dataframe. Then it will seperate the X and y values into arrays.

    Args:
        database_filepath (string): file path of sqlite database

    Returns:
        array: an independent variables array and dependent variable array
    """
    
    engine = create_engine(
    'sqlite:///'+ database_filepath)

    df = pd.read_sql_table('disaster_table', engine)

    X = df.iloc[:, 1].values
    y = df.iloc[:,5:40].values

    return X, y

def tokenize(text):
    """
    _summary_

    Args:
        text (_type_): _description_

    Returns:
        _type_: _description_
    """
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()

    #remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]"," ", text.lower())

    #tokenize text
    tokens = word_tokenize(text)


    #lemmatize and remove stopwords
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens


def build_model():

    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    return pipeline



def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()