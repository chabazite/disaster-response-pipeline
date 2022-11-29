import sys
import nltk
nltk.download('stopwords')
nltk.download(['punkt', 'wordnet'])
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
from sqlalchemy import create_engine
import pandas as pd
import re

from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import  train_test_split

from sklearn.base import BaseEstimator, TransformerMixin
import pickle

import xgboost as xgb

class StartingPronounExtractor(BaseEstimator, TransformerMixin):
    """
    _summary_

    Args:
        BaseEstimator (_type_): _description_
        TransformerMixin (_type_): _description_
    """
    def starting_pronoun(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            # return true if the first word is an appropriate verb or RT for retweet
            if first_tag in ['PRP', 'PRP$']:
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_pronoun)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
    """
    function that will create the engine to read an sqlite table and convert it into a pandas dataframe. Then it will seperate the X and y values into arrays.

    Args:
        database_filepath (string): file path of sqlite database

    Returns:
        array: an independent variables array and dependent variable array. Also returns the training and testing split of these variables
    """
    
    engine = create_engine(
    'sqlite:///'+ database_filepath)

    df = pd.read_sql_table('disaster_table', engine)

    #only zero values, so we will remove this
    df = df.drop(['child_alone'],axis=1)

    X = df.iloc[:, 1].values
    y = df.iloc[:,4:].values

    category_names = list(df.iloc[:,4:].columns)

    return X, y , category_names

def tokenize(text):
    """
    This function will tokenize our text into words. First, it will remove all urls and insert placeholders. Then it will take the string and create individual word tokens. Finally, it will be placed through a lemmatizer, which will turn the words into their root words, lower case, and strip them of any leading or trailing spaces.

    Args:
        text (string): this is the incoming social media feed that will be tokenized

    Returns:
        list: a list of all the clean tokens for the incoming social media string
    """
    lemmatizer = WordNetLemmatizer()

    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    """
    _summary_

    Returns:
        _type_: _description_
    """
    
    pipeline = Pipeline([
    ('features', FeatureUnion([

        ('nlp_pipeline', Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer())
        ])),

        ('prnoun', StartingPronounExtractor())
    ])),

    ('xg', MultiOutputClassifier(xgb.XGBClassifier(learning_rate=0.1, subsample=0.5, max_depth=4, n_estimators=100, eval_metric='mlogloss',use_label_encoder=False)))])

    return pipeline



def evaluate_model(pipeline, X_test, Y_test, category_names):
    """
    _summary_

    Args:
        pipeline (_type_): _description_
        X_test (_type_): _description_
        Y_test (_type_): _description_
        category_names (_type_): _description_
    """

    y_pred = pipeline.predict(X_test)

    for index, label in enumerate(category_names):
        classification = classification_report(Y_test[:,index-1], y_pred[:,index-1]);
        print('----------------------------\n')
        print(label,"\n",classification)
    
    print( classification_report(Y_test,y_pred, target_names = category_names))

    return
   



def save_model(model, model_filepath):
    """
    _summary_

    Args:
        model (_type_): _description_
        model_filepath (_type_): _description_
    """

    pickle.dump(model, open(model_filepath, 'wb'))

    return


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