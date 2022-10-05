import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    takes the csvs and using pandas turns reads the data in and turns them into dataframes. Then the two dataframes are merged by the common id and returned as a single dataframe.

    Args:
        messages_filepath (csv): a csv file of twitter messages
        categories_filepath (csv): a csv file of message categories

    Returns:
        dataframes:the result of merging the two messages data from the input 
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    df = categories_df.merge(messages_df, left_on = 'id', right_on = 'id')

    return df


def clean_data(df):
    """
    a function that takes string value of "category" data and turns it into 36 columns of dummy variables. Then cleans the values to be numeric 1 or 0.
    Finally, the dataframe is deduplicated of rows and the dummy variable features are given names according to their feature. The return
    is a cleaned dataframe.

    Args:
        df (dataframe): merged intermediate dataframe containing messages and categories

    Returns:
        dataframe: a singular cleaned dataframe that seperates categories as dummy variables and removes all duplicate rows
    """
    categories = df['categories'].str.split(';',expand=True)
    categories_colnames = categories.loc[0].str[:-2].tolist()
    categories.columns = categories_colnames

    #turn values into 0 or 1 based on end of string value, then turn numeric
    for columns in categories:
        categories[columns] = categories[columns].str[-1:].astype('int32')
    df.drop(columns="categories", inplace=True)

    #combine and drop duplicate rows
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(keep='first', inplace=True)
    df.reset_index()

    return df

def save_data(df, database_filename):
    """
    saves the dataframe into a sqllite dataframe

    Args:
        df (dataframe): cleaned pandas dataframe of messages and categories
        database_filename (database): file name for the database (ex. data/distaster.db)
    """

    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql("disaster_table", engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()