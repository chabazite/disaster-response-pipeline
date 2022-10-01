import sys
import pandas as pd

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
    pass


def save_data(df, database_filename):
    pass  


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