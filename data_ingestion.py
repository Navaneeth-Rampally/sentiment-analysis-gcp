import pandas as pd
from datasets import load_dataset

def load_imdb_data(sample_size=1000):
    """ Loading IMDB Movie dataset from huggingface performing and learning sentiment analysis."""
    
    # Load dataset 
    dataset = load_dataset('imdb')

    # Converting into pandas dataframes train and test respectively
    train_df = pd.DataFrame(dataset['train'])
    test_df = pd.DataFrame(dataset['test'])

    # Creating a new column named sentiment and mapping them into negative and positive.
    train_df['sentiment'] = train_df['label'].map({0: 'Negative', 1: 'positive'})
    test_df['sentiment'] = test_df['label'].map({0: 'Negative', 1: 'positive'})

    # Renaming the columns 
    train_df = train_df.rename(columns={'text': 'review'})
    test_df = test_df.rename(columns={'text': 'review'})

    # Fetching sample data based on size requested by user.
    if sample_size:
        train_df = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
        test_df = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)

    print(f" Loaded {len(train_df)} training sample.")
    print(f" Loaded {len(test_df)} testing sample.")

    return train_df, test_df

if __name__ == "__main__":
    train_df, test_df = load_imdb_data(sample_size=1000)
    print(train_df.head())
    print(test_df.head())