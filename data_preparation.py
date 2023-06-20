from ast import literal_eval
import pandas as pd

# There are multiple genres per movie
df = pd.read_csv("dataset/movies_metadata.csv",usecols=['genres', 'overview'])
# Convert genres string type to list
df['genres'] = df['genres'].apply(lambda x: literal_eval(x)).apply(lambda x: sorted([d['name'] for d in x]))

# Add String literal of genres to make it hashable for some future data cleaning steps
df['genres_str'] = df['genres'].apply(lambda x: " ".join(str(s) for s in x))

total_no_overview = df['overview'].isnull().sum()

df = df.dropna(subset = ['overview']).reset_index(drop=True)

# Remove movies with no genres
total_no_label = sum(df['genres'].apply(lambda x: len(x) == 0))

# Remove movies with no genres
df = df[df['genres'].apply(lambda x: len(x)) > 0]

# Check duplicate rows, i.e. movies and labels
df = df[~df[["overview", "genres_str"]].duplicated()]

# Remove remaining duplictes since there is only 14 of them
df = df[~df["overview"].duplicated(keep=False)]

# Remove overviews with <= 3 words or with str len <= 20
df = df[~((df["overview"].apply(lambda x: len(x.split(" "))) <= 3) | (df["overview"].apply(lambda x: len(x)) <= 20))]

# Remove rows with single occurrence genre combination in order to be able to stratify the train test split on genres.
# If we wanted we could use a better appraoch for example see Multi-label data stratification (http://scikit.ml/stratification.html)
df = df.groupby("genres_str").filter(lambda x: len(x) > 1)[['overview', 'genres']]

# Train and test split
from sklearn.model_selection import train_test_split

test_split = 0.1

train_df, test_df = train_test_split(
    df,
    test_size=test_split,
    stratify=df["genres"].values,
)

train_df.to_csv("dataset/train_data.csv")
test_df.to_csv("dataset/test_data.csv")
