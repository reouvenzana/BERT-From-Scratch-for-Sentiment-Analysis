import pandas as pd
from sklearn.model_selection import train_test_split

# Charger les données
df = pd.read_csv('./data/rotten_tomatoes_movie_reviews.csv')  # Modifier le chemin si nécessaire

# Prétraiter les données
df = df[['reviewText', 'scoreSentiment']]
df = df.dropna()
df['label'] = df['scoreSentiment'].apply(lambda x: 1 if x == 'POSITIVE' else 0)

# Diviser les données en ensembles d'entraînement (90%) et de validation/test (10%)
train_df, temp_df = train_test_split(df, test_size=0.10, stratify=df['label'], random_state=42)

# Diviser l'ensemble de validation/test en validation (5%) et test (5%)
val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df['label'], random_state=42)

# Enregistrer les ensembles dans des fichiers CSV distincts
train_df.to_csv('./data/train_rotten_tomatoes_movie_reviews.csv', index=False)
val_df.to_csv('./data/val_rotten_tomatoes_movie_reviews.csv', index=False)
test_df.to_csv('./data/test_rotten_tomatoes_movie_reviews.csv', index=False)

print(f"Training set saved to './data/train_rotten_tomatoes_movie_reviews.csv' with {len(train_df)} samples.")
print(f"Validation set saved to './data/val_rotten_tomatoes_movie_reviews.csv' with {len(val_df)} samples.")
print(f"Test set saved to './data/test_rotten_tomatoes_movie_reviews.csv' with {len(test_df)} samples.")