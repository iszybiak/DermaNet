import pandas as pd
from sklearn.model_selection import train_test_split

# Download the HAM10000 collection
dataset_path = "../data/HAM10000_images/"

# Loading metadata
df = pd.read_csv("../data/HAM10000_metadata.csv")

# Displaying disease labels
print(df["dx"].value_counts())

# Splitting data into test and train
train_df , test_df = train_test_split(df, test_size=0.2, random_state=42)

