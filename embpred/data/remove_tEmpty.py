from embpred.config import PROCESSED_DATA_DIR
import pandas as pd

path = 'all-classes_carson-224-3depths.csv'

# load csv from path
df = pd.read_csv(PROCESSED_DATA_DIR / path)

# print the number of rows in dataframe
print(len(df))

# print unique values and counts in 2nd column
print(df.iloc[:, 1].value_counts())

# remove all rows where 2nd column has a value of 13
df = df[df.iloc[:, 1] != 13]

# print the number of rows in dataframe
print(len(df))

# print unique values and counts in 2nd column
print(df.iloc[:, 1].value_counts())

# save the dataframe to a new csv file
df.to_csv(PROCESSED_DATA_DIR / 'all-classes_carson-224-3depths_no13.csv', index=False)