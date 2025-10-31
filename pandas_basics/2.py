import pandas as pd

#in the same directory as this script, there is a file named "titanic.csv"
# Load the CSV file into a DataFrame
data = pd.read_csv("titanic.csv")


# Select all rows where the 'Age' column is less than 30
d1 = data[data['Age'] < 30]

# Select all rows where the 'Age' column is greater than 30
d2 = data[data['Age'] > 30]

# Data with men and age between 20 and 40
d3 = data[(data['Age'] > 20) & (data['Age'] < 40) & (data['Sex'] == 'male')]

# Create a new column 'AgeGroup' that categorizes passengers as 'Adult' (18 and older) or 'Child' (under 18)
d4 = data['AgeGroup'] = data['Age'].apply(lambda x: 'Adult' if x >= 18 else 'Child')

print(d1.head)
print(d2.head)
print(d3.head)
print(d4.head)