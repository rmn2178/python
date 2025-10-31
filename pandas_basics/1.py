import pandas as pd

#in the same directory as this script, there is a file named "titanic.csv"
# Load the CSV file into a DataFrame
data = pd.read_csv("titanic.csv")

# Display the first few rows of the DataFrame
print(data.head())

# Display the last few rows of the DataFrame
print(data.tail())

# Display the shape of the DataFrame (rows, columns)
print(data.shape)

# To print the columns of the DataFrame
print(data.columns)

# Display summary information about the DataFrame
print(data.info())

# Describe the DataFrame to get statistical summary
print(data.describe())


# These are the code to clean the code
data.dropna(inplace=True)  # Remove rows with missing values

data.drop_duplicates(inplace=True)  # Remove duplicate rows

data.fillna(0) # Fill missing values with 0