import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
    # Load the data from CSV files
    info = pd.read_csv("data/info.csv", index_col=0)
    ecg = pd.read_csv("data/ecg.csv")
    rpeaks = pd.read_csv("data/rpeaks.csv")
    waves = pd.read_csv("data/waves.csv")
    analysis = pd.read_csv("data/analysis.csv")

    data = analysis

    # Split the data into training and testing sets (70% training, 30% testing)
    train_data, test_data = train_test_split(data, test_size=0.3, random_state=42)

    return train_data, test_data

# Example usage
train_data, test_data = load_data()
print(train_data.head())
print(test_data.head())
