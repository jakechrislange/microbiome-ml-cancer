import pandas as pd

# Load the CSV you saved previously
dataset = pd.read_csv("TCMA_Genus_Processed/final.csv")

print(dataset.head())

start_idx = dataset.columns.get_loc("HistologicalType") + 1
X = dataset.iloc[:, start_idx:] 
y = dataset["project"]  

print("X shape:", X.shape)
print("y shape:", y.shape)

print(X.head())
print(y.head())