import pandas as pd

df = pd.read_csv("clean_dataset.csv")
classes = {}

for index, row in df.iterrows():
    if row["Labels"] not in classes:
        classes[row["Labels"]] = True

label_counts = df["Labels"].value_counts()
print(f"Numero di classi: {str(label_counts)}")