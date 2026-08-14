"""Print sample questions per intent."""
import pandas as pd

df = pd.read_csv("data/processed/faq_dataset.csv")
g = df.groupby("intent")["question"].apply(lambda s: s.head(4).tolist())
for intent, qs in g.items():
    print("==", intent)
    for q in qs:
        print("   -", q)
