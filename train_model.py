import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pickle, os

os.makedirs("model", exist_ok=True)

df = pd.read_csv("Mall_Customers.csv")
features = ["Age", "AnnualIncome", "SpendingScore"]

scaler = StandardScaler()
X = scaler.fit_transform(df[features])

km = KMeans(n_clusters=5, random_state=42, n_init="auto").fit(X)
df["Cluster"] = km.labels_

pickle.dump(scaler, open("model/scaler.pkl", "wb"))
pickle.dump(km,     open("model/kmeans.pkl", "wb"))
df.to_csv("model/clustered_customers.csv", index=False)

print(df.groupby("Cluster")[features].mean().round(1))
print("Done — model saved to model/")