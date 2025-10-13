import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
data = pd.read_csv("cluster.csv")
df = pd.DataFrame(data)
print("Sample data:\n", df.head()) # display top 5 rows
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])  # Female=0, Male=1

scale = StandardScaler()
scaledata = scale.fit_transform(df)

km = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = km.fit_predict(scaledata)
print("\nCluster centers (scaled):\n", km.cluster_centers_)
print("\nCluster labels:\n", df['Cluster'].value_counts())

summary = df.groupby('Cluster').mean(numeric_only=True)
print("\nCluster Summary:\n",summary)

plt.figure(figsize=(8,6))
plt.scatter(df['Annual_Income'], df['Spending_Score'], c=df['Cluster'], cmap='viridis', s=100)
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Customer Segments (K-Means Clustering)')
plt.show()
