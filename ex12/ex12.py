import pandas as pd
from sklearn.preprocessing import StandardScaler
import scipy.cluster.hierarchy as sch
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

data = pd.read_csv("dendogram.csv")
df = pd.DataFrame(data)
print("Sample Product Data:\n", df)
features = df[['Price', 'Sales', 'Rating', 'Feature_Score']]
scale = StandardScaler()
scaledata = scale.fit_transform(features)
link = linkage(scaledata, method='ward')

# Step 5: Plot Dendrogram
plt.figure(figsize=(10, 6))
dendrogram(link, labels=df['Product_Name'].values)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Products')
plt.ylabel('Euclidean Distance')
plt.show()

# Step 6: Form clusters (e.g., 3 clusters)
df['Cluster'] = fcluster(link, t=3, criterion='maxclust')

print("\nClustered Products:\n", df[['Product_Name', 'Cluster']])
