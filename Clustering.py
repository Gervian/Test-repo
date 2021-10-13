# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 14:04:51 2021

@author: Gervian
"""

#Import packages
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

#Standardizing variables
col_names = ['Leverage', 'Profitability', 'Market_Ratio']
features = df_filtered[col_names]
scaler = StandardScaler().fit(features.values)
features = scaler.transform(features.values)
scaled_features = pd.DataFrame(features, columns = col_names)
scaled_features.head()


#One hot-encoding
gics_sector = df_filtered['GICS_Sectors']
newdf = scaled_features.join(gics_sector.reset_index())

newdf = pd.get_dummies(newdf, prefix=None, prefix_sep='_', dummy_na=False, columns=None, sparse=False, drop_first=True, dtype=None)
newdf = newdf.set_index('index')

newdf.head()


#Building GICS clustering
SSE = []

for cluster in range(1,10):
    kmeans = KMeans(n_jobs = -1, n_clusters = cluster, init='k-means++')
    kmeans.fit(newdf)
    SSE.append(kmeans.inertia_)

# converting the results into a dataframe and plotting them

frame = pd.DataFrame({'Cluster':range(1,10), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['Cluster'], frame['SSE'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#First build 6 clusters and determine model performance
kmeans = KMeans(n_jobs = -1, n_clusters =4, init='k-means++')
kmeans.fit(newdf)

# Now, print the silhouette score of this model

print(silhouette_score(newdf, kmeans.labels_, metric='euclidean'))


#PLotting the clusters
clusters = kmeans.fit_predict(newdf.iloc[:,1:])
newdf["label"] = clusters
 
fig = plt.figure(figsize=(21,10))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(newdf.Leverage[newdf.label == 0], newdf["Profitability"][newdf.label == 0], newdf["Market_Ratio"][newdf.label == 0], c='blue', s=60)
ax.scatter(newdf.Leverage[newdf.label == 1], newdf["Profitability"][newdf.label == 1], newdf["Market_Ratio"][newdf.label == 1], c='red', s=60)
ax.scatter(newdf.Leverage[newdf.label == 2], newdf["Profitability"][newdf.label == 2], newdf["Market_Ratio"][newdf.label == 2], c='green', s=60)
ax.scatter(newdf.Leverage[newdf.label == 3], newdf["Profitability"][newdf.label == 3], newdf["Market_Ratio"][newdf.label == 3], c='orange', s=60)

plt.title('CLUSTERING k=4', 
          fontsize = 22)
ax.set_xlabel('Leverage', 
              fontsize = 16)
ax.set_ylabel('Profitability', 
              fontsize = 16)
ax.set_zlabel('Market_Ratio', 
              fontsize = 16)
plt.legend(loc = 'upper left', fontsize = 14)

ax.view_init(25, 200)
plt.show()

#PCA on dataset
pca = PCA(n_components=4)
principalComponents = pca.fit_transform(newdf)

features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)

PCA_components = pd.DataFrame(principalComponents)


#Plotting different PCA
ks = range(1, 10)
inertias = []

for k in ks:
    model = KMeans(n_clusters=k)
    model.fit(PCA_components.iloc[:,:2])
    inertias.append(model.inertia_)

plt.plot(ks, inertias, '-o', color='black')
plt.xlabel('number of clusters, k')
plt.ylabel('inertia')
plt.xticks(ks)
plt.show()


#Clustering for different Scores

