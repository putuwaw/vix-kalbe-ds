# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# %%
# load data
df_customer = pd.read_csv('../data/Case Study - Customer.csv', delimiter=';')
df_transaction = pd.read_csv('../data/Case Study - Transaction.csv', delimiter=';')

# %%
# fill missing values on df_customer
df_customer.fillna(method='ffill', inplace=True)

# convert categorical data to numerical
df_customer['Marital Status'] = df_customer['Marital Status'].apply(lambda x: 1 if x == 'Married' else 0)

# convert Income to float
df_customer['Income'] = df_customer['Income'].apply(lambda x: x.replace(',', '.')).astype(float)

# %%
# convert Date to datetime
df_transaction['Date'] = pd.to_datetime(df_transaction['Date'], format='%d/%m/%Y')

# %%
# merge df_transaction and df_customer
merged_df = pd.merge(df_transaction, df_customer, on='CustomerID', how='left')
merged_df.head()

# %%
merged_df.info()

# %%
# aggregate data
agg = {
    'TransactionID': 'count',
    'Qty': 'sum',
    'TotalAmount': 'sum'
}
cluster_df = merged_df.groupby('CustomerID').aggregate(agg).reset_index()
cluster_df.head()

# %%
# scale data into same range
scaler = StandardScaler()
scaled_df = scaler.fit_transform(cluster_df[['TransactionID', 'Qty', 'TotalAmount']])
scaled_df = pd.DataFrame(scaled_df, columns=['TransactionID', 'Qty', 'TotalAmount'])
scaled_df.head()

# %%
# finding optimal number of clusters
inertia = []
max_clusters = 11
for n_cluster in range(1, max_clusters):
    kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=n_cluster)
    kmeans.fit(cluster_df.drop('CustomerID', axis=1))
    inertia.append(kmeans.inertia_)

# %%
plt.figure(figsize=(10,8))
plt.plot(np.arange(1, max_clusters), inertia, marker='o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.xticks(np.arange(1, max_clusters))
plt.show()

# %%
# create cluster
n_cluster = 3
kmeans = KMeans(n_clusters=n_cluster, random_state=42, n_init=n_cluster)
kmeans.fit(cluster_df.drop('CustomerID', axis=1))
cluster_df['Cluster'] = kmeans.labels_

# %%
# plot cluster
cluster_df.plot(kind='scatter', x='Qty', y='TotalAmount', c='Cluster', cmap='viridis', figsize=(10,8), legend=True)


