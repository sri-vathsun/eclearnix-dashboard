from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


df = pd.read_csv('ECLEARNIX.csv')

features = df[['Time_Spent_Total_Minutes', 'Days_Since_Last_Activity', 'Feedback_Rating']]
scaler = StandardScaler()
X = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

sns.scatterplot(x='Days_Since_Last_Activity', y='Time_Spent_Total_Minutes', hue='Cluster', data=df)
plt.title("User Segments by Engagement")
plt.show()
