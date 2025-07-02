# Fake collaborative filtering example using feedback rating and event type
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv('ECLEARNIX.csv')

df_enc = df.copy()
df_enc['Event_Type_Code'] = LabelEncoder().fit_transform(df_enc['Event_Type'])

ratings = df_enc.pivot_table(index='User_ID', columns='Event_Type_Code', values='Feedback_Rating').fillna(0)

model = NearestNeighbors(metric='cosine')
model.fit(ratings.values)

# Recommend similar users to a given user
sample_user = 10
distances, indices = model.kneighbors([ratings.values[sample_user]], n_neighbors=3)

print("Users similar to", ratings.index[sample_user], "are:")
for i in indices[0]:
    print(ratings.index[i])
