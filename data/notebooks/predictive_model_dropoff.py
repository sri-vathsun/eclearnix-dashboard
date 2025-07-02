from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

import pandas as pd

df = pd.read_csv('ECLEARNIX.csv')

X = df[['App_Installed', 'Days_Since_Last_Activity', 'Saved_Event_Count', 'Feedback_Rating', 'Time_Spent_Total_Minutes']]
y = df['Course_Completed']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
