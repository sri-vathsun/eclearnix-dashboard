import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import pandas as pd

df = pd.read_csv('ECLEARNIX.csv')



df['Churn'] = df['Days_Since_Last_Activity'] > 30  # Churn if no activity > 30 days

X = df[['App_Installed', 'Time_Spent_Total_Minutes', 'Feedback_Rating', 'Saved_Event_Count']]
y = df['Churn'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))
