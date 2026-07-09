import pandas as pd 
import numpy as np 
import joblib
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,root_mean_squared_error,classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("C:\\Users\\Abdul Rehman\\Documents\\Datasets\\balanced_customer_health_dataset.csv")

X = df[['Age', 'Gender', 'Tenure', 'Usage Frequency', 'Subscription Type', 
        'Contract Length', 'Total Spend', 'Last Interaction', 
        'Monthly_Spend_Rate', 'Interaction_To_Usage_Ratio', 'Lifecycle_Ratio', 'Spend_Per_Interaction']]

Y = df['Customer_Health']

x_train, x_test, y_train, y_test = train_test_split(X,Y,random_state=42,test_size=0.25)

model = RandomForestClassifier(random_state=42,class_weight='balanced')

model.fit(x_train,y_train)

predicted_y = model.predict(x_test)

print(classification_report(y_test,predicted_y))

print(confusion_matrix(y_test,predicted_y))

joblib.dump(model, 'random_forest_health_model.pkl')

