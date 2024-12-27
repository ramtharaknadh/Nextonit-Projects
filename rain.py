import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('rainfall.csv')
#df.head()

X = df[["Temperature (C)", "Humidity (%)", "Wind Speed (km/h)", "Pressure (hPa)"]]
y = df["Rain (1=Yes, 0=No)"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit(X_test)

classifier = RandomForestClassifier(random_state=100)
classifier.fit(X_train,y_train)

pickle.dump(sc, open("scaler.pkl","wb"))
pickle.dump(classifier, open("model.pkl","wb"))
