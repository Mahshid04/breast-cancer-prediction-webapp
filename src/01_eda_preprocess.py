import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


# eda

cols = ['ID','Diagnosis','radius1','texture1','perimeter1','area1','smoothness1','compactness1','concavity1','concave_points1',
        'symmetry1','fractal_dimension1','radius2','texture2','perimeter2','area2','smoothness2','compactness2','concavity2','concave_points2',
        'symmetry2','fractal_dimension2','radius3','texture3','perimeter3','area3','smoothness3','compactness3','concavity3','concave_points3',
        'symmetry3', 'fractal_dimension3']

df = pd.read_csv("data/wdbc.csv", names= cols)

print(df.head(3))
print("shape:",df.shape)
print("missing values:",df.isna().sum())
print("diagnosis count:",df["Diagnosis"].value_counts())

sns.countplot(x= "Diagnosis", data= df)
plt.show()

df.to_csv("data/processed.csv", index=False)

# preprocessing

df = pd.read_csv("data/processed.csv")
df = df.drop(columns= ['ID'])
df['Diagnosis'] = df['Diagnosis'].apply(lambda x: '1' if x == 'M' else '0')
df['Diagnosis'] = df['Diagnosis'].astype(int)

# Split features and target
x = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

x = np.array(x)
y = np.array(y)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# scaling
scaler = StandardScaler()                
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Save train/test datasets
x_train_df = pd.DataFrame(x_train, columns= df.columns[1:])
x_test_df = pd.DataFrame(x_test, columns= df.columns[1:])

y_train_df = pd.DataFrame(y_train)
y_test_df = pd.DataFrame(y_test)


x_train_df.to_csv('data/x_train_df.csv', index=False)
x_test_df.to_csv('data/x_test_df.csv', index=False)

y_train_df.to_csv('data/y_train_df.csv', index=False)
y_test_df.to_csv('data/y_test_df.csv', index=False)

# Save scaler
joblib.dump(scaler, "pkl/scaler.pkl")
