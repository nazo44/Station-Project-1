import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import plotly.express as px
data = pd.read_csv('advertising.csv')
df = pd.DataFrame(data)
#print(df.head())
#print(df.describe())
#print(df.isnull().sum())
#print(df.shape)

features = ['TV', 'Radio', 'Newspaper']
for feature in features:
  figure = px.scatter(df, x=feature, y='Sales', title=f'Sales vs {feature}')
  figure.show()
  
X = df.drop('Sales', axis=1)
Y = df['Sales']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test,y_pred)
print(f"Mean Squared Error: {mse}")
accuracy = model.score(X_test, Y_test)
print(f"Accuracy: {accuracy}")