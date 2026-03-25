import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv('Cellphone.csv')

features = ['ram', 'battery', 'internal mem', 'cpu core', 'RearCam', 'thickness']
X = df[features]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {round(accuracy, 2)}")

joblib.dump(model, 'model.pkl')
print("Model saved!")

plt.scatter(y_test, y_pred, color='blue', alpha=0.1)
plt.plot([0, 4500], [0, 4500], color='red')
plt.xlim(0, 4500)
plt.ylim(0, 4500)
plt.xlabel('Real Price ($)')
plt.ylabel('Predicted Price ($)')
plt.title('Real vs Predicted')
# plt.grid(True)
plt.show()