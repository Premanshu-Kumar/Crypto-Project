
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats


df = pd.read_excel(r"D:\Semester 4\Python\CryptoData.xlsx")

features = ['High', 'Low', 'Open', 'Volume', 'Marketcap']
target = 'Close'
df = df[features + [target]].dropna() 


print("Data Cleaning:")
print("Shape:", df.shape)
print("Null values:\n", df.isnull().sum())
print("Duplicates:", df.duplicated().sum())



print("\nEDA:")
print(df.describe())
print("\nInfo:")
print(df.info())


print("\nVisualization:")

df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()


plt.figure(figsize=(12, 6))
sns.boxplot(data=df[features])
plt.xticks(rotation=45)
plt.show()


print("\nCorrelation:")
corr = df.corr()
print(corr[target].sort_values(ascending=False))


plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


print("\nHypothesis Testing:")

stat, p = stats.shapiro(df[target].sample(5000, random_state=42))  
print(f"Shapiro-Wilk test for normality of {target}: stat={stat:.3f}, p={p:.3f}")
if p > 0.05:
    print(f"{target} is normally distributed")
else:
    print(f"{target} is not normally distributed")


X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()


model.fit(X_train, y_train)


y_pred = model.predict(X_test)

print("\nResults:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

print("\nRegression Summary:")
for i, col in enumerate(X.columns):
    print(f"{col}: {model.coef_[i]:.3f}")
