import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import scipy.stats as stats 

data = pd.read_csv('cleaned_laptop_data.csv')


data.dropna(inplace=True)


X = data[['TypeName', 'resolution_parsed', 'Ram', 'first_storage', 'Weight']] 
y = data['Price_in_euros']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test) 

# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# print("Coefficients:", model.coef_)
# print("Intercept:", model.intercept_)

# res = model.intercept_ + model.coef_[0]*3 + model.coef_[1]*0 + model.coef_[2]*4 + model.coef_[3]*500 + model.coef_[4]*2.1
# print(res)

# -----------------------------------Menu Driven part-------------------------------------

# print("Enter the values of the new product : \n")

# print("Type Name:\n0-2 in 1 convertible\n1-Gamming\n2-Neook\n3-NoteBook\n4-UltraBook\n5-Workstation")
# val1=float(input("Enter your choice : "))
# if (val1>5 or val1<0):
#     val1=0
    
# print("Resolution\n0-1366x768\n1-1440x900\n2-1600x900\n3-Full HD 1920x1080\n4-IPS Panel Touchscreen 1920x1200\n5-IPS Panel Retina Display 2304x1440\n6-Touchscreen / Quad HD+ 3200x1800\n7-IPS Panel Retina Display 2560x1600\n8-IPS Panel Retina Display 2880x1800\n9-IPS Panel Full HD / Touchscreen 1920x1080")
# val2=float(input("Enter your choice"))
# if (val2>9 or val2<0):
#     val2=0

# val3=float(input("Ram(in ) : "))
# if (val3<0):
#     val3=0

# val4=float(input("Enter First Storage (in ) : "))
# if (val4<0):
#     val4=0

# val5=float(input("Enter Weight (in KG) : "))
# if (val5<0):
#     val5=0
    

# print("Price predicted according to the model is : ")

# predict=model.intercept_ + model.coef_[0]*val1 + model.coef_[1]*val2 + model.coef_[2]*val3 + model.coef_[3]*val4 + model.coef_[4]*val5
# print(predict)


plt.figure(figsize=(10, 6))

plt.scatter(y_test, y_pred, color='blue', label='Actual vs Predicted')


plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', lw=2, label='Regression Line')



plt.title('Actual vs Predicted')
plt.xlabel('Actual Price (in euros)')
plt.ylabel('Predicted Price (in euros)')
plt.legend()
plt.grid(True)
plt.show()

