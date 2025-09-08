import numpy as np

#Sample data (weight vs height)
x = np.array([30,35,40,45,50,55,60,65])
y = np.array([150,155,160,165,170,175,180,185])

#Data showing
print("Data: ",list(zip(x,y)))

#calculating slope(m) and intercept(b)
#for mx + b
n = len(x)

sum_x = np.sum(x)
sum_y = np.sum(y)
sum_xy = np.sum(x*y)
sum_xx = np.sum(x*x)

#m and b using least square
m=(n*sum_xy - sum_x*sum_y) / (n*sum_xx - sum_x*sum_x)
b=(sum_y - m*sum_x)/n

#printing equation
print(f"Equation: {m:.1f}x+{b:.1f}")

#LEAST SQUARE
y_prediction = m*x + b

#For input x:
x_input=39
y_output= m*x_input + b

print(f"If x={x_input}, then y={y_output}")
