import numpy as np

#Example matrix: 4 samples x 3 features
A = np.array([
	[1.0, 2.0, 3.0],
	[4.0, 5.0, 6.0],
	[7.0, 8.0, 9.0],
	[3.0, 2.0, 5.9],
])

print('Original matrix: \n',A)
print('Original Shape: ', A.shape)

#SVD calculation

#Step 1: A^T*A
AtA = A.T@A

#Step 2: eigenvalues and eigenvectors of AtA
eigenVal_AtA, V = np.linalg.eigh(AtA)

#Sorting eigenvalues in descending order
idx = np.argsort(eigenVal_AtA)[::-1]
eigenVal_AtA = eigenVal_AtA[idx]
V= V[:, idx]


#Step 3: Calculate Singular values
sigma = np.sqrt(np.maximum(eigenVal_AtA,0))

#Step 4: Compute U matrix
U = np.zeros((A.shape[0], len(sigma)))
for i in range(len(sigma)):
	if sigma[i] > 1e-10:
		U[:, i]=(A@ V[:,-1]) / sigma[i]


#Create V^T
Vt = V.T

#print result
print('SVD results: \n')
print('U shape: ',U.shape)
print('Sigma : ', sigma)
print('V^T shape', Vt.shape)
