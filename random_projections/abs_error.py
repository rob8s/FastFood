import numpy as np
import sklearn as sc
import projections
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
from sklearn_extra.kernel_approximation import Fastfood

d = 10  # Input dimension
num_data = 4000  # Number of data points 
x = np.random.rand(num_data, d)

scale = 0.5
#exact rbf kernel
exact_rbf = rbf_kernel(x, gamma=(1/(2*scale**2)))

dimensions = [1,2,4,8,16,32,64,128,512,1024,2048]
rks_error = []

#rks error
for dim in dimensions:
    #rks approx
    rks = projections.rks(dim, scale)
    rks.fit(x)
    rks_approx = rks.transform(x)

    rks_approx = np.matmul(rks_approx, rks_approx.T)

    difference = np.linalg.norm(np.abs(exact_rbf-rks_approx), 'fro')
    rks_error.append(difference/num_data)

#ff error
ff_error = []
for dim in dimensions:
    #ff approx
    ff = Fastfood(sigma=scale, n_components=dim)
    ff_approx = ff.fit_transform(x)

    ff_approx = np.matmul(ff_approx, ff_approx.T)

    difference = np.linalg.norm(np.abs(exact_rbf-ff_approx), 'fro')
    ff_error.append(difference/num_data)


plt.plot(dimensions,rks_error, label='RBF_Approx', marker='o')
plt.plot(dimensions,ff_error, label='FF_Approx', marker='o')
plt.xlabel('Dimension (n)')
plt.ylabel('Error')
plt.title('Error Plot')
plt.ylim(0, 0.5)
plt.legend()
plt.savefig('abs_error.png', bbox_inches='tight')
plt.show()
