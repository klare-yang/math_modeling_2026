import numpy as np

np.random.seed(123)

arr = np.random.rand(3,4)
print(arr)

arr1 = np.random.uniform(low = 5,high = 10,size = (3,4))
print(arr1)

arr2 = np.random.randint(low = 2,high = 20,size = (4,4))
print(arr2)
print(arr2[(arr2 > 1) & (arr2 < 5)])

arr3 = arr2[arr2 < 10]
print(arr3)
print(arr3.sum())
print(np.sum(arr3))

arr4 = np.random.randint(0,100,16).reshape(4,4)
print(np.sum(arr4[arr4 <= 10]))