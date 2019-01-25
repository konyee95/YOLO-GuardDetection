import numpy as np

hours = [8, 9, 10, 11]
count = [489, 2306, 991, 1905]

zeroes = np.zeros(hours[0]).astype(np.uint8) # create an array of zeroes
results = np.append(zeroes, count) # append the count after zeroes


# fill up the rest with 0
rest = 24 - len(results)
results = np.append(results, np.zeros(rest).astype(np.uint8))
print(results)