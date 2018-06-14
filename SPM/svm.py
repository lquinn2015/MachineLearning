import matplotlib.pyplot as plt
from sklean import datasets
from sklean import svm

digits = datasets.load_digits()

print(digits.data)
print(digits.target)
print(digits.images[0])