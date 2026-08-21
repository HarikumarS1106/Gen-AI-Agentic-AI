from sklearn.preprocessing import OneHotEncoder
import numpy as np

corpus = ['dog', 'cat', 'dog', 'fish']

corpus = np.array(corpus).reshape(-1, 1)

onehot_encoder = OneHotEncoder()

onehot_encoder = onehot_encoder.fit_transform(corpus)

print(onehot_encoder.toarray())