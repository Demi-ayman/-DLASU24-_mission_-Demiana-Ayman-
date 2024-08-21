import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Load the data
X_train_padding = np.load("X_train_padding.npy").astype(np.float32)
y_train_padding = np.load("y_train_padding.npy").astype(np.float32)

# Flatten the data for dimensionality reduction
X_reshaped = X_train_padding.reshape(X_train_padding.shape[0], -1)

pca = PCA(n_components=0.95)  # Keep 95% of variance
X_pca = pca.fit_transform(X_reshaped)

# Plot the explained variance ratio
import matplotlib.pyplot as plt
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of components')
plt.ylabel('Explained  variance ratio')
plt.show() 
