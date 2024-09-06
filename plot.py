import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from utils import read_and_parse

xs, _ = read_and_parse('osu')
print(xs.shape)

xs = StandardScaler().fit_transform(xs)

pca = PCA(n_components=2)
pca.fit(xs)

print(pca.explained_variance_ratio_)

print(pca.singular_values_)

comps = pca.components_

plt.scatter(comps[0], comps[1])
plt.show()
