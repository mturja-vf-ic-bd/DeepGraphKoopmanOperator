import numpy as np
from CONSTANTS import CONSTANTS
import os
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt


def pearson2(X):
    m = torch.mean(X, dim=-1, keepdim=True)
    s = torch.std(X, dim=-1, keepdim=True)
    X = X - m
    return torch.matmul(X, X.transpose(-1, -2)) / torch.matmul(s, s.transpose(-1, -2)) / (X.shape[-1] - 1)


subjectIDs = np.loadtxt(os.path.join(CONSTANTS.HOME, "subjectIDs.txt"))
id = 5
count = 12
parcel = 50
T = 1200
sess = 0
fmri_signal = np.zeros((count, T, parcel), dtype=float)
for i, s in enumerate(subjectIDs):
    if i >= count:
        break
    fmri_signal[i] = np.loadtxt(
        os.path.join(CONSTANTS.HOME,
                     "node_timeseries",
                     f"3T_HCP1200_MSMAll_d{parcel}_ts2",
                     f"{int(s)}.txt"))[sess * T : (sess + 1) * T]
    if np.isnan(fmri_signal[i]).any():
        print(f"Subject {s} has NaN values")

win = 128
stride = 4
fmri_signal = torch.FloatTensor(fmri_signal).unfold(1, win, stride)
triu_indices = torch.triu_indices(50, 50, offset=1)
corr = pearson2(fmri_signal)[:, :, triu_indices[0], triu_indices[1]]
V = corr.flatten(start_dim=2)
pca = PCA(n_components=2, random_state=1)
tsne = TSNE(n_components=2, random_state=1)
embedding = pca.fit_transform(V[id])
print(embedding.shape)

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Create a scatterplot
color_map = plt.cm.get_cmap('viridis', len(embedding) - 1)
scatter_colors = [mcolors.rgb2hex(color_map(i)) for i in range(len(embedding))]
plt.figure(figsize=(20, 20))
plt.scatter(embedding[:, 0], embedding[:, 1], label='Data Points', color=scatter_colors, s=50)

# Connect consecutive points with lines
for i in range(len(embedding) - 1):
    line_color = mcolors.rgb2hex(color_map(i))
    plt.plot([embedding[i, 0], embedding[i+1, 0]], [embedding[i, 1], embedding[i+1, 1]], color=line_color)

# Add labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Scatterplot with Lines Between Consecutive Points')
plt.legend()
plt.tight_layout()
# Show the plot
plt.show()

