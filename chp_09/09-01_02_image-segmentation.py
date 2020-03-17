# 9.1.1 K-means Image Segmentation

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.cluster import KMeans
import numpy as np

img = mpimg.imread("datasets/image/example.jpg")
n, m = img.shape[0], img.shape[1]
data = img.reshape(n * m, 3)

k_list = [2, 3, 10]
for k in range(len(k_list)):
    K = k_list[k]
    kmeans = KMeans(n_clusters=K, random_state=0).fit(data)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    centers = centers.astype(np.int)
    print("K-%d, centers=" % K)
    print(centers)
    seg_img = []
    for i in range(len(labels)):
        seg_img.append(centers[labels[i]])
    seg_img = np.array(seg_img)
    seg_img = seg_img.reshape(n, m, 3)
    plt.subplot(1, 4, k + 1)
    plt.imshow(seg_img)

plt.subplot(1, 4, 4)
plt.imshow(img)
plt.show()
