import numpy as np
from sklearn import cluster
import seaborn as sns
import matplotlib.pyplot as plt
import os


class ImgCompressor:

    def __init__(self, img_ref, n_colors, sample_size=None, random_state=42):
        self.img_reference = img_ref
        self.white_mask = np.alltrue(np.isclose(img_ref, 1., atol=0.1), axis=-1)  # axis = color_axis
        self.h, self.w, self.n_channels = self.img_reference.shape
        self.n_colors = n_colors
        self.sample_size = sample_size
        self.random_state = random_state

        self.indices = np.arange(self.h*self.w)[~self.white_mask.ravel()]
        if self.sample_size:
            np.random.shuffle(self.indices)
            self.indices = self.indices[:self.sample_size]
        self.quantizer = cluster.KMeans(self.n_colors, random_state=self.random_state)
        self.quantizer.fit(self.img_reference.reshape(-1, self.n_channels)[self.indices])
        self.sort_indices = self._argsort(colors=self.quantizer.cluster_centers_)
        self.quantizer.cluster_centers_ = self.quantizer.cluster_centers_[self.sort_indices]
        self.quantizer.labels_ = np.argsort(self.sort_indices)[self.quantizer.labels_]

        colors = self.quantizer.cluster_centers_.copy()
        self.colors = colors
        labels = self.quantizer.predict(img_ref.reshape(-1, self.n_channels))
        self.img_compressed = colors[labels].reshape(img_ref.shape)
        self.img_compressed[self.white_mask] = np.array([1., 1., 1.], dtype=np.float32)

    @staticmethod
    def _argsort(colors):
        r, g, b = colors.T
        sort_indices = np.lexsort((b, g, r))
        return sort_indices

    def save_colors(self, save_dir):
        save_path = os.path.join(save_dir, f'{self.n_colors}colorsRGB.npy')
        np.save(save_path, self.colors)

    def save_palette(self, save_dir):
        save_path = os.path.join(save_dir, f'{self.n_colors}colors_palette')
        sns.palplot(self.colors, size=0.8)
        plt.savefig(save_path)

    def get_closest_color(self, colors):
        labels = self.quantizer.predict(colors)
        colors_quantized = self.colors[labels]
        return labels, colors_quantized

    def compress_camera_canvas(self, img, cc_white, color_indices):

        h, w, n_channels = img.shape
        # indices = np.arange(h*w)
        # if self.sample_size:
        #     np.random.shuffle(indices)
        #     indices = indices[:self.sample_size]
        n_colors = len(color_indices)

        # build white mask ====
        # quantizer = cluster.KMeans(n_clusters=n_colors+1, random_state=self.random_state)  # +1 for canvas white
        # quantizer.fit(img.reshape(-1, 3)[indices])
        #
        # colors = quantizer.cluster_centers_.copy()
        # labels = quantizer.predict(img.reshape(-1, 3))
        # cc_white_idx = np.argmin(np.sum(np.abs(colors - cc_white), axis=-1))
        # white_mask = (labels == cc_white_idx).reshape(h, w)
        white_mask = np.alltrue(np.isclose(img, cc_white, atol=0.1), axis=-1)
        # ======
        indices = np.arange(h*w)[~white_mask.ravel()]
        if self.sample_size:
            np.random.shuffle(indices)
            indices = indices[:self.sample_size]
        quantizer = cluster.KMeans(n_clusters=n_colors, random_state=self.random_state)
        quantizer.fit(img.reshape(-1, 3)[indices])

        colors = quantizer.cluster_centers_.copy()
        sort_idx = self._argsort(colors)
        colors = colors[sort_idx]
        quantizer.cluster_centers_ = colors.copy()
        labels = quantizer.predict(img.reshape(-1, 3))
        img_compressed = colors[labels].reshape(img.shape)
        img_compressed[white_mask] = np.array([1.,1.,1.])
        return img_compressed



    # def compress_camera_canvas(self, img, cc_white, color_indices):
    #     # cc_white for camera canvas white color
    #     h, w, _ = img.shape
    #     indices = np.arange(0, h*w)
    #     if self.sample_size:
    #         np.random.shuffle(indices)
    #         indices = indices[:self.sample_size]
    #     n_colors = len(color_indices)
    #     quantizer = cluster.KMeans(n_clusters=n_colors+1, random_state=self.random_state)  # +1 for cc_white
    #     quantizer.fit(img.reshape(-1, 3)[indices])
    #
    #
    #     colors = quantizer.cluster_centers_.copy()
    #     sort_idx = self._argsort(colors)
    #     colors = colors[sort_idx]
    #     cc_white_idx = np.argmin(np.sum(np.abs(colors - cc_white), axis=-1))
    #     colors[[cc_white_idx, -1]] = colors[[-1, cc_white_idx]]
    #     quantizer.cluster_centers_ = colors.copy()
    #
    #     labels = quantizer.predict(img.reshape(-1, 3))
    #     color_indices = np.sort(color_indices)
    #     colors = np.vstack([self.colors[color_indices], np.array([1., 1., 1.])])
    #     img_compressed = colors[labels].reshape(img.shape)
    #     return img_compressed
