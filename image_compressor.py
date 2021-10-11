import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import seaborn as sns
import os
# sns.set()
# from skimage import color


class ImgCompress:
    
    def __init__(self, image, n_clusters, save_dir, img_name):

        self.image = image  # range [0, 1] RGB img
        self.n_clusters = n_clusters
        self.save_dir = save_dir
        if os.path.exists(self.save_dir) is False: os.mkdir(self.save_dir)

        self.img_name = img_name
        self.save_dir = os.path.join(self.save_dir, self.img_name)
        if os.path.exists(self.save_dir) is False: os.mkdir(self.save_dir)

        self.features = self.image.reshape(-1,3)
        self.x, self.y, self.z = self.image.shape
        
        # using KMeans to cluster our data to n_clusters
        clusterer = cluster.KMeans(n_clusters,
                                      random_state=42
                                      )
        clusterer.fit(self.features)
        self.clusterer = clusterer
        
        self.cluster_centers = self.clusterer.cluster_centers_
        # self.cluster_centers = color.lab2rgb(self.cluster_centers[None, ...]).squeeze()
        self.cl_centers_rgb = (self.cluster_centers * 255.).astype(np.uint8)  #  [n_cl, 3]
        self.cluster_labels = self.clusterer.labels_
        self.sort_colors()
        self.save_colors()
        self.save_palette()


        
        self.image_compressed = self.image_compression(save=True)
   
    def image_compression(self, save=True):
        # replacing pixel colors with corresponding cluster centroid's colors
        # and returning to the original size
#         img_compressed = self.cluster_centers[:,2:5][self.cluster_labels].astype('uint8')
        img_compressed = self.cluster_centers[self.cluster_labels]
        image_compressed = img_compressed.reshape(self.x, self.y, self.z)
        if save == True:
            file_path = os.path.join(self.save_dir, self.img_name + '_'+ str(self.n_clusters) + 'colors.png')
            plt.imsave(file_path, image_compressed)
        return image_compressed

  
    def compare_images(self):
        # plotting original vs compressed
        fig, axes = plt.subplots(1, 2, figsize = (10, 5))
        axes[0].imshow(self.image)
        axes[0].set_title('Original Image')
        axes[1].imshow(self.image_compressed)
        axes[1].set_title('Compressed Image with {} colors'.format(self.n_clusters))
        for ax in axes:
            ax.axis('off')
        
    def compare_memory(self):
        print('Original image size: {} kB'.format(round(os.path.getsize(self.name)/1024,2)))
        print('Compressed image size: {} kB'.format(round(os.path.getsize('compressed.jpg')/1024, 2)))
        
        
    def change_n_clusters(self, n_clusters):
        # changing number of used clusters(colors) if needed
        self.n_clusters=n_clusters
        print('New  number of clusters is set to {}.'.format(n_clusters))
        clustered_im = cluster.KMeans(n_clusters)
        clustered_im.fit(self.features)
        
        self.cluster_centers = clustered_im.cluster_centers_
        self.cluster_labels = clustered_im.labels_
        
        self.image_compressed = self.image_compression()    
        
    def centroids_palette(self):
        # plotting centroids color palette
        color_palette = self.cluster_centers
        sns.palplot(color_palette, size=0.4)
        # plt.show()

        
    def get_closest_color(self, colors):
        """

        :param colors: [batch_size, 3]
        :return:
        """
        
        labels = self.clusterer.predict(colors)
        labels = np.array([self.sort_dict[l] for l in labels])
        discrete_colors = self.cluster_centers[labels]
        assert discrete_colors.shape == colors.shape
        return labels, discrete_colors

    def sort_colors(self):
        r, g, b = self.cl_centers_rgb.T
        sort_idx = np.lexsort((b, g, r))
        sort_dict = dict(zip(list(sort_idx), list(range(len(sort_idx)))))
        self.sort_dict = sort_dict
        self.cl_centers_rgb = self.cl_centers_rgb[sort_idx]
        for i in range(len(self.cluster_labels)):
            self.cluster_labels[i] = sort_dict[self.cluster_labels[i]]
        self.cluster_centers = self.cluster_centers[sort_idx]


    def save_colors(self):
        file_dir = os.path.join(self.save_dir, self.img_name + f'_{self.n_clusters}colorsRGB')
        np.save(file_dir, self.cl_centers_rgb)

    def save_palette(self):
        color_palette = self.cluster_centers
        # print('SAVE_DIR :', self.save_dir)
        # file_dir = self.save_dir + f'_{self.n_clusters}colors_palette'
        file_dir = os.path.join(self.save_dir, self.img_name + f'_{self.n_clusters}colors_palette')
        sns.palplot(color_palette, size=0.6)
        plt.savefig(file_dir)




if __name__ == "__main__":
    import torch
    import numpy as np
    from work import args
    # path = '/Users/jiji/Desktop/Учеба/ВШЭ/MLDM (тетрадки+слайды)/Dataset_T1/parrot1.jpg'
    img_ = cv2.imread(args['img_path'], cv2.IMREAD_COLOR)
    img_ = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    z = img_.reshape(-1, 3)
    clusterer = cluster.KMeans(6,
                               # random_state=42
                               )
    clusterer.fit(z)
