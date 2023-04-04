import scipy.misc
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib import image
import matplotlib.pyplot as plt
def plot_means(means):

    k = means.shape[0]
    rows = k // 5 + 1
    columns = min(k, 5)

    for i in range(k):
        plt.subplot(rows, columns, i + 1)
        plt.imshow(Image.fromarray((means[i].reshape(28,28)*255).astype('uint8'), mode='L').convert('RGB'))


