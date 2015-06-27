import numpy as np
from scipy.misc import imresize
from numpy.core.fromnumeric import reshape
from scipy.cluster.vq import vq #, whiten, kmeans
from skimage.segmentation import quickshift

def segment(img):
    """
    segment an image

    @type img; LAB image
    """
    if 3 != img.ndim:
        return None

    # convert to lab
    #img_lab = rgb2lab(img)

    #print provider.img
    # quickshift
    qs = quickshift(img, return_tree=False, convert2lab=False)

    return qs


def quantize(img, palette, density=True, size=0.10):

    # resize image
    img = imresize(img, size)
    #img = imresize(img, size)

    # reshape to array of points
    pixels = reshape(img,(img.shape[0]*img.shape[1],3))
    #print pixels.shape

    #whitened = whiten(pixels)
    #print whitened.shape
    #print whitened

    # custom pallete or adaptive
    #if colors is None:
    #  centroids, _ = kmeans(pixels, 8)
    #print "centroid shape"
    #print centroids.shape
    #else:
    #  centroids = colors

    # quantize
    qnt, _ = vq(pixels, palette)
    #print qnt

    # reshape back to image
    centers_idx = reshape(qnt,(img.shape[0], img.shape[1]))
    clustered = palette[centers_idx]

    #if False:
    #  print "centroids"
    #  print centroids.shape
    #  print centroids
    #
    #  print "quantize"
    #  print qnt.shape
    #  print qnt
    #
    #  print "centers_idx"
    #  print centers_idx.shape
    #  print centers_idx
    #
    #  print "clusters"
    #  print clustered.shape
    #  print clustered

    h, _ = np.histogram(qnt, len(palette), range=(0,len(palette)-1), normed=density)
    #print h
    return clustered, h

    #imsave('out.png', clustered)
    #centers_idx = reshape(qnt,(img.shape[0],img.shape[1]))

    #book = array((whitened[0], whitened[2]))

    #print book

    #km = kmeans(whitened, book)

    #print km
