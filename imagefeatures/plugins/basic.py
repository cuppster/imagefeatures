from . import FeaturePlugin
from collections import namedtuple
from numpy.core.fromnumeric import reshape
from scipy.misc import imresize
import scipy.ndimage as nd
import numpy as np
import sys
#from scipy.fftpack import dct

#
# width/height ratio
#
@FeaturePlugin.register('whratio')
def whratio(provider):
    
    # this works for RGB or GRAYSCALE
    ly, lx = provider.img.shape[:2]
    return 1.0 * lx / ly


#
# sobel GM
#
@FeaturePlugin.register('sobelgm')
def sobelgm(provider):
    """
    mean sobel gradient magnitude

    see: http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
    """
    gray = provider.as_gray()

    #print "gray im", im
    dx = nd.sobel(gray, 0)  # horizontal derivative
    dy = nd.sobel(gray, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    mag *= 100.0 / np.max(mag)

    #sys.exit(1)
    return np.mean(mag)

#
# sobel horizontal
#
@FeaturePlugin.register('sobelh')
def sobelh(provider):

    gray = provider.as_gray()

    # see: http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
    im = gray.astype('int32')
    dx = nd.sobel(im, 0)  # horizontal derivative

    #mag = 100.0 / np.max(dx)  # normalize (Q&D)

    return 100.0 * np.mean(dx)

#
# sobel horizontal
#
@FeaturePlugin.register('sobelv')
def sobelv(provider):

    gray = provider.as_gray()

    # see: http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
    im = gray.astype('int32')
    dy = nd.sobel(im, 1)  # horizontal derivative

    #mag = 100.0 / np.max(dx)  # normalize (Q&D)

    return 100.0 * np.mean(dy)

#
# sobel direction
#
@FeaturePlugin.register('sobeld')
def sobeld(provider):

    gray = provider.as_gray()

    # see: http://stackoverflow.com/questions/7185655/applying-the-sobel-filter-using-scipy
    im = gray.astype('int32')
    dx = nd.sobel(im, 0)  # horizontal derivative
    dy = nd.sobel(im, 1)  # horizontal derivative

    dirs = np.arctan2(dy, dx)

    return 100.0 * np.mean(dirs)

#
# corners
#
@FeaturePlugin.register('corners')
def corners(provider):

    from skimage.feature import corner_harris, corner_subpix, corner_peaks
    #from skimage.feature import corner_harris

    gray = provider.as_gray()

    # TODO custom parameters would give arise to exceptions of mis-matched shapes
    coords = corner_peaks(corner_harris(gray))#, min_distance=5)
    coords_subpix = corner_subpix(gray, coords)#, window_size=13)

    return len(coords_subpix)

#
# colorfulness
#
@FeaturePlugin.register('colorfulness')
def colorfulness(provider):
    '''
    colorfulness (0,3.2)
    '''

    img = provider.img
    s = img.shape
    len = s[0]*s[1]

    r = reshape(img[...,0] , (len,)) / 255.0
    g = reshape(img[...,1] , (len,)) / 255.0
    b = reshape(img[...,2] , (len,)) / 255.0

    rg = r - g
    yb = 0.5 * ( r + g ) - b

    sd_rgyb = ( rg.std() ** 2 + yb.std() ** 2 ) ** 0.5
    m_rgyb  = ( rg.mean() **2 + yb.mean() ** 2 ) ** 0.5

    return sd_rgyb + 3.0 * m_rgyb


#
# entropy
#
@FeaturePlugin.register('entropy_v')
def entropy_v(provider):
    """
    entropy of the value channel in HSV space
    """
    hsv = provider.as_hsv()
    if 3 == hsv.ndim:
        return _entropy_v(hsv)
    else:
        return None


@FeaturePlugin.register('entropy_s')
def entropy_s(provider):
    """
    entropy of the saturation channel in HSV space
    """
    hsv = provider.as_hsv()
    if 3 == hsv.ndim:
        return _entropy_s(hsv)
    else:
        return None


@FeaturePlugin.register('entropy')
def entropy(provider):
    """
    entropy of the RGB channels
    """
    return _entropy_rgb(provider.img)

@FeaturePlugin.register('entropy_sv')
def entropy_sv(provider):
    """
    entropy of the saturation and value channels in HSV space
    """

    hsv = provider.as_hsv()
    if 3 == hsv.ndim:
        pixels = hsv[...,1:]
        pixels = reshape(pixels, (pixels.shape[0] * pixels.shape[1], 2))
        h,e = np.histogramdd(pixels, bins=(16,)*2, range=((0,1.0),)*2)
        prob = h/np.sum(h) # normalize
        prob = prob[prob>0] # remove zeros
        return -np.sum(prob*np.log2(prob))
    else:
        return None


@FeaturePlugin.register('entropy_ab')
def entropy_ab(provider):
    """
    entropy of the AB channels in the LAB space
    """

    lab = provider.as_lab()
    if 3 == lab.ndim:
        pixels = lab[...,1:] + 127
        pixels = reshape(pixels, (pixels.shape[0] * pixels.shape[1], 2))
        h,e = np.histogramdd(pixels, bins=(16,)*2, range=((0,255),)*2)
        prob = h/np.sum(h) # normalize
        prob = prob[prob>0] # remove zeros
        return -np.sum(prob*np.log2(prob))
    else:
        return None

#
# emotion
#
Emotion = namedtuple('Emotion', 'pleasure arousal dominance')

@FeaturePlugin.register('emotion', Emotion)
def emotion(provider):
    """
    pleasure, arousal, dominance
    """
    hsv = provider.as_hsv()

    sat = hsv[...,1].mean()
    val = hsv[...,2].mean()

    return (0.69 * val + 0.22 * sat, -0.31 * val + 0.60 * sat, -0.76 * val + 0.32 * sat)




#
# naturalness
#
Naturalness = namedtuple('Naturalness', 'naturalness skin grass sky')

@FeaturePlugin.register('naturalness', Naturalness)
def naturalness(provider):
    """
    naturalness, skin, grass, sky

    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.108.2839&rep=rep1&type=pdf
    """
    hsv = provider.as_hsv()

    hue = hsv[...,0]
    sat = hsv[...,1]
    val = hsv[...,2]

    # Thresholding L and S components: L values
    # between 20 and 80 are kept, S values over 0.1
    # are kept.
    mask = (val > (20.0 / 255)) & (val < (80. / 255)) & (sat > 0.1)


    skin_mask = mask & (hue > (25./360)) & (hue < (70. / 360))
    grass_mask = mask & (hue > (95./360)) & (hue < (135. / 360))
    sky_mask = mask & (hue > (185./360)) & (hue < (260. / 360))

    n_skin = np.sum(skin_mask)
    n_grass = np.sum(grass_mask)
    n_sky = np.sum(sky_mask)

    s_mean_skin = np.mean(sat[skin_mask])
    s_mean_grass = np.mean(sat[grass_mask])
    s_mean_sky = np.mean(sat[sky_mask])

    cni_skin = np.exp( -0.5 * (( s_mean_skin - 0.76 ) / 0.52) **2 )
    cni_grass = np.exp( -0.5 * (( s_mean_grass - 0.81 ) / 0.53) **2 )
    cni_sky = np.exp( -0.5 * (( s_mean_sky - 0.43 ) / 0.22) **2 )

    cni_image = 1. * ( n_skin * cni_skin + n_grass * cni_grass + n_sky * cni_sky) / ( n_skin + n_grass + n_sky)

    return (cni_image, cni_skin, cni_grass, cni_sky)


# entropy
def entropy(img):
    """

    entropy for a single channel

    http://en.wikipedia.org/wiki/Entropy_estimation#Histogram_estimator
    http://stackoverflow.com/questions/5524179/how-to-detect-motion-between-two-pil-images-wxpython-webcam-integration-exampl
    """

    # reshape to single dim array
    pixels = reshape(img, (img.shape[0] * img.shape[1], 1))

    # histogram with 16 bins
    h, _ = np.histogram(pixels, bins=16, range=(0.0,1.0))

    # normalize
    prob = 1.0 * h / np.sum(h)

    # remove zeroes
    prob = prob[prob > 0.0]

    # calc entropy
    return -np.sum(prob * np.log2(prob))

# entropy
def _entropy_s(hsvimg):
    """
    compute entropy values for the Saturation and Value components of a HVS image
    """

    # extract saturation
    sat = hsvimg[...,1]

    return entropy(sat)

# entropy
def _entropy_v(hsvimg):
    """
    compute entropy values for the Saturation and Value components of a HVS image
    """

    # extract value
    val = hsvimg[...,2]

    return entropy(val)


def _entropy_rgb(img):
    """
    http://en.wikipedia.org/wiki/Entropy_estimation#Histogram_estimator
    http://stackoverflow.com/questions/5524179/how-to-detect-motion-between-two-pil-images-wxpython-webcam-integration-exampl
    """
    if 3 == img.ndim:
        pixels = reshape(img, (img.shape[0] * img.shape[1], 3))
        h,e = np.histogramdd(pixels, bins=(16,)*3, range=((0,256),)*3)
        prob = h/np.sum(h) # normalize
        prob = prob[prob>0] # remove zeros
        return -np.sum(prob*np.log2(prob))
    else:
        return None
