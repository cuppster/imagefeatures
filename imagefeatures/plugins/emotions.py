from collections import namedtuple
import numpy as np
from numpy.core.fromnumeric import reshape
#from skimage.color.colorconv import rgb2lab
from scipy.misc import imresize
from skimage.color.colorconv import rgb2lab
from skimage.filter import sobel, hsobel, vsobel
from skimage.feature import corner_harris, corner_subpix, corner_peaks
from skimage.segmentation import quickshift
from plugins import FeaturePlugin
import numpy as np
import numpy.ma as ma

import math
from imageops import segment


def _out_harmony(lab1, lab2):
    """
    @type lab1: tuple[float, float, float]
    @type lab2: tuple[float, float, float]
    """

    l1, a1, b1 = lab1
    l2, a2, b2 = lab2

    h1 = math.atan(b1 / a1)
    h2 = math.atan(b2 / a2)

    C1 = math.sqrt(a1**2 + b1**2)
    C2 = math.sqrt(a2**2 + b2**2)

    Cab_diff = abs(C2 - C1)
    h_diff = abs(h2 - h1)


    #

    delta_C = math.sqrt(h_diff ** 2 + ( Cab_diff / 1.46 ) ** 2)

    # chromatic effect

    Hc = 0.04 + 0.53 * math.tanh(0.8 - 0.045 * delta_C)

    # lightness effect

    L_sum = l1 + l2
    L_delta = abs(l2 - l1)
    H_delta_L = 0.14 + 0.15 * math.tanh(-2 + 0.2 * L_delta)
    H_Lsum = 0.28 * 0.54 * math.tanh(-3.88 + 0.029 * L_sum)

    Hl = H_Lsum + H_delta_L

    # hue effect

    Hsy1 = _ou_hue_effect((l1, a1, b1))
    Hsy2 = _ou_hue_effect((l2, a2, b2))

    Hh = Hsy1 + Hsy2

    # LAST ONE

    return Hc + Hl + Hh




def _ou_hue_effect(lab):

    l, a, b = lab

    hab = math.atan(b / a)

    C = math.sqrt(a**2 + b**2)

    Ey = ((0.22 * l - 12.8) / 10.0) \
        * math.exp( \
            ((90 - hab) / 10.0 ) \
            - math.exp((90.0 - hab) / 10.0 ))

    Hs = -0.08 - 0.14 * math.sin(hab + 50.0) - 0.07 * math.sin(2 * hab + 90.0)

    Ec = 0.5 + 0.5 * math.tanh(-2 + 0.5 * C)

    Hsy = Ec * ( Hs + Ey)

    return Hsy


def _ou_emotion(lab):
    """
    @type lab: tuple[float, float, float]
    """

    l, a, b = lab

    h = math.atan(b / a)

    C = math.sqrt(a**2 + b**2)

    activity = -2.1 + 0.06 * \
        math.sqrt( \
            (l - 50) ** 2 \
            + (a - 3) ** 2 \
            + ((b - 17) / 1.4) ** 2 \
        )

    weight = -1.8 + 0.04 * (100 - l) + 0.45 * math.cos(h - 100)

    heat = -.05 + 0.02 * (C**1.07) * math.cos(h - 50)

    return activity, weight, heat


@FeaturePlugin.register('ou_harmony')
def ou_harmony(provider):
    """
    @type provider imagefeatures.ImageProvider
    """

    img = provider.img
    if 3 != img.ndim:
        return None

    # resize
    length = 128
    img = imresize(img, (length, length))

    # get segment labels
    img_lab = rgb2lab(img)
    segs = segment(img_lab)

    #print img_lab
    #print segs

    # get bin counts and sort them
    bincounts = [ (i, c) for i, c in enumerate(np.bincount(segs.flat)) ]
    #print "bincounts", bincounts

    sortedbincounts = sorted(bincounts, key=lambda x: x[1], reverse=True)
    #print "sortedbincounts", sortedbincounts

    L = img_lab[...,0]
    A = img_lab[...,1]
    B = img_lab[...,2]

    #print "L channel", img_lab[...,0]
    #print "A channel", img_lab[...,1]
    #print "B channel", img_lab[...,2]

    # get the average color of the three largest segments
    colors = []
    for i, c in sortedbincounts[:2]:

        #print "segment (count, label)", c, i

        # mask (set to True) where the label is NOT the current index (i)
        mask = segs != i
        #print "mask", mask

        # mean of lab values
        L_mask = ma.masked_array(L, mask=mask)
        A_mask = ma.masked_array(A, mask=mask)
        B_mask = ma.masked_array(B, mask=mask)

        #print "L mask", L_mask
        #print "A mask", A_mask
        #print "B mask", B_mask

        L_mean = L_mask.mean()
        A_mean = A_mask.mean()
        B_mean = B_mask.mean()

        colors.append( (L_mean, A_mean, B_mean))

        #print "L mean = ", L_mask.mean()
        #print "A mean = ", A_mask.mean()
        #print "B mean = ", B_mask.mean()

    if 2 == len(colors):

        h = _out_harmony(colors[0], colors[1])

        return h


    else:

        return None

    #img_lab = provider.as_lab()

    #print "img_lab", img_lab
    #lab = ( np.mean(img_lab[...,0]), np.mean(img_lab[...,1]), np.mean(img_lab[...,2]))



    return 0; # _out_harmony()


OuEmotionType = namedtuple('OuEmotion', 'ou_activity, ou_weight, ou_heat')
@FeaturePlugin.register('ou_emotion', cons=OuEmotionType)
def ou_emotion(provider):
    """
    @type provider imagefeatures.ImageProvider
    """

    img = provider.img
    if 3 != img.ndim:
        return None

    img_lab = provider.as_lab()

    #print "img_lab", img_lab

    lab = ( np.mean(img_lab[...,0]), np.mean(img_lab[...,1]), np.mean(img_lab[...,2]))

    return _ou_emotion(lab)
