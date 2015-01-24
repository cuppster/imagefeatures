from collections import namedtuple
from scipy.misc import imsave
from imagefeatures import imageops

from imagefeatures.plugins import FeaturePlugin
from ..imageops import quantize
import numpy as np
from ..hash import gen_signed_hash, unsigned2signed64

_code_book = [

    # tone - vivid
    [231, 47, 30],
    [238, 113, 25],
    [255, 200, 8],
    [170, 198, 27],
    [19, 166, 50],
    [4, 148, 87],
    [1, 134, 141],
    [3, 86, 155],
    [46, 20, 141],
    [204, 63, 92],

    # tone - strong
    [207, 46, 49],
    [226, 132, 45],
    [227, 189, 28],
    [162, 179, 36],
    [18, 154, 47],
    [6, 134, 84],
    [3, 130, 122],
    [6, 133, 148],
    [92, 104, 163],
    [175, 92, 87],

    # tone - bright
    [231, 108, 86],
    [241, 176, 102],
    [255, 228, 15],
    [169, 199, 35],
    [88, 171, 45],
    [43, 151, 89],
    [0, 147, 159],
    [59, 130, 157],
    [178, 137, 166],
    [209, 100, 109],

    # tone - pale
    [233, 163, 144],
    [242, 178, 103],
    [255, 236, 79],
    [219, 220, 93],
    [155, 196, 113],
    [146, 198, 131],
    [126, 188, 209],
    [147, 184, 213],
    [197, 188, 213],
    [218, 176, 176],

    # tone - very pale
    [236, 217, 202],
    [245, 223, 181],
    [249, 239, 189],
    [228, 235, 191],
    [221, 232, 207],
    [209, 234, 221],
    [194, 222, 242],
    [203, 215, 232],
    [224, 218, 230],
    [235, 219, 224],

    # tone - light grayish
    [213, 182, 166],
    [218, 196, 148],
    [233, 227, 143],
    [209, 116, 73],
    [179, 202, 157],
    [166, 201, 163],
    [127, 175, 166],
    [165, 184, 199],
    [184, 190, 189],
    [206, 185, 179],

    # tone - light
    [211, 142, 110],
    [215, 145, 96],
    [255, 203, 88],
    [195, 202, 101],
    [141, 188, 90],
    [140, 195, 110],
    [117, 173, 169],
    [138, 166, 187],
    [170, 165, 199],
    [205, 154, 149],

    # tone - grayish
    [171, 131, 115],
    [158, 128, 110],
    [148, 133, 105],
    [144, 135, 96],
    [143, 162, 121],
    [122, 165, 123],
    [130, 154, 145],
    [133, 154, 153],
    [151, 150, 139],
    [160, 147, 131],

    # tone - dull
    [162, 88, 61],
    [167, 100, 67],
    [139, 117, 65],
    [109, 116, 73],
    [88, 126, 61],
    [39, 122, 62],
    [24, 89, 63],
    [53, 109, 98],
    [44, 77, 143],
    [115, 71, 79],

    # tone - deep

    [172, 36, 48],
    [169, 87, 49],
    [156, 137, 37],
    [91, 132, 47],
    [20, 114, 48],
    [23, 106, 43],
    [20, 88, 60],
    [8, 87, 107],
    [58, 55, 119],
    [111, 61, 56],

    # tone - dark
    [116, 47, 50],
    [115, 63, 44],
    [103, 91, 44],
    [54, 88, 48],
    [30, 98, 50],
    [27, 86, 49],
    [18, 83, 65],
    [16, 76, 84],
    [40, 57, 103],
    [88, 60, 50],

    # tone - dark grayish
    [79, 46, 43],
    [85, 55, 43],
    [75, 63, 45],
    [44, 60, 49],
    [34, 62, 51],
    [31, 56, 45],
    [29, 60, 47],
    [25, 62, 63],
    [34, 54, 68],
    [53, 52, 48],

    ## neutrals
    #[244, 244, 244],
    #[236, 236, 236],
    #[206, 206, 206],
    #[180, 180, 180],
    #[152, 152, 152],
    #[126, 126, 126],
    #[86, 86, 86],
    #[60, 60, 60],
    #[38, 38, 38],
    #[10, 10, 10]
]


code_book = np.array(_code_book)

koba_hues = ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB' , 'P', 'RP']
koba_tone = ['V', 'S', 'B', 'P', 'Vp', 'Lgr', 'L', 'Gr', 'Dl', 'Dp', 'Dk', 'Dgr']


#
# kobayashi palette hash
#

@FeaturePlugin.register('kyp3o')
def kyp3o(provider):

    """
    quantize and return the colors sorted by most frequent

    """
    result, h = imageops.quantize(provider.img, code_book, size=(100,100), density=False )

    report = []
    for i, num in enumerate(h):
        report.append((i, num))

    s = sorted(report, key=lambda x: x[1], reverse=True)

    color_hash = ''

    for index, num in s[:3]:

        row = int(index / len(koba_hues)) # tone
        col = int(index % len(koba_hues)) # hue

        kcolor = "{}/{}".format(koba_hues[col], koba_tone[row])

        color_hash += kcolor

    # only create a palette from 2 or more colors
    if 3 <= len(s):
        hash = gen_signed_hash(color_hash)
        return unsigned2signed64(hash)
    else:
        return None

@FeaturePlugin.register('kyp3') #, signature_id=2)
def kyp3(provider):

    """
    quantize and return the colors sorted by most frequent
    """
    result, h = imageops.quantize(provider.img, code_book, size=(100,100), density=False )

    report = []
    for i, num in enumerate(h):
        report.append((i, num))

    s = sorted(report, key=lambda x: x[1], reverse=True)

    two_color_hash = 0

    for index, num in s[:3]:

        row = int(index / len(koba_hues)) # tone
        col = int(index % len(koba_hues)) # hue

        kcolor = "{}/{}".format(koba_hues[col], koba_tone[row])

        hash = gen_signed_hash(kcolor)
        two_color_hash = two_color_hash ^ hash

    # only create a palette from 2 or more colors
    if 3 <= len(s):
        return unsigned2signed64(two_color_hash)
    else:
        return 0

@FeaturePlugin.register('kyp2')
def kyp2(provider):

    """
    quantize and return the colors sorted by most frequent

    """
    result, h = imageops.quantize(provider.img, code_book, size=(100,100), density=False )

    report = []
    for i, num in enumerate(h):
        report.append((i, num))

    s = sorted(report, key=lambda x: x[1], reverse=True)

    two_color_hash = 0

    for index, num in s[:2]:

        row = int(index / len(koba_hues)) # tone
        col = int(index % len(koba_hues)) # hue

        kcolor = "{}/{}".format(koba_hues[col], koba_tone[row])

        hash = gen_signed_hash(kcolor)
        two_color_hash = two_color_hash ^ hash

    # only create a palette from 2 or more colors
    if 2 <= len(s):
        return unsigned2signed64(two_color_hash)
    else:
        return 0

#
# kobayashi colors
#
KobaType = namedtuple('Koba', 'ky0h ky0t ky1h ky1t')

@FeaturePlugin.register('koba', KobaType)
def koba(provider):
        """
        quantize and return the colors sorted by most frequent

        """
        # setup tolerance
#        if tolerance is None:
#            tolerance = 0.25


        result, h = imageops.quantize(provider.img, code_book, size=(100,100), density=False )

        #print "koba", result, h

        report = []
        for i, num in enumerate(h):
            report.append((i, num))

        s = sorted(report, key=lambda x: x[1], reverse=True)

        ret = []

        #two_color_hash = 0

        for index, num in s[:2]:

            row = int(index / len(koba_hues)) # tone
            col = int(index % len(koba_hues)) # hue

            #kcolor = "{}/{}".format(koba_hues[col], koba_tone[row])

            #print i, num, row, col, kcolor

            ret.append(col)
            ret.append(row)

            #hash = gen_signed_hash(kcolor)
            #two_color_hash = two_color_hash ^ hash

        ## only create a palette from 2 or more colors
        #if 2 <= len(s):
        #    ret.append( unsigned2signed64( two_color_hash))
        #else:
        #    ret.append(0)

        #imsave('/home/jason/Desktop/out.png', result)
        #return [ x[0] for x in top[:2]]

        return ret