from PIL import Image
from skimage.color import rgb2hsv as skimage_rgb2hsv
from skimage.color import rgb2lab as skimage_rgb2lab
from skimage.color import rgb2gray as skimage_rgb2gray

from plugins import FeaturePlugin

class FeaturePatch:
    All, Portrait, BottomHalf, TopHalf, PortraitBackground, PortraitForeground, \
    BottomThird, TopThird \
        = range(8)

class ImageProvider:
    """
    obtain various features of images
    uses caching to avoid extra conversion
    operations
    """

    def __init__(self, src, patch=None, lid=None):

        import numpy as np
        import scipy.ndimage as nd

        self.gray = None
        self.hsv = None
        self.lab = None

        self.colorhasher = None
        self.source = None
        self.pil = None
        self.lid = lid

        self._patches = dict()
        """:type : dict[int, numpy.array]"""

        # file source
        if isinstance(src, basestring):
            self.img = nd.imread(src)
            self.source = src

        # stream source
        elif hasattr(src, 'read'):
            self.img = np.asarray(Image.open(src))
            src.close()

        # numpy array source
        else:
            self.img = src

    def patch(self, patch):
        """
        return a portion of the image

        @type patch: int
        """

        if patch in self._patches:
            return self._patches[patch]

        # NOTE: only works on RGB images, 3 == len(self.img.shape)
        ly, lx, _ = self.img.shape

        if FeaturePatch.Portrait == patch:
            self._patches[patch] = self.img[ly/3:-ly/3, lx/3:-lx/3, ...]
            return self._patches[patch]

        elif FeaturePatch.BottomHalf == patch:
            self._patches[patch] = self.img[ly/2:ly, 0:lx, ...]
            return self._patches[patch]

        elif FeaturePatch.BottomThird == patch:
            self._patches[patch] = self.img[-ly/3:ly, 0:lx, ...]
            return self._patches[patch]

        elif FeaturePatch.TopThird == patch:
            self._patches[patch] = self.img[0:ly/3, 0:lx, ...]
            return self._patches[patch]

        else:
            raise Exception("Patch {} not implemented!".format(patch))

    def extract(self, feature):

        func = FeaturePlugin.get(feature)
        if func is not None:
            return func(self)
        else:
            return None

    def as_lab(self):

        if self.lab is None:
            self.lab = skimage_rgb2lab(self.img)
        return self.lab

    def as_hsv(self):

        if self.hsv is None:
            self.hsv = skimage_rgb2hsv(self.img)
        return self.hsv

    def as_pil(self):

        if self.pil is None:
            self.pil = Image.fromarray(self.img)
        return self.pil

    def as_gray(self):
        """
        range of gray channel is 0.0 - 1.0
        """

        if self.gray is None:
            self.gray = skimage_rgb2gray(self.img)
        return self.gray