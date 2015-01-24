import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import pkg_resources
import unittest
import sys

from nose.tools import assert_is_not_none

from imagefeatures import ImageProvider

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RawFeatureExtractTests(unittest.TestCase):

    def setUp(self):
        im = pkg_resources.resource_stream('tests', 'images/mona.jpg')
        self.provider = ImageProvider(im)

    def test_whratio(self):

        result = self.provider.extract('whratio')
        logger.debug("whratio {}".format(result))
        assert_is_not_none(result)

    def test_faces(self):

        result = self.provider.extract('faces')
        logger.debug("faces {0}".format(result))
        assert_is_not_none(result)

    def test_sobelgm(self):

        result = self.provider.extract('sobelgm')
        logger.debug("sobelgm {}".format(result))
        assert_is_not_none(result)

    def test_sobelh(self):

        result = self.provider.extract('sobelh')
        logger.debug("sobelh {}".format(result))
        assert_is_not_none(result)

    def test_sobelv(self):

        result = self.provider.extract('sobelv')
        logger.debug("sobelv {}".format(result))
        assert_is_not_none(result)

    def test_sobeld(self):

        result = self.provider.extract('sobeld')
        logger.debug("sobeld {}".format(result))
        assert_is_not_none(result)

    def test_corners(self):

        result = self.provider.extract('corners')
        logger.debug("corners {}".format(result))
        assert_is_not_none(result)

    def test_colorfulness(self):

        result = self.provider.extract('colorfulness')
        logger.debug("colorfulness {}".format(result))
        assert_is_not_none(result)

    def test_entropy(self):

        result = self.provider.extract('entropy')
        logger.debug("entropy {}".format(result))
        assert_is_not_none(result)

    def test_entropy_s(self):

        result = self.provider.extract('entropy_s')
        logger.debug("entropy_s {}".format(result))
        assert_is_not_none(result)

    def test_entropy_v(self):

        result = self.provider.extract('entropy_v')
        logger.debug("entropy_v {}".format(result))
        assert_is_not_none(result)

    def test_entropy_sv(self):

        result = self.provider.extract('entropy_sv')
        logger.debug("entropy_sv {}".format(result))
        assert_is_not_none(result)

    def test_entropy_ab(self):

        result = self.provider.extract('entropy_ab')
        logger.debug("entropy_ab {}".format(result))
        assert_is_not_none(result)

    def test_emotion(self):

        result = self.provider.extract('emotion')
        logger.debug("emotion {}".format(result))
        assert_is_not_none(result)

    def test_naturalness(self):

        result = self.provider.extract('naturalness')
        logger.debug("naturalness {}".format(result))
        assert_is_not_none(result)


if __name__ == '__main__':
    unittest.main()
