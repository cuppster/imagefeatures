import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
import requests

import pkg_resources
import unittest
import sys

from nose.tools import assert_is_not_none



import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RestFeatureExtractTests(unittest.TestCase):

    def setUp(self):
       pass

    def test_image1(self):
        r = requests.post("http://127.0.0.1:8099/features"



if __name__ == '__main__':
    unittest.main()
