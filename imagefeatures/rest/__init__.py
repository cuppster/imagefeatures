# -*- coding: utf-8 -*-
"""
Created on Thu Nov 29 11:16:17 2012

@author: jason
"""

import sys, os, io
from flask import Flask, make_response, render_template, request, g, Response
import flask
import sys

import time
import cStringIO
import random
import numpy as np
from bson.json_util import dumps

import logging
logger = logging.getLogger(__name__)


# logging...
FORMAT = '[%(levelname)s - %(asctime)s] %(message)s'
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# features
from imagefeatures.plugins.basic import Emotion, Naturalness
from imagefeatures.plugins.emotions import OuEmotionType
from imagefeatures.plugins.kobayashi import KobaType
import imagefeatures.plugins.faces

ALL_FEATURES = ['entropy', 'entropy_v', 'entropy_s',
                'corners', 'colorfulness',
                'faces', 'koba',
                'emotion', 'naturalness',
                'sobelgm', 'sobeld', 'sobelv', 'sobelh',
                'ou_emotion', 'ou_harmony'
                ]


SIGNATURES = ['kyp3']

ALL_FEATURES_AND_SIGNATURES = ALL_FEATURES + SIGNATURES


INDEX_FEATURES = []
INDEX_FEATURES.extend(ALL_FEATURES)
INDEX_FEATURES.extend(SIGNATURES)

# extend with complex types
INDEX_FEATURES.extend(Naturalness._fields)
INDEX_FEATURES.extend(Emotion._fields)
INDEX_FEATURES.extend(KobaType._fields)
INDEX_FEATURES.extend(OuEmotionType._fields)



# create flask app
app = Flask(__name__, instance_relative_config=True, static_folder='static', static_url_path='/static')

# configure app
app.config.from_pyfile('application.cfg', silent=True)
app.debug = app.config['DEBUG']

def extract_from_image(provider, features, fail_on_error=True, return_provider=False):

    # extract features

    for feature in features:

        #print "TRYING", feature

        try:

            # extract values
            result = provider.extract(feature)

            #print "result: ", feature, result

            #if result is None:
            #    yield lid, feature, None
            #    continue
            #if result is None:
            #    print "raising!"
            #    raise Exception("Unable to extract feature: {}".format(feature))

        except Exception as ex:

            if fail_on_error:
                raise ex
            else:
                logger.warn(ex)
                continue

        if result is None:
            yield feature, None
            continue

        if hasattr(result, "_asdict"):
            result = result._asdict().iteritems()
        else:
            result = dict({feature:result}).iteritems()

        for f, value in result:

            # convert from numpy dtype to python build-in type
            if type(value) is int or type(value) is float or type(value) is str or type(value) is long:
                conv = value
            else:
                conv = np.asscalar(value)

            # provide values to client
            if return_provider:
                yield f, conv, provider
            else:
                yield f, conv

@app.route('/features', methods=['GET'])
def status():

    return make_response("OK", 200)


@app.route('/features', methods=['POST'])
def extract_features():

    import requests
    from StringIO import StringIO
    from imagefeatures import ImageProvider
    # from leo.match import SIGMatcher
    # from leo.images.extract import extract_from_image
    # from leo.features import ALL_FEATURES_AND_SIGNATURES

    # obj = request.get_json()
    # if 'url' not in obj:
    #     raise "Expected url"
    # url = obj['url']

    # start timer
    start_time = time.time()

    # logger.debug('fetching from %s' % url)
    # r = requests.get(url)

    # get buffer/stream
    data_stream = StringIO(request.data)

    # print data_stream

    # create provider
    p = ImageProvider(data_stream)

    # get signature
    sig = p.extract('signature')
    whratio = p.extract('whratio')

    # look up title and artist name
    # matcher = SIGMatcher()
    title, artist = "", "" # matcher.match(p)

    # if title is not None or artist is not None:
    # logger.debug("TITLE, ARTIST : '%s', '%s'", title, artist)

    # return None

    # extract features
    features = dict([ (f,v) for f, v in extract_from_image(p, ALL_FEATURES_AND_SIGNATURES) ])


    features['sig'] = sig
    features['whratio'] = whratio

    # logger.debug('found raw features %s' % features)

    elapsed_time = (time.time() - start_time)

    return make_response((dumps(features), 200, { 'x-leonardo-api-time': str(elapsed_time), 'Content-Type': 'application/json'}))