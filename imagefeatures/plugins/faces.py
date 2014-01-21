import os
import cv2
from imagefeatures.plugins import FeaturePlugin


class FaceDetect:
    """
    The default parameters ( scale_factor =1.1, min_neighbors =3, flags =0) are tuned for accurate yet slow object detection.

    For a faster operation on real video images the settings are: scale_factor =1.2, min_neighbors =2,
    flags = CV_HAAR_DO_CANNY_PRUNING , min_size = minimum possible face size
    (for example,  1/4 to 1/16 of the image area in the case of video conferencing).

    HaarDetectObjects(image, cascade, storage, scaleFactor=1.1, minNeighbors=3, flags=0, minSize=(0, 0))
    """

    _cascade = None
  
    def __init__(self):
        pass


    @staticmethod
    def detect_face( provider):

        if FaceDetect._cascade is None:
            haar = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            FaceDetect._cascade = cv2.CascadeClassifier(haar)

        img = provider.img
          
        # http://fideloper.com/facial-detection
        faces = FaceDetect._cascade.detectMultiScale(img, 1.2, 2, cv2.cv.CV_HAAR_SCALE_IMAGE)

        #if outfile is not None:
        #    if faces:
        #        for (x,y,w,h),n in faces:
        #            cv.Rectangle(img, (x,y), (x+w,y+h), 255)
        #        cv.SaveImage(outfile, img)

        # return the number of faces
        return len(faces)


@FeaturePlugin.register('faces')
def face(provider):
    """
    detect faces
    """
    face_detect = FaceDetect()
    return face_detect.detect_face(provider)
