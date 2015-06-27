


## Installing opencv on a Mac

http://www.learnopencv.com/install-opencv-3-on-yosemite-osx-10-10-x/

```bash
brew tap homebrew/science
brew install opencv
```

### Set up Python by creating a couple of symlinks.

```bash
cd /Library/Python/2.7/site-packages/
ln -s /usr/local/Cellar/opencv/2.4.9/lib/python2.7/site-packages/cv.py cv.py
ln -s /usr/local/Cellar/opencv/2.4.9/lib/python2.7/site-packages/cv2.so cv2.so
```

/Users/jcupp/Projects/leo/imagefeatures/venv/lib/python2.7/site-packages
(venv)theros:site-packages jcupp$ ln -s /usr/local/Cellar/open
opencv/  openexr/ openssl/
(venv)theros:site-packages jcupp$ ln -s /usr/local/Cellar/opencv/2.4.1
2.4.10.1/ 2.4.11_1/
(venv)theros:site-packages jcupp$ ln -s /usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages/cv.py cv.py
(venv)theros:site-packages jcupp$ ln -s /usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages/cv
cv.py   cv2.so  
(venv)theros:site-packages jcupp$ ln -s /usr/local/Cellar/opencv/2.4.11_1/lib/python2.7/site-packages/cv2.so cv2.so
