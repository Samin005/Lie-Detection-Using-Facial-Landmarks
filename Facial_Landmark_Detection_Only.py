from __future__ import division
import dlib
import cv2
import numpy as np
import time

# win = dlib.image_window
# count = 0


def resize(img, width=None, height=None, interpolation=cv2.INTER_AREA):
    global ratio
    w, h = img.shape

    if width is None and height is None:
        return img
    elif width is None:
        ratio = height / h
        width = int(w * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized
    else:
        ratio = width / w
        height = int(h * ratio)
        resized = cv2.resize(img, (height, width), interpolation)
        return resized


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords

# if you have an external webcam use 1 instead of 0


camera = cv2.VideoCapture(0)

predictor_path = 'shape_predictor_68_face_landmarks.dat'

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

while True:

    ret, frame = camera.read()
    if ret == False:
        print('Failed to capture frame from camera. Check camera index in cv2.VideoCapture(0) \n')
        break

    frame_grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_resized = resize(frame_grey, width=120)

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    dets = detector(frame_resized, 1)
    if len(dets) > 0:
        for k, d in enumerate(dets):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy array
            shape = predictor(frame_resized, d)
            # print("Before np: Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                          shape.part(1)))
            # win.add_overlay(shape)
            shape = shape_to_np(shape)
            # shape is an array from 0-67, which carries x,y co-ordiantes in a single index.
            print shape
            print "\n"
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:
                cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
            # to add square around face
            # cv2.rectangle(frame, (int(d.left() / ratio), int(d.top() / ratio)),
            #               (int(d.right() / ratio), int(d.bottom() / ratio)), (0, 255, 0), 1)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
