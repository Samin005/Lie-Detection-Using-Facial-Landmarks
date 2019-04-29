from __future__ import division
import dlib
import cv2
import numpy as np
import math
import time
import datetime

win = dlib.image_window
count = 0
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
            #print("Before np: Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                          shape.part(1)))
            #win.add_overlay(shape)
            #print shape
            shape = shape_to_np(shape)
            # shape is an array from 0-67, which carries x,y co-ordiantes in a single index.
            #print shape[37]
            #yolo = shape[37]-shape[38]
            #time.sleep(2)

            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')





            # AU2 & AU4
            left_eye_top = [shape[37], shape[38], shape[39], shape[40]]
            left_eyebrow = [shape[19], shape[20], shape[21], shape[22]]

            right_eye_top = [shape[43], shape[44], shape[45], shape[46]]
            right_eyebrow = [shape[23], shape[24], shape[25], shape[26]]

            distanceL = [left_eye_top[0] - left_eyebrow[0], left_eye_top[1] - left_eyebrow[1],
                         left_eye_top[2] - left_eyebrow[2], left_eye_top[3] - left_eyebrow[3]]

            #AU2 = left_eye_top[2] - left_eyebrow[2]
            left_eye_top_39 = shape[38]
            left_eye_top_39_x = left_eye_top_39[0]
            left_eye_top_39_y = left_eye_top_39[1]
            #print "left_eye_top_39_x = {}, left_eye_top_39_y = {}".format(left_eye_top_39_x, left_eye_top_39_y)

            left_eyebrow_21 = shape[20]
            left_eyebrow_21_x = left_eyebrow_21[0]
            left_eyebrow_21_y = left_eyebrow_21[1]
            #print "left_eyebrow_21_x = {}, left_eyebrow_21_y = {}".format(left_eyebrow_21_x, left_eyebrow_21_y)

            dx = (left_eye_top_39_x-left_eyebrow_21_x)**2
            dy = (left_eye_top_39_y-left_eyebrow_21_y)**2
            #print "dx = {}, dy = {}".format(dx, dy)
            #print "Left_eye_top (shape[38]) = {}, Left_eybrow (shape[20]) = {}".format(shape[38], shape[20])
            AU2 = math.sqrt(dx+dy)




            # This is the actual left eye, previously it was mistaken for the right eye
            # AU5
            left_eyelid_top_45 = shape[45]
            left_eyelid_top_45_x = left_eyelid_top_45[0]
            left_eyelid_top_45_y = left_eyelid_top_45[1]

            left_eyelid_bottom_47 = shape[47]
            left_eyelid_bottom_47_x = left_eyelid_bottom_47[0]
            left_eyelid_bottom_47_y = left_eyelid_bottom_47[1]
            AU5 = math.sqrt((left_eyelid_top_45_x - left_eyelid_bottom_47_x) ** 2 + (
                left_eyelid_top_45_y - left_eyelid_bottom_47_y) ** 2)



            #AU9
            nose_left = shape[36]
            nose_left_y = nose_left[1]

            nose_right = shape[32]
            nose_right_y = nose_right[1]

            nose_tip = shape[34]
            nose_tip_y = nose_tip[1]
            #print "left = {}, right = {}, tip = {}".format(nose_left_y, nose_right_y, nose_tip_y)




            # AU12 & AU15
            lip_left = shape[55]
            lip_left_y = lip_left[1]

            lip_right = shape[49]
            lip_right_y = lip_right[1]

            lip_tip = shape[52]
            lip_tip_y = lip_tip[1]



            # AU53, AU54, AU55 & AU56
            head_left = shape[17]
            head_left_y = head_left[1]

            head_right = shape[1]
            head_right_y = head_right[1]

            head_tip = shape[34]
            head_tip_y = head_tip[1]


            if count == 0:
                #AU2_previous = AU2[1]
                AU2_previous = AU2

                AU5_previous = AU5

                nose_tip_y_previous = nose_tip_y
                nose_left_y_previous = nose_left_y
                nose_right_y_previous = nose_right_y

                lip_tip_y_previous = lip_tip_y
                lip_left_y_previous = lip_left_y
                lip_right_y_previous = lip_right_y

                head_tip_y_previous = head_tip_y
                head_left_y_previous = head_left_y
                head_right_y_previous = head_right_y
                #print "Previous Distance= {}".format(AU2_previous)
                #print "value updated"

            #print "Current Distance = {}".format(AU2)
            #print math.sqrt((5+5)**(2-2))
            if AU2-AU2_previous >= 1.25:
                print "AU2 Detected! {}".format(st)
            elif AU2_previous-AU2 >= 1.25:
                print "AU4 Detected! {}".format(st)
            elif AU5-AU5_previous >= 1:
                print "AU5 Detected! {}".format(st)
            elif nose_tip_y_previous == nose_tip_y and nose_left_y_previous > nose_left_y and nose_right_y_previous > nose_right_y:
                print "AU9 Detected! {}".format(st)
            elif lip_tip_y_previous == lip_tip_y and lip_left_y_previous > lip_left_y and lip_right_y_previous > lip_right_y:
                print "AU12 Detected! {}".format(st)
            elif lip_tip_y_previous == lip_tip_y and lip_left_y_previous < lip_left_y and lip_right_y_previous < lip_right_y:
                print "AU15 Detected! {}".format(st)
            elif head_tip_y_previous > head_tip_y and head_left_y_previous == head_left_y and head_right_y_previous == head_right_y:
                print "AU53 Detected! {}".format(st)
            elif head_tip_y_previous < head_tip_y and head_left_y_previous == head_left_y and head_right_y_previous == head_right_y:
                print "AU54 Detected! {}".format(st)
            elif head_tip_y_previous == head_tip_y and head_left_y_previous < head_left_y and head_right_y_previous > head_right_y:
                print "AU55 Detected! {}".format(st)
            elif head_tip_y_previous == head_tip_y and head_left_y_previous > head_left_y and head_right_y_previous < head_right_y:
                print "AU55 Detected! {}".format(st)
            if count == 1:
                count = -1
            count += 1
            #print "Left eye-eyebrow Distance: {}, {}, {}, {}".format(distanceL[0], distanceL[1], distanceL[2],
            #                                                         distanceL[3])
            '''
            print "Left eye top Distance: {}, {}, {}, {}".format(left_eye_top[0], left_eye_top[1], left_eye_top[2],
                                                                 left_eye_top[3])
            print "Left eyebrow Distance: {}, {}, {}, {}".format(left_eyebrow[0], left_eyebrow[1], left_eyebrow[2],
                                                                 left_eyebrow[3])

            print "     right eye top Distance: {}, {}, {}, {}".format(right_eye_top[0], right_eye_top[1],
                                                                       right_eye_top[2],
                                                                       right_eye_top[3])
            print "     right eyebrow Distance: {}, {}, {}, {}".format(right_eyebrow[0], right_eyebrow[1],
                                                                       right_eyebrow[2],
                                                                       right_eyebrow[3])
            '''
            #print("After np: Part 0: {}, Part 1: {} ...".format(shape.part(0),
            #                                                     shape.part(1)))

            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            for (x, y) in shape:

                cv2.circle(frame, (int(x/ratio), int(y/ratio)), 3, (255, 255, 255), -1)
            #cv2.rectangle(frame, (int(d.left()/ratio), int(d.top()/ratio)),(int(d.right()/ratio), int(d.bottom()/ratio)), (0, 255, 0), 1)

    cv2.imshow("image", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break