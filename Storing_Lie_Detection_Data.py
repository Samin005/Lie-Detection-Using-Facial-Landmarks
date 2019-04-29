from __future__ import division
import dlib
import cv2
import numpy as np
import os.path


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


print "1. Record Lie Data"
print "2. Record Truth Data"
print "3. Record Test Data"
print "4. Test Facial Landmark Detection in Real-time"

x = input("Please enter value: ")
dataFile = 'Data_for_Lie_Detection.csv'
if x == 1:
    print "You are about to record Lie Data."
    if not os.path.exists(dataFile):
        excelFile = open(dataFile, 'w')
        header = ""
        for x in range(0, 69):
            if x == 68:
                header += "is_Truth\n"
            else:
                header += `(x + 1)` + "_x, " + `(x + 1)` + "_y, "
        excelFile.write(header)
        excelFile.close()
    raw_input("Press enter to continue...\n")
    print "Recording Lie Data, press 'q' while recording to quit."
    print "Press 'q' to quit."
    excelFile = open(dataFile, 'a')
    # if you have an external webcam use 1 instead of 0
    camera = cv2.VideoCapture(1)
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
        dets = detector(frame_resized, 1)
        if len(dets) > 0:
            for k, d in enumerate(dets):
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)
                isTruth = 0
                shapeString = ""
                for x in range(0, 69):
                    if x == 68:
                        shapeString += `isTruth`+"\n"
                    else:
                        tempArr = shape[x]
                        shapeString += `tempArr[0]` + ", " + `tempArr[1]` + ", "
                excelFile.write(shapeString)
                for (x, y) in shape:
                    cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break

elif x == 2:
    print "You are about to record Truth Data."
    if not os.path.exists(dataFile):
        excelFile = open(dataFile, 'w')
        header = ""
        for x in range(0, 69):
            if x == 68:
                header += "is_Truth\n"
            else:
                header += `(x + 1)` + "_x, " + `(x + 1)` + "_y, "
        excelFile.write(header)
        excelFile.close()
    raw_input("Press enter to continue...\n")
    print "Recording Truth Data, press 'q' while recording to quit."
    excelFile = open(dataFile, 'a')
    # if you have an external webcam use 1 instead of 0
    camera = cv2.VideoCapture(1)
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
        dets = detector(frame_resized, 1)
        if len(dets) > 0:
            for k, d in enumerate(dets):
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)
                isTruth = 1
                shapeString = ""
                for x in range(0, 69):
                    if x == 68:
                        shapeString += `isTruth` + "\n"
                    else:
                        tempArr = shape[x]
                        shapeString += `tempArr[0]` + ", " + `tempArr[1]` + ", "
                excelFile.write(shapeString)
                for (x, y) in shape:
                    cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break


elif x == 3:
    print "You are about to record Test Data."
    TestFile = 'Test_Data_for_Lie_Detection.csv'
    if not os.path.exists(TestFile):
        excelFile = open(TestFile, 'w')
        header = ""
        for x in range(0, 69):
            if x == 68:
                header += "\n"
            elif x == 67:
                header += `(x + 1)` + "_x, " + `(x + 1)` + "_y"
            else:
                header += `(x + 1)` + "_x, " + `(x + 1)` + "_y, "
        excelFile.write(header)
        excelFile.close()
    raw_input("Press enter to continue...\n")
    print "Recording Truth Data, press 'q' while recording to quit."
    excelFile = open(TestFile, 'a')
    # if you have an external webcam use 1 instead of 0
    camera = cv2.VideoCapture(1)
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
        dets = detector(frame_resized, 1)
        if len(dets) > 0:
            for k, d in enumerate(dets):
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)
                shapeString = ""
                for x in range(0, 69):
                    if x == 68:
                        shapeString += "\n"
                    elif x == 67:
                        tempArr = shape[x]
                        shapeString += `tempArr[0]` + ", " + `tempArr[1]`
                    else:
                        tempArr = shape[x]
                        shapeString += `tempArr[0]` + ", " + `tempArr[1]` + ", "
                excelFile.write(shapeString)
                for (x, y) in shape:
                    cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break


elif x == 4:
    print "\nTesting Facial Landmark Detection in Real-time...\n"
    # if you have an external webcam use 1 instead of 0
    camera = cv2.VideoCapture(1)
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
        dets = detector(frame_resized, 1)
        if len(dets) > 0:
            for k, d in enumerate(dets):
                shape = predictor(frame_resized, d)
                shape = shape_to_np(shape)
                isTruth = "is_Truth"
                shapeString = ""
                for x in range(0, 69):
                    if x == 68:
                        shapeString += isTruth + "\n"
                    else:
                        tempArr = shape[x]
                        shapeString += `tempArr[0]` + ", " + `tempArr[1]` + ", "
                print shapeString
                for (x, y) in shape:
                    cv2.circle(frame, (int(x / ratio), int(y / ratio)), 3, (255, 255, 255), -1)
        cv2.imshow("image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break
else:
    print "Please give a valid input."
