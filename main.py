import os
import shutil
import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing.image import load_img, img_to_array

# load model # Accuracy=99.3 , validation Accuracy = 99.3 # heavy model, size =226MB
model = load_model('trainedmodel.h5')

print(" Enter a option ")
print(" 1. Detect Mask using image")
print(" 2. Detect Mask using live video")
a=input("Press 1 / 2 : ")
if a==2:
    # model accept below hight and width of the image
    img_width, img_hight = 224, 224
    direc="faces"
    a=0o777
    # model accept below hight and width of the image
    img_width, img_hight = 200, 200

    # ......................................
    # Load the Cascade face Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # startt  web cam
    cap = cv2.VideoCapture(0)  # for webcam
    # cap = cv2.VideoCapture('videos/Mask - 34775.mp4') # for video

    img_count_full = 0

    # parameters for text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (1, 1)
    class_lable = ' '
    # fontScale
    fontScale = 1  # 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2  # 1

    # sart reading images and prediction
    while True:
        os.mkdir(direc,a)
        img_count_full += 1

        # read image from webcam
        responce, color_img = cap.read()
        # color_img = cv2.imread('sandeep.jpg')

        # if respoce False the break the loop
        if responce == False:
            break

            # Convert to grayscale
        gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

        # Detect the faces
        faces = face_cascade.detectMultiScale(gray_img, 1.2, 3)  # 1.1, 3) for 1.mp4

        # take face then predict class mask or not mask then draw recrangle and text then display image
        img_count = 0
        for (x, y, w, h) in faces:
            org = (x - 10, y - 10)
            img_count += 1
            color_face = color_img[y:y + h, x:x + w]  # color face
            cv2.imwrite('faces/%d%dface.jpg' % (img_count_full, img_count), color_face)
            img = load_img('faces/%d%dface.jpg' % (img_count_full, img_count), target_size=(img_width, img_hight))

            img = img_to_array(img) / 255
            img = np.expand_dims(img, axis=0)
            pred_prob = model.predict(img)
            # print(pred_prob[0][0].round(2))
            pred = np.argmax(pred_prob)

            if pred == 0:
                print("User with mask - predic = ", pred_prob[0][0])
                class_lable = "Mask"
                color = (0, 255, 0)

            else:
                print('user not wearing mask - prob = ', pred_prob[0][1])
                class_lable = "No Mask"
                color = (0,0, 255)

            cv2.rectangle(color_img, (x, y), (x + w, y + h), color, 3)
            # Using cv2.putText() method
            cv2.putText(color_img, class_lable, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

            # display image
        cv2.imshow('LIVE face mask detection', color_img)
        shutil.rmtree(direc)

        if cv2.waitKey(1) & 0xFF == ord('q'):

            break


    # Release the VideoCapture object
    cap.release()
    cv2.destroyAllWindows()
else :
    # model accept below hight and width of the image
    img_width = 200
    img_hight = 200
    # Load the Cascade face Classifier
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # parameters for text
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX
    # org
    org = (1, 1)
    class_lable = ' '
    # fontScale
    fontScale = 1  # 0.5
    # Blue color in BGR
    color = (255, 0, 0)
    # Line thickness of 2 px
    thickness = 2  # 1

    # read image from webcam
    color_img = cv2.imread('B612_20170324_142650.jpg')

    # Convert to grayscale
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # Detect the faces
    faces = face_cascade.detectMultiScale(gray_img,
                                          scaleFactor=1.2,
                                          minNeighbors=5,
                                          flags=cv2.CASCADE_SCALE_IMAGE)

    # take face then predict class mask or not mask then draw recrangle and text then display image
    img_count = 0
    for (x, y, w, h) in faces:
        org = (x - 10, y - 10)
        img_count += 1
        color_face = color_img[y:y + h, x:x + w]  # color face
        cv2.imwrite('imgfaces/%dface.jpg' % (img_count), color_face)
        img = load_img('imgfaces/%dface.jpg' % (img_count), target_size=(img_width, img_hight))

        img = img_to_array(img) / 255
        img = np.expand_dims(img, axis=0)
        pred_prob = model.predict(img)
        # print(pred_prob[0][0].round(2))
        pred = np.argmax(pred_prob)

        if pred == 0:
            print("User with mask - predic = ", pred_prob[0][0])
            class_lable = "Mask"
            color = (255, 0, 0)
            cv2.imwrite('faces/with_mask/%dface.jpg' % (img_count), color_face)
            cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Using cv2.putText() method
            cv2.putText(color_img, class_lable, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite('faces/with_mask/%dmask.jpg' % (img_count), color_img)

        else:
            print('user not wearing mask - prob = ', pred_prob[0][1])
            class_lable = "No Mask"
            color = (0, 255, 0)
            cv2.rectangle(color_img, (x, y), (x + w, y + h), (0, 0, 255), 3)
            # Using cv2.putText() method
            cv2.putText(color_img, class_lable, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
            cv2.imwrite('faces/with_mask/%dno_mask.jpg' % (img_count), color_img)

    # display image
    dim = (600, 600)
    resized = cv2.resize(color_img, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow('LIVE face mask detection', resized)

    cv2.waitKey()

    # close all windows
    cv2.destroyAllWindows()

input('Press ENTER to exit')
