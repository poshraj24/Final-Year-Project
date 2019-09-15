#Import required modules
import cv2
import dlib
import math
import pandas as pd
import pathlib

#
# #Defining points representing the facial features
# REB = [17,18,19,20] #Right Eyebrow
# LEB = [22,23,24,25] #Left Eyebrow
# REYE = [36,37,38,39,40] #Right Eye
# REYE_UP= [36,37,38]
# REYE_DOWN = [39,40,41]
# REYE_UP_DOWN = [37,40]
# LEYE = [42,43,44,45,46] #Left Eye
# LEYE_UP =[42,43,44]
# LEYE_UP_DOWN = [43,46]
# LEYE_DOWN=[45,46,47]
# NSE_VER = [27,28,29] #Vertical Nose
# NSE_HOR = [31,32,33,34] # Horizontol Nose
# INR_LIPS = [60,61,62,63,64,65,66] #Inner Lips
# OUT_LIPS = [48,49,50,51,52,53,54,55,56,57,58] #Outer Lips
#
#
# #Defining a pandas dataframe with following columns
# df = pd.DataFrame(columns = ['FP01','FP02','FP03','FP04','FP05','FP06',
#                        'FP07','FP08','FP09','FP10','FP11','FP12',
#                        'FP13','FP14','FP15','FP16','FP17','FP18',
#                        'FP19','FP20','Emotion'])
#
# #Directory for the data files
# current_dir = pathlib.Path.cwd()
# path = pathlib.Path.cwd()  / 'cohn-kanade'/ '1'
#
# #Initializing the face detector and landmarks predictor
# detector = dlib.get_frontal_face_detector() #Face detector
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") #Landmark identifier.
#
# #Defining function for calculating the euclidian distance of two points and return it
def distance(shape, pnt_1 , pnt_2):
    a = (shape.part(pnt_2).x - shape.part(pnt_1).x)**2
    b = (shape.part(pnt_2).y - shape.part(pnt_1).y)**2
    dis = math.sqrt((a+b))
    return dis
#
# #Iterating through the datase folders to calculate the features
# for i in path.iterdir():
#     img = cv2.imread(str(i))
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#
#     detections = detector(gray, 1) #Detect the faces in the image
#     for k,d in enumerate(detections): #For each detected face
#         shape = predictor(gray, d) #Get coordinates
#
#         # Calculate average distance between right brows and  right eye
#         dist_1937 = distance(19,37)
#         dist_2038 = distance(20,38)
#         FP01 = (dist_1937 + dist_2038) / 2
#
#         # Calculate average distance between left brows and left eye
#         dist_2343 = distance(23,43)
#         dist_2444 = distance(24,44)
#         FP02 = (dist_2343 + dist_2444) / 2
#
#         FP03 = distance(21,39)  # Calculate distance between left corner point of right eye and brows
#         FP04 = distance(17,36)  # Calculate distance between right corner point of right eye and brows
#         FP05 = distance(22,42)  # Calculate distance between right corner point of left eye and brows
#         FP06 = distance(26,45)  # Calculate distance between left corner point of left eye and brows
#         FP07 = distance(21,22) # Calculate distance between corner point of two eyes
#
#         FP08 = distance(27,31) # Calculate distance between upper nose point and right most point of lower nose
#         FP09 = distance(27,35) # Calculate distance between upper nose point and left most point of lower nose
#         FP10 = (FP08 + FP09) / 2
#         FP11 = distance(30,33) # Calculate distance between lower centre nose point and upper centre nose point.
#         dist_3150 = distance(31,50) #calculate distance between nose right corner and right corner of upper lips
#         dist_3552 = distance(35,52) #calculate distance between nose left corner and left corner of upper lips
#         FP12 = (dist_3150 + dist_3552) /2
#         dist_3148 = distance(31,48)
#         dist_3554 = distance(35,54)
#         FP12 = (dist_3148+ dist_3554) /2
#         FP13 = distance(48,54)
#         FP14 = distance(61,67) #  calculate distance between right corner of inner lips
#         FP15 = distance(62,66)  #  calculate distance between left corner of inner lips
#         FP16 = distance(63,65) #  calculate distance between middle of inner lips
#         FP17 = distance(58,7) # calculate distance between right corner of lower lips and right chin
#         FP18 = distance(57,8) # calculate distance between middle of lower lips and middlechin
#         FP19 = distance(56,9) # calculate distance between left corner of lower lips and left chin
#         FP20 = distance(60,64) # calculate distance between inner lips corner
#         label = 1
#         data = [pd.Series([FP01,FP02,FP03,FP04,FP05,FP06,FP07,FP08,FP09,FP10,
#                  FP11,FP12,FP13,FP14,FP15,FP16,FP17,FP18,FP19,FP20, label], index = df.columns)]
#
#         df = df.append(data, ignore_index=True)
#
#        # for i in range(1,68): #There are 68 landmark points on each face
#         #    #For each point, draw a red circle with thickness2 on the original frame
#          #   cv2.circle(gray, (shape.part(i).x, shape.part(i).y), 1, (255,0,0), thickness=2)
#
#
#    # cv2.imshow("Facial Sentiment Analysis", gray) #Display the frame
#    # cv2.waitKey()
#
# df.to_csv(r'face_dataset.csv', header = True)
#

def extract_features(detector, predictor, image_file):

    img = cv2.imread(image_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector(gray, 1)[0]  # Detect the faces in the image

    shape = predictor(gray, detections)  # Get coordinates

    # Calculate average distance between right brows and  right eye
    dist_1937 = distance(shape,19, 37)
    dist_2038 = distance(shape,20, 38)
    FP01 = (dist_1937 + dist_2038) / 2

    # Calculate average distance between left brows and left eye
    dist_2343 = distance(shape,23, 43)
    dist_2444 = distance(shape,24, 44)
    FP02 = (dist_2343 + dist_2444) / 2

    FP03 = distance(shape,21, 39)  # Calculate distance between left corner point of right eye and brows
    FP04 = distance(shape,17, 36)  # Calculate distance between right corner point of right eye and brows
    FP05 = distance(shape,22, 42)  # Calculate distance between right corner point of left eye and brows
    FP06 = distance(shape,26, 45)  # Calculate distance between left corner point of left eye and brows
    FP07 = distance(shape,21, 22)  # Calculate distance between corner point of two eyes

    FP08 = distance(shape,27, 31)  # Calculate distance between upper nose point and right most point of lower nose
    FP09 = distance(shape,27, 35)  # Calculate distance between upper nose point and left most point of lower nose
    FP10 = (FP08 + FP09) / 2
    FP11 = distance(shape,30, 33)  # Calculate distance between lower centre nose point and upper centre nose point.
    dist_3150 = distance(shape,31, 50)  # calculate distance between nose right corner and right corner of upper lips
    dist_3552 = distance(shape,35, 52)  # calculate distance between nose left corner and left corner of upper lips
    FP12 = (dist_3150 + dist_3552) / 2
    dist_3148 = distance(shape,31, 48)
    dist_3554 = distance(shape,35, 54)
    FP12 = (dist_3148 + dist_3554) / 2
    FP13 = distance(shape,48, 54)
    FP14 = distance(shape,61, 67)  # calculate distance between right corner of inner lips
    FP15 = distance(shape,62, 66)  # calculate distance between left corner of inner lips
    FP16 = distance(shape,63, 65)  # calculate distance between middle of inner lips
    FP17 = distance(shape,58, 7)  # calculate distance between right corner of lower lips and right chin
    FP18 = distance(shape,57, 8)  # calculate distance between middle of lower lips and middle chin
    FP19 = distance(shape,56, 9)  # calculate distance between left corner of lower lips and left chin
    FP20 = distance(shape,60, 64)  # calculate distance between inner lips corner
    data = [FP01, FP02, FP03, FP04, FP05, FP06, FP07, FP08, FP09, FP10,
                       FP11, FP12, FP13, FP14, FP15, FP16, FP17, FP18, FP19, FP20]
    return data

def extract_features_from_img(detector, predictor, img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector(gray, 1)[0]  # Detect the faces in the image

    shape = predictor(gray, detections)  # Get coordinates

    # Calculate average distance between right brows and  right eye
    dist_1937 = distance(shape,19, 37)
    dist_2038 = distance(shape,20, 38)
    FP01 = (dist_1937 + dist_2038) / 2

    # Calculate average distance between left brows and left eye
    dist_2343 = distance(shape,23, 43)
    dist_2444 = distance(shape,24, 44)
    FP02 = (dist_2343 + dist_2444) / 2

    FP03 = distance(shape,21, 39)  # Calculate distance between left corner point of right eye and brows
    FP04 = distance(shape,17, 36)  # Calculate distance between right corner point of right eye and brows
    FP05 = distance(shape,22, 42)  # Calculate distance between right corner point of left eye and brows
    FP06 = distance(shape,26, 45)  # Calculate distance between left corner point of left eye and brows
    FP07 = distance(shape,21, 22)  # Calculate distance between corner point of two eyes

    FP08 = distance(shape,27, 31)  # Calculate distance between upper nose point and right most point of lower nose
    FP09 = distance(shape,27, 35)  # Calculate distance between upper nose point and left most point of lower nose
    FP10 = (FP08 + FP09) / 2
    FP11 = distance(shape,30, 33)  # Calculate distance between lower centre nose point and upper centre nose point.
    dist_3150 = distance(shape,31, 50)  # calculate distance between nose right corner and right corner of upper lips
    dist_3552 = distance(shape,35, 52)  # calculate distance between nose left corner and left corner of upper lips
    FP12 = (dist_3150 + dist_3552) / 2
    dist_3148 = distance(shape,31, 48)
    dist_3554 = distance(shape,35, 54)
    FP12 = (dist_3148 + dist_3554) / 2
    FP13 = distance(shape,48, 54)
    FP14 = distance(shape,61, 67)  # calculate distance between right corner of inner lips
    FP15 = distance(shape,62, 66)  # calculate distance between left corner of inner lips
    FP16 = distance(shape,63, 65)  # calculate distance between middle of inner lips
    FP17 = distance(shape,58, 7)  # calculate distance between right corner of lower lips and right chin
    FP18 = distance(shape,57, 8)  # calculate distance between middle of lower lips and middle chin
    FP19 = distance(shape,56, 9)  # calculate distance between left corner of lower lips and left chin
    FP20 = distance(shape,60, 64)  # calculate distance between inner lips corner
    data = [FP01, FP02, FP03, FP04, FP05, FP06, FP07, FP08, FP09, FP10,
                       FP11, FP12, FP13, FP14, FP15, FP16, FP17, FP18, FP19, FP20]
    return data