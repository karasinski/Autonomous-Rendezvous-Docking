import dcomm
import cv2
import numpy as np
import urllib
from time import sleep


def load_image():
    width = 1600
    height = 900
    try:
        # Give the server some room to breath
        sleep(.02)

        req = urllib.urlopen(
            'http://127.0.0.1:8080/image?width=' + str(width) + '&height=' + str(height))
    except IOError:
        print('EDGE is not running as a local server.')

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1)  # 'load it as it is'
    return img


def scan_docking_port(target):
    # Import images
    target.flags['VRD_HIGHLIGHT_NODE'] = False
    dcomm.client()
    image1 = load_image()

    target.flags['VRD_HIGHLIGHT_NODE'] = True
    dcomm.client()
    image2 = load_image()

    # Find difference between images
    image = image1 - image2

    RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define range of color in RGB
    lower_white = np.array([100, 100, 100], dtype=np.uint8)
    upper_white = np.array([200, 200, 200], dtype=np.uint8)

    # Threshold the RGB image to get only desired colors
    mask = cv2.inRange(RGB, lower_white, upper_white)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Convert this to gray scale and take threshold and contours
    gray_result = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_result, 127, 255, 0)
    contours, heirarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    output = image1.copy()

    return output, contours


def detect_features(image, contours):
    ''' Go through contours and detect features '''
    circles, ellipses, rectangles = [], [], []
    cnts = []
    for cnt in contours:
        try:
            # Cannot uniquely fit an ellipse with less than 5 points
            if len(cnt) > 5:
                ellipse = cv2.fitEllipse(cnt)
            circle = cv2.minEnclosingCircle(cnt)
            rectangle = cv2.boundingRect(cnt)

            ellipses.append(ellipse)
            circles.append(circle)
            rectangles.append(rectangle)

            area = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if area > 0:
                cnts.append([area, perimeter, cnt])
        except Exception as e:
            # print(e)
            pass

    # Sort these arrays from largest to smallest area, drop repeats
    circles = list(set(circles))
    ellipses = list(set(ellipses))
    rectangles = list(set(rectangles))

    circles.sort(key=lambda circle: circle[1] ** 2, reverse=True)
    ellipses.sort(key=lambda ellipse: ellipse[1][0] * ellipse[1][1], reverse=True)
    rectangles.sort(key=lambda rectangle: rectangle[2] * rectangle[3], reverse=True)

    cnts.sort(key=lambda cnt: cnt[0], reverse=True)

    # Locate central marker
    rect = cv2.minAreaRect(cnts[0][2])
    xy = rect[0]

    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    # cv2.drawContours(image, [box], 0, (0, 0, 255), 2)

    # cv2.imshow('image', image)
    # cv2.waitKey(0)

    # return circles, ellipses, rectangles

    features = circles, ellipses, rectangles
    c = features[0][:10]

    # Find the largest group of objects that are roughly the same size
    out = cluster(c, 0.5)
    c = max(out, key=len)
    if len(c) != 4:
        print('Wrong number of elements.')

    # Grab estimated distance
    estimated_distance = estimate_distance(c)
    return xy, estimated_distance


def estimate_distance(markers):
    # Find the mean marker size
    x = np.array(markers)[:, 1].mean()

    # Fit parameters from calibration script
    p = [1.87599710e+02, -3.59449427e-01, 4.50895146e-02, -2.95463925e+00, 5.29948831e+01]

    return p[0] * np.exp(p[1] * x) + p[2] * x ** 2 + p[3] * x + p[4]


def estimate_offset(distance, pixels):
    # Fit parameters from calibration script
    return distance * pixels / 566.89611162999995


def cluster(data, maxgap):
    '''
    Arrange data into groups where successive elements differ by no more than
    maxgap.
    '''

    groups = [[data[0]]]
    for x in data[1:]:
        if abs(x[1] - groups[-1][-1][1]) <= maxgap:
            groups[-1].append(x)
        else:
            groups.append([x])
    return groups


class Point:
    x, y = 0.0, 0.0

    def __init__(self, x, y):
        self.data = []
        self.x = x
        self.y = y


def DistanceFormula(p1, p2):
    return np.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)


def SenseYZ(ctr_pt, frame_ctr):
    ''' calculate the y and z offset from the frame center'''

    center = Point(ctr_pt[0], ctr_pt[1])
    frame = Point(frame_ctr[0], frame_ctr[1])

    # if(abs(center.x - frame.x) <= 1.0) and (abs(center.y - frame.y) <= 1):
    #     center.x = frame.x
    #     center.y = frame.y

    temp_pt = Point(center.x, frame.y)
    y = DistanceFormula(center, temp_pt)
    z = DistanceFormula(temp_pt, frame)

    return np.array([y, z])


def estimate_state(center, distance, image):
    center = [center[1], center[0]]
    YZ = SenseYZ(center, np.array(image.shape[:2])/2)

    # Estimate state vector
    X = distance
    Y = estimate_offset(distance, YZ[0])
    Z = estimate_offset(distance, YZ[1])

    return np.array([X, Y, Z])
