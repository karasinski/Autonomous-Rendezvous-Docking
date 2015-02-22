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

    features = circles, ellipses, rectangles
    c = features[0][:10]

    # Find the largest group of objects that are roughly the same size
    out = cluster(c, 0.5)
    largest = max(out, key=len)

    if len(largest) != 4:
        # print('Wrong number of elements, attempting to fix.', len(largest))
        pass

    # Iterative approach to fix problem
    tries, diff = 0, 0.1
    while len(largest) != 4:
        tries += 1

        if len(largest) > 4:
            out = cluster(c, 0.5 - diff)
        elif len(largest) < 4:
            out = cluster(c, 0.5 + diff)
        largest = max(out, key=len)

        if len(largest) == 4:
            # print('Fixed after this many tries', tries)
            break
        elif tries > 100:
            # print('Giving up!')
            raise ValueError
        else:
            diff += 0.01

    # Grab estimated distance
    estimated_distance = estimate_distance(largest)
    return xy, estimated_distance


def estimate_distance(markers):
    # Find the mean marker size
    x = np.array(markers)[:, 1].mean()

    # Fit parameters from calibration script
    p = [2.10044493e+02, -3.68823114e-01, 4.66638416e-02, -3.08006328e+00, 5.54876863e+01]

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

    if center[0] > np.array(image.shape[:2])[0]/2:
        y_sign = 1.
    else:
        y_sign = -1.

    if center[1] > np.array(image.shape[:2])[1]/2:
        z_sign = 1.
    else:
        z_sign = -1.

    # Estimate state vector
    X = distance
    Y = y_sign * estimate_offset(distance, YZ[1])
    Z = z_sign * estimate_offset(distance, YZ[0])

    return np.array([X, Y, Z])
