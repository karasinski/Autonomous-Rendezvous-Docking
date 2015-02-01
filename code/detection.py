from __future__ import division
import numpy as np
from numpy.linalg import eig, inv
import cv2
from scipy import optimize


def calc_R(x, y, xc, yc):
    """
    Calculate the distance of each 2D points from the center (xc, yc).
    """
    return np.sqrt((x - xc) ** 2 + (y - yc) ** 2)


def f(c, x, y):
    """
    Calculate the algebraic distance between the data points and the mean
    circle centered at c=(xc, yc).
    """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()


def leastsq_circle(x, y):
    ''' Coordinates of the barycenter. '''
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x, y))
    xc, yc = center
    Ri = calc_R(x, y, *center)
    R = Ri.mean()
    residu = np.sum((Ri - R) ** 2)
    return xc, yc, R, residu


def find_ellipse(x, y):
    xmean = x.mean()
    ymean = y.mean()
    x -= xmean
    y -= ymean
    a = fitEllipse(x, y)
    center = ellipse_center(a)
    center[0] += xmean
    center[1] += ymean
    phi = ellipse_angle_of_rotation(a)
    axes = ellipse_axis_length(a)
    x += xmean
    y += ymean
    return center, phi, axes


def fitEllipse(x, y):
    x = x[:, np.newaxis]
    y = y[:, np.newaxis]
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))
    S = np.dot(D.T, D)
    C = np.zeros([6, 6])
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1
    E, V = eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:, n]
    return a


def ellipse_center(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    num = b * b - a * c
    x0 = (c * d - b * f) / num
    y0 = (a * f - b * d) / num
    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    return 0.5 * np.arctan(2 * b / (a - c))


def ellipse_axis_length(a):
    b, c, d, f, g, a = a[1] / 2, a[2], a[3] / 2, a[4] / 2, a[5], a[0]
    up = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
    down1 = (b * b - a * c) * ((c - a) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    down2 = (b * b - a * c) * ((a - c) * np.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
    res1 = np.sqrt(up / down1)
    res2 = np.sqrt(up / down2)
    return np.array([res1, res2])


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


for distance in ["5", "10", "20", "40", "60", "80", "100"]:
    # Import images
    image1 = cv2.imread("samples/ISS_" + distance + ".jpeg")
    image2 = cv2.imread("samples/ISS_" + distance + "_HIGHLIGHT.jpeg")

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
    contours, heirarchy = cv2.findContours(thresh, 1, 2)

    # Go through contours and detect objects
    output = image1.copy()
    circles, ellipses, rectangles = [], [], []
    if contours:
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

        # Grab the largest elements
        c = circles[:6]
        e = ellipses[:6]
        r = rectangles[:6]

        # Find the largest group of objects that are roughly the same size
        out = cluster(c, 0.5)
        c = max(out, key=len)
        if len(c) != 4:
            print('Wrong number of elements.')

        # Grab the centers of the circles as the 'best guess' location for the
        # identification markers
        xy = np.array([circ[0] for circ in c])
        x = xy[:, 0]
        y = xy[:, 1]

        # Draw the elements
        # for circle, ellipse, rectangle in zip(c, e, r):
        for circle in c:
            # Draw an ellipse
            # cv2.ellipse(output, ellipse, (0, 0, 255), 2)

            # Draw a circle
            x_c = int(circle[0][0])
            y_c = int(circle[0][1])
            r_c = int(circle[1])
            cv2.circle(output, (x_c, y_c), r_c, (0, 255, 0), 2)

            # Draw a rectangle
            # x, y, w, h = rectangle
            # cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)


        # Try to find the best ellipse, if ellipse fitting fails, fall back to
        # fitting a circle

        # Find the best fit ellipse
        center, phi, axes = find_ellipse(x, y)
        if True not in np.isnan(axes):
            axes = np.array([2 * axes[0], 2 * axes[1]])
            ellipse_fit = (tuple(abs(center)), tuple(abs(axes)), abs(phi))
            # print(ellipse_fit)
            cv2.ellipse(output, ellipse_fit, (0, 0, 255), 2)
        else:
            # Find the best fit circle
            x_fit, y_fit, r_fit, _ = leastsq_circle(x, y)
            cv2.circle(output, (int(x_fit), int(y_fit)), int(r_fit), (255, 0, 0), 2)

    # Show the result
    # cv2.imshow('output', output)
    # cv2.waitKey(0)

    # Save the result
    cv2.imwrite("output/ISS_" + distance + "_features.jpeg", output)
