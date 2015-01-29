import cv2
import numpy as np
import os

# read image
for name in os.listdir("samples"):
    # load the image, clone it for output, and then convert it to grayscale
    print name
    image = cv2.imread("samples/" + name)
    output = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.cv.CV_HOUGH_GRADIENT, 1.2, 350)

    # ensure at least some circles were found
    if circles is not None:
    # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")

    # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            print(x, y, r)
    # draw the circle in the output image, then draw a rectangle
    # corresponding to the center of the circle
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imwrite('output/' + name, output)

    # ret, thresh = cv2.threshold(image, 127, 255, 0)
    # contours, heirarchy = cv2.findContours(thresh, 1, 2)

    # if contours:
    #     # output = image.copy()

    #     # for cnt in contours:
    #     #     try:
    #     #         (x, y), r = cv2.minEnclosingCircle(cnt)
    #     #         cv2.circle(output, (int(x), int(y)), int(r), (0, 255, 0), 4)
    #     #     except Exception as e:
    #     #         pass

    #     # cv2.imwrite('cir_output_' + name + '.jpg', output)

    #     output = image.copy()

    #     for cnt in contours:
    #         try:
    #             ellipse = cv2.fitEllipse(cnt)
    #             cv2.ellipse(output, ellipse, (0, 0, 255), 2)
    #         except Exception as e:
    #             pass

    #     cv2.imwrite('ell_output_' + name + '.jpeg', output)




# import cv2
# import numpy as np
# image = cv2.imread('6.jpeg', 0)
# ret, thresh = cv2.threshold(image, 127, 255, 0)
# contours, heirarchy = cv2.findContours(thresh, 1, 2)
# cnt = contours[0]
# M = cv2.moments(cnt)
# print(M)
# area = cv2.contourArea(cnt)
# epsilon = 0.1*cv2.arcLength(cnt, True)
# approx = cv2.approxPolyDP(cnt, epsilon, True)
# hull = cv2.convexHull(cnt)
# cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
# cv2.imshow('image', image)
