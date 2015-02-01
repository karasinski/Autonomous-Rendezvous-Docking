import numpy as np
import cv2


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

    # Convert this to grayscale and take threshold and contours
    gray_result = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_result, 127, 255, 0)
    contours, heirarchy = cv2.findContours(thresh, 1, 2)

    # Go through contours and detect objects
    output = image1.copy()
    circles, ellipses, rectangles = [], [], []
    if contours:
        for cnt in contours:
            try:
                ellipse = cv2.fitEllipse(cnt)
                circle = cv2.minEnclosingCircle(cnt)
                rectangle = cv2.boundingRect(cnt)

                ellipses.append(ellipse)
                circles.append(circle)
                rectangles.append(rectangle)
            except Exception as e:
                # print(e)
                pass

        # Sort these arrays from larges to smallest area
        circles.sort(key=lambda circle: circle[1] ** 2, reverse=True)
        ellipses.sort(key=lambda ellipse: ellipse[1][0] * ellipse[1][1], reverse=True)
        rectangles.sort(key=lambda rectangle: rectangle[2] * rectangle[3], reverse=True)

        # Grab the largest elements
        c = circles[0:6]
        e = ellipses[0:6]
        r = rectangles[0:6]

        # Draw the elements
        for circle, ellipse, rectangle in zip(c, e, r):
            # # Draw an ellipse
            # cv2.ellipse(output, ellipse, (0, 0, 255), 2)

            # # Draw a circle
            # x = int(circle[0][0])
            # y = int(circle[0][1])
            # r = int(circle[1])
            # cv2.circle(output, (x, y), r, (0, 255, 0), 2)

            # Draw a rectangle
            x, y, w, h = rectangle
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Show the result
        # cv2.imshow('ellipse', output)
        # cv2.waitKey(0)

    # Save the result
    cv2.imwrite("output/ISS_" + distance + "_features.jpeg", output)
