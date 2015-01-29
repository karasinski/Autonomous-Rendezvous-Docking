import cv2
import numpy as np
import os


min_contour_length = 10
max_aspect_ratio = 10.
max_relative_inclination = 30.
max_ctrs_distance = 5.
max_sizes_ratio_error = [0.25, 0.50]
outer_circle_diam = 1500.  # mm
inner_circle_diam = 1220.  # mm
focal_length = 100.  # mm
radius = outer_circle_diam / 2.


def aspect_ratio(e):
    return abs(e['major'] / e['minor'])


def aspect_ratio_ok(e):
    return aspect_ratio(e) <= max_aspect_ratio


def distance_ok(a, b):
    return np.linalg.norm(a['ctr'] - b['ctr']) <= max_ctrs_distance


def sizes_ratio_ok(larger, smaller):
    outer = outer_circle_diam
    inner = inner_circle_diam
    correct_ratio = outer / inner
    ratio = abs(larger['major'] / smaller['major'])
    not_too_small = ratio > (correct_ratio - max_sizes_ratio_error[0])
    not_too_big = ratio < (correct_ratio + max_sizes_ratio_error[1])
    return not_too_small and not_too_big


def both_circular(a, b):
    return (aspect_ratio(a) < 1.2) and (aspect_ratio(b) < 1.2)


def relative_inclination_ok(a, b):
    return abs(a['alpha'] - b['alpha']) < max_relative_inclination


def larger_smaller(a, b):
    if b['major'] > a['major']:
        return b, a
    else:
        return a, b


def ellipse_to_dict(ellipse):
    ed = {}
    ed['minor'], ed['major'] = ellipse[1][0], ellipse[1][1]
    ed['ctr'] = np.asarray(ellipse[0])
    ed['alpha'] = ellipse[2]
    ed['object'] = ellipse

    return ed


def marker_filter(ellipses):
    candidates = []

    for i, ellipse in enumerate(ellipses):
        a = ellipse_to_dict(ellipse)

        try:
            if not aspect_ratio_ok(a):
                # print('aspect_ratio_ok(a) fail')
                continue

            for j in xrange(len(ellipses) - (i + 1)):
                b = ellipse_to_dict(ellipses[j + i + 1])
                larger, smaller = larger_smaller(a, b)

                if not aspect_ratio_ok(b):
                    # print('aspect_ratio_ok(b) fail')
                    continue
                if not distance_ok(a, b):
                    # print('distance_ok(a, b) fail')
                    continue
                if not both_circular(a, b):
                    # print('both_circular(a, b) fail')
                    if not relative_inclination_ok(a, b):
                        # print('relative_inclination_ok(a, b) fail')
                        continue
                if not sizes_ratio_ok(larger, smaller):
                    # print('sizes_ratio_ok(larger, smaller) fail')
                    continue

                if larger['object'] not in candidates:
                    candidates.append(larger['object'])
        except Exception as e:
            print(e)

    return candidates


def get_quadratic(ellipse):
    """
    Obtains general quadratic form parameters (note that f does not denote
    the focal length) for :
    Ax^2 + 2Bxy + Cy^2 + 2Dx + 2Ey + F = 0
    And returns these in matrix form A, where p A p.T = 0 describes the
    ellipse (as in Chen et al., eqn. 2). The approach used is taken from:
    http://www.mathworks.ch/
        matlabcentral/answers/37124-ellipse-implicit-equation-coefficients

    """

    ((x_0, y_0), (b, a), alpha) = ellipse

    if b == 0.0 or a == 0.0:
        print('b = %f, a = %f'.format(b, a))

    X_0 = np.array([x_0, y_0])
    R_e = np.array([[np.cos(alpha), -(np.sin(alpha))],
                    [np.sin(alpha), np.cos(alpha)]])
    temp = np.diag(np.array([1.0 / np.power(a, 2), 1.0 / np.power(b, 2)]))
    M = np.dot(R_e, np.dot(temp, R_e.T))
    A = M[0, 0]
    B = M[0, 1]
    C = M[1, 1]
    D = A * x_0 + B * y_0
    E = B * x_0 + C * y_0
    F = np.dot(X_0.T, np.dot(M, X_0)) - 1.0

    return np.array([
        [A, B, D],
        [B, C, E],
        [D, E, F]
    ])


def get_obl_el_cone(E, f):
    """
    Returns the matrix of the oblique elliptical cone, as in
    Chen et al. eqn. 5.

    """

    return np.array([[E[0, 0], E[0, 1], - E[0, 2] / f],
                     [E[1, 0], E[1, 1], - E[1, 2] / f],
                     [- E[2, 0] / f, - E[2, 1] / f, E[2, 2] / (f * f)]])


def get_chen_eigend(Q):
    """
    Returns the eigendecomposition of the ellipse matrix, ordered as
    specified in Chen et al.
    """

    l, V = np.linalg.eig(Q)

    rev_abs_sort_ind = np.argsort(np.power(np.abs(l), -1))
    l_opt1 = l[rev_abs_sort_ind]
    if (l_opt1[0] * l_opt1[1] > 0) and (l_opt1[1] * l_opt1[2] < 0):
        return l_opt1, V[:, rev_abs_sort_ind]
    else:
        l_opt2 = np.array([l_opt1[1], l_opt1[2], l_opt1[0]])
        if (l_opt2[0] * l_opt2[1] > 0) and (l_opt2[1] * l_opt2[2] < 0):
            return l_opt2, V[:, (1, 2, 0)]
        else:
            l_opt3 = np.array([l_opt1[0], l_opt1[2], l_opt1[1]])
            if (l_opt3[0] * l_opt3[1] > 0) and (l_opt3[1] * l_opt3[2] < 0):
                return l_opt3, V[:, (0, 2, 1)]
            else:
                print('could not satisfy Chen et al. eqn. 16')


def get_Cs_Ns(l, V, r):
    """
    Derives the centre and normal of the circle expressed in the camera
    coordinate system. Employs the definitions in Chen et al., eqn. 20.

    """

    Cs = np.zeros((8, 3))
    Ns = np.zeros((8, 3))
    signs = [1.0, -1.0]
    i = 0
    for s1 in signs:
        for s2 in signs:
            for s3 in signs:

                z0 = (s3 * l[1] * r) / np.sqrt(-l[0] * l[2])

                temp = np.array([
                    [s2 * (l[2] / l[1]) * np.sqrt((l[0] - l[1]) / (l[0] - l[2]))],
                    [0],
                    [-s1 * (l[0] / l[1]) * np.sqrt((l[1] - l[2]) / (l[0] - l[2]))]
                ])

                Cs[i] = (z0 * np.dot(V, temp)).flatten()

                temp = np.array([
                    [s2 * np.sqrt((l[0] - l[1]) / (l[0] - l[2]))],
                    [0],
                    [- s1 * np.sqrt((l[1] - l[2]) / (l[0] - l[2]))]
                ])

                Ns[i] = (np.dot(V, temp)).flatten()

                i += 1

    return Cs, Ns


def remove_impossible(Cs, Ns):
    """
    As per Chen et al. eqn. 21, removes those (centre, normal) pairs that
    would represent a marker behind the camera or a marker facing away from
    the camera. This should leave two (C,N) pairs per ellipse.
    """

    c_result, n_result = np.zeros((2,3)), np.zeros((2,3))
    nr_found = 0
    for i in xrange(len(Cs)):
        zmask = np.array([[0.0, 0.0, 1.0]])
        faces = (np.dot(Ns[i].flatten(), zmask.T) < 0.0 )
        infront = (np.dot(Cs[i].flatten(), zmask.T) > 0.0)
        if faces and infront:
            c_result[nr_found] = Cs[i]
            n_result[nr_found] = Ns[i]
            nr_found += 1
    if not (nr_found == 2):
        print('nr(candidate (C,N) pairs) not 2')

    return c_result, n_result


# filenames = [str(i) for i in range(1, 11)]
filenames = os.listdir("samples")
for name in filenames:
    image = cv2.imread("samples/" + name, cv2.CV_LOAD_IMAGE_GRAYSCALE)
    orig = np.copy(image)

    canv = cv2.equalizeHist(orig)
    canv = cv2.GaussianBlur(canv, (7, 7), sigmaX=2., sigmaY=2.)
    canv = cv2.Canny(canv, 50, 150)

    conts, hierarchy = cv2.findContours(canv, cv2.RETR_LIST, cv2.CHAIN_APPROX_TC89_L1)
    cv2.drawContours(canv, conts, -1, (128, 128, 128))

    ellipses = []
    for cont in conts:
        if len(cont) < min_contour_length:
            continue
        ellipses.append(cv2.fitEllipse(cont))

    candidates = marker_filter(ellipses)
    for candidate in candidates:
        cv2.ellipse(img=canv,
                    box=candidate,
                    color=(255, 255, 255),
                    thickness = 2,
                    lineType = cv2.CV_AA)

    cv2.imwrite("output/" + name, canv)

    try:
        est_markers = []
        for ellipse in ellipses:
            E = get_quadratic(ellipse)
            Q = get_obl_el_cone(E, focal_length)
            l, V = get_chen_eigend(Q)
            Cs, Ns = get_Cs_Ns(l, V, radius)
            Cs, Ns = remove_impossible(Cs, Ns)
            est_markers.append((Cs, Ns))
            # print('Marker centered at ', Cs, 'with norm ', Ns)
    except Exception as e:
        print(e)
