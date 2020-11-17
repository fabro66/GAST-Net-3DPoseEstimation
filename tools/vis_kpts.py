import numpy as np
import cv2


joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6],
               [0, 7], [7, 8], [8, 9], [9, 10], [8, 11], [11, 12],
               [12, 13], [8, 14], [14, 15], [15, 16]]

colors_kps = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [50, 205, 50], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255]]


def write(x, img):
    # c1 = tuple(x[1:3].int())
    # c2 = tuple(x[3:5].int())
    c1 = (int(x[0]), int(x[1]))
    c2 = (int(x[2]), int(x[3]))

    cls = int(x[-1])
    color = [0, 97, 255]
    label = 'People {}'.format(x[-1])
    cv2.rectangle(img, c1, c2, color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2, color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);
    return img


def plot_keypoint(image, coordinates):
    # USE cv2
    for i in range(coordinates.shape[0]):
        pts = coordinates[i]
        for color_i, jp in zip(colors_kps, joint_pairs):
            pt0 = pts[jp, 0]
            pt1 = pts[jp, 1]
            pt0_0, pt0_1, pt1_0, pt1_1 = int(pt0[0]), int(pt0[1]), int(pt1[0]), int(pt1[1])

            cv2.line(image, (pt0_0, pt1_0), (pt0_1, pt1_1), color_i, 5)
            # cv2.circle(image,(pt0_0, pt0_1), 2, color_i, thickness=-1)
            # cv2.circle(image,(pt1_0, pt1_1), 2, color_i, thickness=-1)
    return image

