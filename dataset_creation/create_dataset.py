import cv2
import os
import sys
import numpy as np


bg_folder = './data/scenic/'
fg_folder = './data/animal_database/'
output_folder = './data/fake3/'
N = 10048  # Dataset size


def get_image_path(root, img_type='jpg'):
    walks = []
    paths = []

    for w in os.walk(root):
        walks.append(w)

    for w in walks:
        w_path = [w[0] + '/' + i for i in w[2] if i[-3:] == img_type]
        paths = paths + w_path

    return paths


def merge_fg_bg(fg, seg, bg):
    fg = cv2.resize(fg, bg.shape[:-1])
    seg = cv2.resize(seg, bg.shape[:-1])

    # Normalize segmentation
    seg = 1.0 * seg / seg.max()
    seg[seg == seg.min()] = 0.

    neg_seg = 1.0 - seg

    fg = fg * 1.    # Cast foreground to float
    fg_crop = cv2.multiply(fg, neg_seg)

    bg = bg * 1.    # Cast bg to float
    bg_hole = cv2.multiply(bg, seg)

    composed = cv2.add(fg_crop, bg_hole)

    return composed, seg * 255




def main():
    fg_img_files = get_image_path(fg_folder, 'jpg')
    # fg_segmentation_files = get_image_path(fg_folder, 'png')
    bg_files = get_image_path(bg_folder)

    fg_img_files = np.random.choice(fg_img_files, size=N, replace=True)
    fg_segmentation_files = np.array([p.replace('original', 'segment').replace('jpg', 'png') for p in fg_img_files])
    bg_files = np.random.choice(bg_files, size=N, replace=True)

    for i, (fg_file, seg_file, bg_file) in enumerate(zip(fg_img_files, fg_segmentation_files, bg_files)):
        fg = cv2.imread(fg_file)
        seg = cv2.imread(seg_file)
        bg = cv2.imread(bg_file)

        composed, mask = merge_fg_bg(fg, seg, bg)
        composed = np.concatenate((composed, mask), 1)
        cv2.imwrite(output_folder + '/imgs/img/' + str(i) + '.png', composed)
        #cv2.imwrite(output_folder + '/masks/mask/' + str(i) + '.png', mask)



if __name__ == '__main__':
    main()






