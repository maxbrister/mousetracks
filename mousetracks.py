import bz2
import cv2
import dlib
import numpy as np
import tarfile
import os
import os.path
import sys
import urllib.request
from os import path

def download_predictor():
    name = 'shape_predictor_68_face_landmarks.dat'
    url = f'http://dlib.net/files/{name}.bz2'
    dest = path.join('data', name)
    if path.exists(dest):
        return dest

    os.makedirs('data')
    tmp_dest = dest + '.bz2'
    urllib.request.urlretrieve(url, tmp_dest)
    fin = bz2.open(tmp_dest)
    data = fin.read()
    with open(dest, 'wb') as fout:
        fout.write(data)
    return dest


def largest_face(faces):
    selected = 0
    selected_area = faces[0].width() * faces[0].height()
    for idx in range(1, len(faces)):
        area = faces[idx].width() * faces[idx].height()
        if area > selected_area:
            selected_area = area
            selected = idx
    return faces[selected]

def eyepos(img, highlight_idx):
    faces = detector(img, 0)
    if len(faces) == 0:
        return img

    face = largest_face(faces)
    shape = predictor(img, face)
    for (idx, p) in enumerate(shape.parts()):
        clr = (255, 0, 0) if idx == highlight_idx else (255, 255, 0)
        cv2.rectangle(img, (p.x-2, p.y-2), (p.x+2, p.y+2), clr, 2)
    return img


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(download_predictor())

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    highlight_idx = 0
    while True:
        ret, img = cap.read()
        img = eyepos(img, highlight_idx)
        highlight_idx %= 68
        cv2.putText(img, str(highlight_idx), (0, 25), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255))
        cv2.imshow('Output', img)
        key = cv2.waitKey(1)
        if key == 9: # tab
            highlight_idx += 1
        elif key == 98: # b
            highlight_idx -= 1
        elif key != -1:
            print(key)
            break