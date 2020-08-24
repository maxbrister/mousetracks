import bz2
import cv2
import dlib
import math
import numpy as np
import tarfile
import os
import os.path
import pyautogui
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

def face_68(img):
    faces = detector(img, 0)
    if len(faces) == 0:
        return []

    face = largest_face(faces)
    shape = predictor(img, face)
    return shape.parts()

def draw_68(img, shape, highlight_idx):
    for (idx, p) in enumerate(shape):
        clr = (255, 0, 0) if idx == highlight_idx else (255, 255, 0)
        cv2.rectangle(img, (p.x-2, p.y-2), (p.x+2, p.y+2), clr, 2)
    return img

def dist(a, b):
    return math.sqrt(a.x * b.x + a.y * b.y)

class ClickOnOpen:
    def __init__(self):
        self.calibration = []
        self.calibrating = True
        self.mouse_down = False

    def update(self, img, face):
        if len(face) != 68:
            return

        if self.calibrating:
            self.calibrate(face)
            return

        # Determine the current nose/mouth ratio and compare it against the calibrated ratio
        (nose_dist, mouth_dist) = self.compute(face)
        ratio = mouth_dist / nose_dist
        diff = ratio - self.ratio
        if DEBUG:
            print(f'Ratio diff: {diff}')
        down = diff > 0.01

        # Update the mouse state to match the mouth state
        if down != self.mouse_down:
            self.mouse_down = down
            if down:
                pyautogui.mouseDown()
            else:
                pyautogui.mouseUp()

    def calibrate(self, face):
        if len(face) != 68:
            return

        item = self.compute(face)
        self.calibration.append(item)
        if len(self.calibration) >= 100:
            self.calibrating = False
            nose_avg = sum([nose for (nose, _) in self.calibration])
            mouth_avg = sum([mouth for (_, mouth) in self.calibration])
            self.ratio = mouth_avg / nose_avg

    def compute(self, face):
        ''' Computes the vertical nose and mouth length

        return [nose_length, mouth_length]
        '''
        nose_top = face[27]
        nose_bottom = face[30]
        nose_dist = dist(nose_top, nose_bottom)

        mouth_top = face[51]
        mouth_bottom = face[57]
        mouth_dist = dist(mouth_top, mouth_bottom)
        return (nose_dist, mouth_dist)
        

class Debug68:
    def __init__(self):
        self.highlight_idx = 0
        self.calibrating = False
    
    def update(self, img, face):
        # Explore the 68 face points
        #
        # Really hacky... Press an key (except tab or b) to exit.
        # tab - Go to next point
        # b - Go to brevious point (back)
        highlight_idx = self.highlight_idx % 68
        img = draw_68(img, face, highlight_idx)
        cv2.putText(img, str(highlight_idx), (0, 25), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 1, (255, 255, 255))
        cv2.imshow('Output', img)
        key = cv2.waitKey(1)
        if key == 9: # tab
            self.highlight_idx += 1
        elif key == 98: # b
            self.highlight_idx -= 1
        elif key != -1:
            print(key)
            sys.exit(0)

DEBUG = False
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(download_predictor())

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    funcs = [ClickOnOpen()]
    if len(sys.argv) > 1 and sys.argv[1] in ['-d', '--debug']:
        funcs.append(Debug68())
        DEBUG = True
    print('Calibrating. Please keap mouth closed...')
    calibrating = True
    while True:
        ret, img = cap.read()
        face = face_68(img)

        for f in funcs:
            f.update(img, face)

        if calibrating and all([not f.calibrating for f in funcs]):
            calibrating = False
            print('Calibration complete')

    cap.release()