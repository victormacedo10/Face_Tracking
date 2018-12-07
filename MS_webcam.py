import numpy as np
import cv2
import matplotlib.pyplot as plt

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

def detect_one_face(im):
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 8)
    if len(faces) == 0:
        return (0,0,0,0)
    if len(faces) > 1:
        x,y,w,h = faces[0]
        a_max = w*h
        n = 0
        for i in range(1, len(faces)):
            x,y,w,h = faces[i]
            a = w*h
            if(a>a_max):
                a_max = a
                n = i
        return faces[n]
    else:
        return faces[0]

def BackProject(im, hist):
    f = lambda x: hist[x]
    return f(im)

def MeanShift(im, window, stop):
    x,y,w,h = window
    if((x,y,w,h)==(0,0,0,0)):
        return (0,0,0,0)
    for k in range(1):
        im_crop = im[y:y+h,x:x+w,0]
        i = np.arange(1,im_crop.shape[0]+1)
        j = np.arange(1,im_crop.shape[1]+1)
        jj, ii = np.meshgrid(j, i, sparse=False)
        M00 = np.sum(im_crop)
        M10 = np.sum(np.multiply(jj,im_crop))
        M01 = np.sum(np.multiply(ii,im_crop))
        dx = int(np.round(M10/M00))
        dy = int(np.round(M01/M00))
        x_new = x+dx-int(w/2)
        y_new = y+dy-int(h/2)
        if(abs(x_new-x)<=1 or abs(y_new-y)<=1):
            break
        x = x_new
        y = y_new
    return (x, y, w, h)

def hsv_histogram_for_window(frame, window):
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
    im_hist, _ = np.histogram(hsv_roi, 180, [0,179], density = True)
    return roi_hist

v = cv2.VideoCapture(0);

track_window = (0,0,0,0)
while True:
    ret ,frame = v.read()
    frame = cv2.flip(frame, 1)
    x,y,w,h = detect_one_face(frame)
    track_window = (x,y,w,h)
    if((x,y,w,h) != (0,0,0,0)):
        pt = track_window
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
        break
    pt = track_window
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
while True:
    ret ,frame = v.read()
    frame = cv2.flip(frame, 1)
    if ret == False:
        break
    timer = cv2.getTickCount()
    x,y,w,h = detect_one_face(frame)
    if((x,y,w,h) != (0,0,0,0)):
        roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
        track_window = (x,y,w,h)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    dst = BackProject(hsv[:,:,0], roi_hist)
    track_window = MeanShift(dst, track_window, 10)
    x,y,w,h = track_window
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255),2)
    fps = float(cv2.getTickFrequency() / (cv2.getTickCount() - timer))
    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
v.release()
