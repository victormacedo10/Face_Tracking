import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

v = cv2.VideoCapture("ftv.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')

file_k = pd.read_csv("Results/ftvms.txt", header=None, delimiter="\n")
labels = np.array(file_k[0].str.split(',', expand=True))

out = cv2.VideoWriter('Results/Videos/ftv_mean_shift.mp4',fourcc, 20.0, (640,480))

for i in range(1, len(labels)):
    ret ,frame = v.read()
    x,y,w,h = (labels[i,:].astype(float)).astype(int)
    cv2.putText(frame, "MeanShift", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2);
    cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    out.write(frame)
    cv2.imshow('video',frame)

out.release()
cv2.destroyAllWindows()
v.release()