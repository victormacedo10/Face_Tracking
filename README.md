# Face_Tracking

Face tracking system using Mean Shift for tracking and Haar Cascades for detection. Project made for the Digital Image Processing course at University of Brasilia UnB. All the code was written using python and requires opencv.

- Projeto_Final.py makes use of several opencv builtin trackers and compare them with the implemented MeanShift algortithm. The dataset used contains samples extracted from https://www.kaggle.com/kmader/videoobjecttracking. The results are saved both as txt files, saving the bounding boxes in each frame, and as a video with the bounding box overlaid in the original video, in the Results/Videos folder. Tables are made using pandas dataframe for the comparison using some given metrics, as described in the article, and saved in the Tables Folder as csv.

- generate_video.py tests the method in a video from author and saves it in the Results/Videos folder.

- MS_webcam.py allows for a real time tracking using the webcam.

Some sample videos from the results were given in the main repository.
