import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

face_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')

# funcao implementada para o backproject do histograma para o mean shift
def BackProject(im, hist):
    f = lambda x: hist[x] # lookup table para o vetor de histograma
    return f(im) # imagem aplicada no histograma

# funcao implementada para o calculo iterativo do centroide no mean shift
def MeanShift(im, window, stop):
    x,y,w,h = window
    if((x,y,w,h)==(0,0,0,0)): # checa se janela e valida
        return (0,0,0,0) # retorna zero caso contrario
    for k in range(stop): # recalcula centroide stop vezes
        im_crop = im[y:y+h,x:x+w,0] # recorta regiao de interesse
        i = np.arange(1,im_crop.shape[0]+1)
        j = np.arange(1,im_crop.shape[1]+1)
        # produz matriz multiplicativa para o calculo de momento de ordem 1
        jj, ii = np.meshgrid(j, i, sparse=False)
        M00 = np.sum(im_crop) # momento de ordem zero
        if(M00 == 0): # retorna valor anterior caso momento tenha dado zero (regiao preta)
            return (x, y, w, h)
        M10 = np.sum(np.multiply(jj,im_crop)) # momento de primeira ordem na horizontal
        M01 = np.sum(np.multiply(ii,im_crop)) # momento de primeira ordem na vertical
        dx = int(np.round(M10/M00)) # correcao para o centroide em x a partir da ROI
        dy = int(np.round(M01/M00)) # correcao para o centroide em y a partir da ROI
        x_new = x+dx-int(w/2) # atualizacao temporaria de x
        y_new = y+dy-int(h/2) # atualizacao temporaria de y
        # segundo criterio de parada caso tenha atingido convergencia (min de 1 pixel)
        if(abs(x_new-x)<=1 or abs(y_new-y)<=1):
            break
        x = x_new # atualizacao definitiva de x
        y = y_new # atualizacao definitiva de y
    return (x, y, w, h) # retorna valores

# conversao de uma janela da imagem para o Hue e realizacao do histograma
def hsv_histogram_for_window(frame, window):
    c,r,w,h = window
    roi = frame[r:r+h, c:c+w]
    hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roi_hist = cv2.calcHist([hsv_roi],[0],None,[180],[0,180])
    return roi_hist

# funcao de delimitacao dos bounding boxes usadas pelos trackers da opencv
def bb_I(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    I = interArea / float(boxAArea + boxBArea - interArea)
    return I

# funcao implementada para de deteccao de face Viola Jones com cascatas de Haar
def detect_one_face(im):
    # converte para escala de cinza para mandar para a funcao
    gray=cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # aplica funcao
    faces = face_cascade.detectMultiScale(gray, 1.2, 8)
    if len(faces) == 0: # caso nenhuma face detectada, retorna zero
        return (0,0,0,0)
    if len(faces) > 1: # caso tenha sido mais de uma
        x,y,w,h = faces[0]
        a_max = w*h # calcular area da primeira
        n = 0
        # iterar em todas as faces identificadas e definir a com maior area
        for i in range(1, len(faces)):
            x,y,w,h = faces[i]
            a = w*h
            if(a>a_max):
                a_max = a
                n = i
        return faces[n] # retornar face com maior area
    else:
        return faces[0] # caso so uma tenha sido detectada, retorna-la

# funcao implementada para tracking com mean shift
def MS_FT(v, output_txt):
    output = open(output_txt,"w") # abre arquivo para salvar pontos
    track_window = (0,0,0,0)
    # le frames ate que a primeira face sera encontrada por Haar
    while True:
        ret ,frame = v.read()
        if ret == False:
            return
        x,y,w,h = detect_one_face(frame)
        track_window = (x,y,w,h)
        if((x,y,w,h) != (0,0,0,0)): # face encontrada definida como ROI inicial
            pt = track_window 
            break
        pt = track_window
        output.write("%d,%d,%d,%d\n" % pt) # escrever frames nao identificados

    # calcular histograma do H da ROI
    roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
    # itera no video completo
    while True:
        ret ,frame = v.read() # le cada frame
        if ret == False:
            break
        timer = cv2.getTickCount() # inicializa temporizador para o fps
        x,y,w,h = detect_one_face(frame) # detecta uma face
        # atualiza histograma da ROI caso face tenha sido detectada
        if((x,y,w,h) != (0,0,0,0)): 
            roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
            track_window = (x,y,w,h)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV) # calcula HSV do frame
        dst = BackProject(hsv[:,:,0], roi_hist) # realiza backproject do histograma
        # aplica mean shift com max de 10 iteracoes
        track_window = MeanShift(dst, track_window, 10)
        x,y,w,h = track_window # atualizacao regiao
        pt = (x,y,w,h)
        output.write("%d,%d,%d,%d\n" % pt) # salva regiao no arquivo
        # calcula fps pelo tempo desde a deteccao de face
        fps = float(cv2.getTickFrequency() / (cv2.getTickCount() - timer))

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# funcao implementada para tracking com mean shift e kalman
def MS_Kalman(v, output_txt):
    # funcionamento geral analogo ao mean shift, apenas com alguns acrescimoss
    print(output_txt)
    output = open(output_txt,"w")
    print(output_txt)
    track_window = (0,0,0,0)
    while True:
        ret ,frame = v.read()
        if ret == False:
            return
        x,y,w,h = detect_one_face(frame)
        track_window = (x,y,w,h)
        if((x,y,w,h) != (0,0,0,0)):
            pt = track_window
            break
        pt = track_window
        output.write("%d,%d,%d,%d\n" % pt)

    # declaracao dos parametros de modelo do filtro de Kalman
    kalman = cv2.KalmanFilter(4,2)
    kalman.measurementMatrix = np.array([[1,0,0,0],
                                         [0,1,0,0]],np.float32)
    kalman.transitionMatrix = np.array([[1,0,1,0],
                                        [0,1,0,1],
                                        [0,0,1,0],
                                        [0,0,0,1]],np.float32)
    kalman.processNoiseCov = np.array([[1,0,0,0],
                                       [0,1,0,0],
                                       [0,0,1,0],
                                       [0,0,0,1]],np.float32) * 0.03
    measurement = np.array((2,1), np.float32)
    prediction = np.zeros((2,1), np.float32)

    px, py = x+w/2, y+h/2  
    measurement = np.array([px, py], dtype='float32')
    prediction = kalman.predict() # realiza primeira previsao

    roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
    while True:
        ret ,frame = v.read()
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
        px, py = int(x+w/2), int(y+h/2)
        measurement = np.array([px, py], dtype='float32')
        if((x,y,w,h)!=(0,0,0,0)):
            # corrige medida caso face tenha sido identificada
            posterior = kalman.correct(measurement)
            px,py,_,_ = posterior
            
        prediction = kalman.predict() # realiza nova previsao
        px,py,_,_ = prediction # filtra os pontos 
        x, y = int(px-w/2), int(py-(h/2))
        pt = (x,y,w,h)
        output.write("%d,%d,%d,%d\n" % pt)
        fps = float(cv2.getTickFrequency() / (cv2.getTickCount() - timer))

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# funcao de tracking por camshift usada como comparacao
def CamShift_FT(v, output_txt):
    output = open(output_txt,"w")

    track_window = (0,0,0,0)
    while True:
        ret ,frame = v.read()
        if ret == False:
            return
        x,y,w,h = detect_one_face(frame)
        track_window = (x,y,w,h)
        if((x,y,w,h) != (0,0,0,0)):
            pt = track_window
            break
        pt = track_window
        output.write("%d,%d,%d,%d\n" % pt)

    roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
    
    term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
    while True:
        ret ,frame = v.read()
        if ret == False:
            break

        timer = cv2.getTickCount()

        x,y,w,h = detect_one_face(frame)
        if((x,y,w,h) != (0,0,0,0)):
            roi_hist = hsv_histogram_for_window(frame, (x,y,w,h))
            track_window = (x,y,w,h)
        hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        x,y,w,h = track_window
        pt = (x,y,w,h)
        output.write("%d,%d,%d,%d\n" % pt)
        fps = float(cv2.getTickFrequency() / (cv2.getTickCount() - timer));

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# funcao de tracking por lucas-kanade-tomasi usada como comparacao
def LK_FT(v, output_txt):
    output = open(output_txt,"w")

    ret, frame = v.read()
    if ret == False:
        return
    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
     
    lk_params = dict(winSize = (15, 15),
                     maxLevel = 4,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
     
    point_selected = False
    old_points = np.array([[]])
    w_prev, h_prev = 0, 0
    while True:
        ret, frame = v.read()
        if ret == False:
            break
        timer = cv2.getTickCount()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x,y,w,h = detect_one_face(frame)
        if((x,y,w,h) != (0,0,0,0)):
            x, y = int(x+w/2), int(y+h/2)
            old_points = np.array([[x, y]], dtype=np.float32)
            point_selected = True
            w_prev, h_prev = w, h


        if point_selected is True:

            new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
            old_gray = gray_frame.copy()
            old_points = new_points
     
            x, y = new_points.ravel()

        if((x,y)!=(0,0)):
            x, y, w, h = int(x-w/2), int(y-h/2), w_prev, h_prev
            pt = (x,y,w,h)
        else:
            pt = (0,0,0,0)

        fps = float(cv2.getTickFrequency() / (cv2.getTickCount() - timer));

        output.write("%d,%d,%d,%d\n" % pt)
        
    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# funcao de tracking por filtro de kalman usado como comparacao
def Kalman_FT(v, output_txt):
    output = open(output_txt,"w")

    track_window = (0,0,0,0)
    while True:
        ret ,frame = v.read()
        if ret == False:
            return
        x,y,w,h = detect_one_face(frame)
        track_window = (x,y,w,h)
        if((x,y,w,h) != (0,0,0,0)):
            pt = track_window
            break
        pt = track_window
        output.write("%d,%d,%d,%d\n" % pt)

    state = np.array([x+w/2,y+h/2,0,0], dtype='float64')
    kalman = cv2.KalmanFilter(4,2,0)	
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
                                    [0., 1., 0., .1],
                                    [0., 0., 1., 0.],
                                    [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    measurement = np.array([x+w/2, y+h/2], dtype='float64')
    
    w_prev, h_prev = w, h
    while True:
        ret ,frame = v.read()
        if ret == False:
            break
        timer = cv2.getTickCount()
        
        prediction = kalman.predict()
        x,y,w,h = detect_one_face(frame)
        measurement = np.array([x+w/2, y+h/2], dtype='float64')
            
        if not (x ==0 and y==0 and w==0 and h ==0):
            posterior = kalman.correct(measurement)
            w_prev, h_prev = w, h
        if x ==0 and y==0 and w==0 and h ==0:
            x,y,w,h = prediction
            w, h = w_prev, h_prev
        else:
            x,y,w,h = posterior
            w, h = w_prev, h_prev
        x, y, w, h = int(x-w/2), int(y-h/2), w, h
        pt = (x, y, w, h)
        fps = float(cv2.getTickFrequency() / (cv2.getTickCount() - timer));
        output.write("%d,%d,%d,%d\n" % pt)

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# funcao generica para as classes de tracker da opencv
def Tracking(tracker, vj, v, output_txt):
    output = open(output_txt,"w")

    track_window = (0,0,0,0)
    while True:
        ret ,frame = v.read()
        if ret == False:
            return
        x,y,w,h = detect_one_face(frame)
        track_window = (x,y,w,h)
        if((x,y,w,h) != (0,0,0,0)):
            pt = track_window
            break
        pt = track_window
        output.write("%d,%d,%d,%d\n" % pt)

    bbox = (x, y, w, h)

    ret = tracker.init(frame, bbox)

    while True:
        ret, frame = v.read()
        if not ret:
            break
        timer = cv2.getTickCount()
        x,y,w,h = detect_one_face(frame)
        if((x,y,w,h) != (0,0,0,0) and vj == True):
            bbox = (x, y, w, h)
        else: 
            ret, bbox = tracker.update(frame)

        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        output.write("%d,%d,%d,%d\n" % bbox)

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# ensemble do KCL com o lucas-kanade-tomasi
def KCL_LK(tracker, vj, v, output_txt):
    thres = 100
    output = open(output_txt,"w")
    
    track_window = (0,0,0,0)
    while True:
	    ret ,frame = v.read()
	    if ret == False:
	        return
	    x,y,w,h = detect_one_face(frame)
	    track_window = (x,y,w,h)
	    if((x,y,w,h) != (0,0,0,0)):
	        pt = track_window
	        break
	    pt = track_window
	    output.write("%d,%d,%d,%d\n" % pt)

    old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    bbox = (x, y, w, h)
    ret = tracker.init(frame, bbox)
    lk_params = dict(winSize = (15, 15),maxLevel = 4, criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    	
    w_prev, h_prev = w, h
    old_points = np.array([[int(x+w/2), int(y+h/2)]], dtype=np.float32)
    p_old = (int(x+w/2), int(y+h/2), w, h)
    
    while True:
        ret, frame = v.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        timer = cv2.getTickCount()
        x,y,w,h = detect_one_face(frame)
        if((x,y,w,h) == (0,0,0,0)):
            ret, bbox = tracker.update(frame)
            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            if((x,y,w,h)==(0,0,0,0)):
                old_points = np.array([[p_old[0], p_old[1]]], dtype=np.float32)
                p, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points,
	                                                        None, **lk_params)
                w,h = int(p_old[2]), int(p_old[3])
                x,y = int(p[0][0]-w/2), int(p[0][1]-h/2)
                p = (int(x+w/2), int(y+h/2))
                dist2 = np.sqrt((p[0]-p_old[0])**2 + (p[1]-p_old[1])**2)
                if(dist2>thres):
	            
	                x,y = int(p_old[0]-w/2), int(p_old[1]-h/2)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);
    
        bbox = (x, y, w, h)
        p_old = (int(x+w/2), int(y+h/2), w, h)
    
        output.write("%d,%d,%d,%d\n" % bbox)
        old_gray = gray_frame.copy()

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# ensemble do KCL com um filtro de kalman
def KCL_Kalman(tracker, vj, v, output_txt):
    thres = 100
    output = open(output_txt,"w")

    track_window = (0,0,0,0)
    while True:
        ret ,frame = v.read()
        if ret == False:
            return
        x,y,w,h = detect_one_face(frame)
        track_window = (x,y,w,h)
        if((x,y,w,h) != (0,0,0,0)):
            pt = track_window
            break
        pt = track_window
        output.write("%d,%d,%d,%d\n" % pt)


    bbox = (x, y, w, h)

    ret = tracker.init(frame, bbox)

    state = np.array([x+w/2,y+h/2,0,0], dtype='float64')
    kalman = cv2.KalmanFilter(4,2,0)	
    kalman.transitionMatrix = np.array([[1., 0., .1, 0.],
    	                                [0., 1., 0., .1],
    	                                [0., 0., 1., 0.],
    	                                [0., 0., 0., 1.]])
    kalman.measurementMatrix = 1. * np.eye(2, 4)
    kalman.processNoiseCov = 1e-5 * np.eye(4, 4)
    kalman.measurementNoiseCov = 1e-3 * np.eye(2, 2)
    kalman.errorCovPost = 1e-1 * np.eye(4, 4)
    kalman.statePost = state
    measurement = np.array([x+w/2, y+h/2], dtype='float64')

    w_prev, h_prev = w, h
    p_old = (int(x+w/2), int(y+h/2), w, h)

    while True:
        ret, frame = v.read()
        if not ret:
            break
        timer = cv2.getTickCount()
        prediction = kalman.predict()
        x,y,w,h = detect_one_face(frame)
        measurement = np.array([x+w/2, y+h/2], dtype='float64')
        if((x,y,w,h) != (0,0,0,0) and vj == True):
            posterior = kalman.correct(measurement)
            w_prev, h_prev = w, h
            x,y,_,_ = posterior
            x, y, w, h = int(x-w/2), int(y-h/2), w, h
        else: 
            ret, bbox = tracker.update(frame)
            x,y,w,h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            measurement = np.array([x+w/2, y+h/2], dtype='float64')
            if((x,y,w,h)==(0,0,0,0)):
                x,y,w,h = prediction
                w, h = w_prev, h_prev
                w,h = int(p_old[2]), int(p_old[3])
                p = (int(x), int(y))
                dist2 = np.sqrt((p[0]-p_old[0])**2 + (p[1]-p_old[1])**2)
                if(dist2>thres):
                    x,y = int(p_old[0]), int(p_old[1])
                x, y, w, h = int(x-w/2), int(y-h/2), w, h
            else:
                posterior = kalman.correct(measurement)
                w_prev, h_prev = w, h
                x,y,_,_ = posterior
                x, y, w, h = int(x-w/2), int(y-h/2), w, h
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

        bbox = (x, y, w, h)
        p_old = (int(x+w/2), int(y+h/2), w, h)

        output.write("%d,%d,%d,%d\n" % bbox)

    cv2.destroyAllWindows()
    v.release()
    output.close()
    return fps

# define diretorios do dataset utilizado
dataset_dir = 'Dataset/'
# define o nomes dos videos utilizados
names = ['moving','illumination','rotations',
          'occlusion','occ_rotations','jumping']
# define os metodos a serem aplicados
methods = ['ms', 'ms_k', 'kalman', 'lk', 
           'camshift','kcf','kcf_lk', 'kcf_k']
output_dir = 'Results/' # diretorio dos resultados

# inicializa um dataframe para gerar tabelas
df = pd.DataFrame(index=methods,
                   columns=['E_m', 'E_sd', 'I_m', 'I_sd', 'TA', 'TE', 'FN', 'FPS'])
# para cada video iterar no loop
for name in names:
    print("Executando em: %s" % name)
    # para cada metodo iterar no loop
    for method in methods: 
        print("    %s" % method) 
        files_dir = dataset_dir + name + '/'
        video_dir = files_dir + name + '.mp4'
        v = cv2.VideoCapture(video_dir) # iniciar leitura do video
        if not v.isOpened():
            print("Could not open video")
            sys.exit()
        output_txt = output_dir + name + method + '.txt'
        fps = 0
        # interpreta e inicia funcao referente a cada metodo no loop, recebendo o fps
        if (method == 'kalman'):
            fps = Kalman_FT(v, output_txt)
            print("Kalman: %d", int(fps))
        elif (method == 'lk'):
            fps = LK_FT(v, output_txt)
            print("LK: %d", int(fps))
        elif (method == 'camshift'):
            fps = CamShift_FT(v, output_txt)
            print("CamShift: %d", int(fps))
        elif (method == 'kcf'):
            tracker = cv2.TrackerKCF_create()
            fps = Tracking(tracker, True, v, output_txt)
            print("KCF: %d", int(fps))
        elif (method == 'kcf_lk'):
            tracker = cv2.TrackerKCF_create()
            fps = KCL_LK(tracker, True, v, output_txt)
            print("KCL_LK: %d", int(fps))
        elif (method == 'kcf_k'):
            tracker = cv2.TrackerKCF_create()
            fps = KCL_Kalman(tracker, True, v, output_txt)
            print("KCL_Kalman: %d", int(fps))
        elif (method == 'ms'):
            fps = MS_FT(v, output_txt)
            print("MS_FT: %d", int(fps))
        elif (method == 'ms_k'):
            fps = MS_Kalman(v, output_txt)
            print("MS_Kalman: %d", int(fps))

        # abre arquivo com as marcacoes manuais dos videos, groundtruth (GT)
        file_gt = pd.read_csv(files_dir + "/groundtruth.txt", header=None, delimiter="\n")
        labels = np.array(file_gt[0].str.split(',', expand=True)) # interpreta o arquivo e salva GT como label
        # abre arquivo com as marcacoes acabadas de serem realizadas pelos metodos
        file_est = pd.read_csv(output_txt, header=None, delimiter="\n")
        estimates = np.array(file_est[0].str.split(',', expand=True)) # interpreta o arquivo e salva marcacoes como estimates
        
        i = 0
        E = []
        I = []
        FN = 0
        
        v = cv2.VideoCapture(video_dir) # reinicia video para comparacao do metodo com GT
        ret ,frame = v.read()
        
        x,y,w,h = (labels[i,:].astype(float)).astype(int) # pontos a serem identificados pelo GT
        x1,y1,w1,h1 = (estimates[i,:].astype(float)).astype(int) # pontos a serem identificados pelo metodo
        if((x1,y1,w1,h1) == (0,0,0,0)):
            FN+=1
        else:
            p = (int(x+w/2), int(y+h/2))
            p1 = (int(x1+w1/2), int(y1+h1/2))
            E.append(np.sqrt((p[0]-p1[0])**2 + (p[1]-p1[1])**2)) # calcula parte da metrica de erro
            I.append(bb_I([x,y,x+w,y+h],[x1,y1,x1+w1,y1+h1])) # calcula parte da intersecao pela uniao
            cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2) # desenha ROI do metodo (azul)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) # desenha ROI do metodo (verde)
        cv2.imshow('video',frame) # mostra resultados
        height, width, channels = frame.shape
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # escrita em um arquivo de video
        output_vid = "Results/Videos/" + name + "_" + method + '.mp4'
        out = cv2.VideoWriter(output_vid, fourcc, 15.0, (width, height))
        out.write(frame)
        
        for i in range(1, len(estimates)): # itera no vetor de estimativas
            ret ,frame = v.read() # le frame a frame
            x,y,w,h = (labels[i,:].astype(float)).astype(int)
            x1,y1,w1,h1 = (estimates[i,:].astype(float)).astype(int)
            if((x1,y1,w1,h1) == (0,0,0,0)): # caso medicao nao identificada, incrementa FN
                FN+=1
            else:
                p = (int(x+w/2), int(y+h/2))
                p1 = (int(x1+w1/2), int(y1+h1/2))
                E.append(np.sqrt((p[0]-p1[0])**2 + (p[1]-p1[1])**2))# calcula parte da metrica de erro
                I.append(bb_I([x,y,x+w,y+h],[x1,y1,x1+w1,y1+h1])) # calcula parte da intersecao pela uniao
                cv2.rectangle(frame, (x1,y1), (x1+w1, y1+h1), (255,0,0), 2) # desenha ROI do metodo (azul)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2) # desenha ROI do metodo (verde)
            out.write(frame) # escreve no arquivo
            cv2.imshow('video',frame) # mostra resultado
            if (cv2.waitKey(1) & 0xFF) == ord('q'): # permite comando de parada pela tecla q
                break
        
        out.release()
        cv2.destroyAllWindows()
        v.release()

        # calcula metricas resultantes
        E = np.array(E) 
        E_m = (np.sum(E))/len(E) # metrica de erro medio
        df['E_m'][method] = float('%.2f'%E_m) # coloca no dataframe
        E_sd = np.sqrt(np.sum(np.power(E - E_m, 2))/len(E)) # metrica do desvio padrao do erro
        df['E_sd'][method] = float('%.2f'%E_sd) # coloca no dataframe
        I = np.array(I)
        TA_tmp = np.copy(I)
        TE_tmp = np.copy(I)
        I_m = (np.sum(I))/len(I) # calcula intersecao pela uniao media
        df['I_m'][method] = float('%.2f'%I_m) # coloca no dataframe
        I_sd = np.sqrt(np.sum(np.power(I - I_m, 2))/len(I)) # calcula desvio padrao intersecao pela uniao
        df['I_sd'][method] = float('%.2f'%I_sd) # coloca no dataframe
        thres_H = 0.7 # threshold da taxa de acerto
        thres_M = 0.3 # threshold da taxa de erro
        TA_tmp = np.where(TA_tmp>=thres_H, 1, 0)
        TA = 100*((np.sum(TA_tmp))/len(TA_tmp)) # calcula taxa de acerto
        TE_tmp = np.where(TE_tmp<=thres_M, 1, 0)
        TE = 100*((np.sum(TE_tmp))/len(TE_tmp)) # calcula taxa de erro
        FN = 100*(FN/i)
        df['TA'][method] = float('%.2f'%TA) # coloca no dataframe
        df['TE'][method] = float('%.2f'%TE) # coloca no dataframe
        df['FN'][method] = float('%.2f'%FN) # coloca no dataframe
        df['FPS'][method] = int(fps)  # coloca no dataframe

    df.to_csv(output_dir + 'Tables/' + name + '.csv') # cria arquivo csv a partir do dataframe
