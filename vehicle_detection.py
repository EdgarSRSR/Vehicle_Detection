import cv2
import numpy as np
import imutils

cap = cv2.VideoCapture('cvtest.avi')

fgbg = cv2.createBackgroundSubtractorMOG2()
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
cv2.destroyAllWindows()
while True:
	ret, frame = cap.read()
	if ret == False: break
	frame = imutils.resize(frame, width=640)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # rgb в grayscale
	cv2.rectangle(frame,(0,0),(frame.shape[1],40),(0,0,0),-1) # нарисует черный прямоугольник, чтобы разместить текст, описывающий статус обнаружения движения
	color = (0, 255, 0)
	text_state = "Транспорт не обнаружен"
	#tl, tr, br, bl
	# указываем область, которая будет проанализирована в видео
	area_pts = np.array([[272,125], [385,125], [364,345], [0,295]])
    # вспомогательное изображение для анализа, это будет нулевой массив
	imAux = np.zeros(shape=(frame.shape[:2]), dtype=np.uint8)
	imAux = cv2.drawContours(imAux, [area_pts], -1, (255), -1)
	image_area = cv2.bitwise_and(gray, gray, mask=imAux)
	# создаем маску региона, чтобы представить движение
	fgmask = fgbg.apply(image_area)
	# уменьшить шум в маске двоичного изображения
	fgmask = cv2.dilate(fgmask, None, iterations=2)
	# находим существующие контуры в fgmask, чтобы определить движение на основе площади контуров, это уменьшает количество ложных срабатываний
	cnts = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
	for cnt in cnts:
		if cv2.contourArea(cnt) > 500:
			x, y, w, h = cv2.boundingRect(cnt)
			# создает зеленый прямоугольник для обнаруженного контура
			cv2.rectangle(frame, (x,y), (x+w, y+h),(0,255,0), 2)
			text_state = "Внимание, обнаружен Транспорт!"
			color = (0, 0, 255)
	cv2.drawContours(frame, [area_pts], -1, color, 2) # рисуем контур анализируемой области
	cv2.putText(frame, text_state , (10, 30), # текст с информацией об обнаружении движения
    	cv2.FONT_HERSHEY_COMPLEX, 1, color,2)
	cv2.imshow("frame", frame) # замедляет видео для удобства работы с ним, предыдущее значение 30
	k = cv2.waitKey(70) & 0xFF
	if k == 27:
		break
cap.release()
cv2.destroyAllWindows()    