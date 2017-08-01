import cv2
from keras.models import load_model
import numpy as np
import time

cap = cv2.VideoCapture(0)
model = load_model("new_modeal24.h5")
model.summary()
count = 0
prev_class = 0
call_flag = False
print("\n"*30)
while True:
	count += 1
	count_padded = '/Users/kai/Desktop/keras2/kireishoes2/67/%05d' % count
	ret, frame = cap.read()

	frame = cv2.resize(frame, (100, 100))
	# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	a = model.predict(np.array([frame]), verbose=0)
	# print(a, np.argmax(a))
	now = np.argmax(a[0])
	# if now == prev_class and not call_flag:
	if now == 0:
		print("きれい！")
	else:
		print("きたない")
		# print("change", now)
		call_flag = True
	# elif prev_class != now:
		prev_class = now
		call_flag = False
	write_file_name = count_padded + ".jpg"
	#cv2.imwrite(write_file_name, frame)
	cv2.imshow('camera capture', frame)
	#print(write_file_name)
	time.sleep(4)

	#10msecキー入力待ち
	k = cv2.waitKey(10)
	#Escキーを押されたら終了
	if k == 27:
		break
#キャプチャを終了
cap.release()
cv2.destroyAllWindows()
