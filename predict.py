from tensorflow.keras.models import load_model
from collections import deque
import numpy as np
import pickle
import cv2

model_path = ''
lb_path = ''
model = load_model(model_path)
lb = pickle.loads(open(lb_path, "rb").read())


mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=32)

video_path = ''
output_video_path = ''
vs = cv2.VideoCapture(video_path)
writer = None
(W, H) = (None, None)
print('Predicting ...')
while True:
	(grabbed, frame) = vs.read()
	if not grabbed:
		break
	if W is None or H is None:
		(H, W) = frame.shape[:2]
	output = frame.copy()
	# Preprocessing each frame in the video
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = cv2.resize(frame, (224, 224)).astype("float32")
	frame -= mean

	# Predicting label for each frame
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	# Taking rolling average of predictions to avoid flickering predictions
	Q.append(preds)
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]

	# Pasting the label on the output frame
	text = "activity: {}".format(label)
	cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 255, 0), 5)
	if writer is None:
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(output_video_path, fourcc, 30, (W, H), True)
		writer.write(output)
writer.release()
vs.release()