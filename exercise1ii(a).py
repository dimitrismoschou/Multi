import cv2
import numpy as np
from scipy.stats import entropy

cap = cv2.VideoCapture('../auxiliary2021/original_movies/golf.avi')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

buffer = []
size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

framesPerSeconds = 30
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
out = cv2.VideoWriter('../auxiliary2021/final_movies/exercise1ii/golf(a).avi', fourcc , framesPerSeconds , size ,False)

def entropy1(labels, base=None):
  value,counts = np.unique(labels, return_counts=True)
  return entropy(counts, base=base)

fc = 0
ret, r_frame = cap.read()
r_frame = cv2.cvtColor(r_frame, cv2.COLOR_BGR2GRAY)
while cap.isOpened():
    ret, c_frame = cap.read()
    if not ret:
        break
    c_frame = cv2.cvtColor(c_frame, cv2.COLOR_BGR2GRAY)
    out.write(np.subtract(c_frame, r_frame))


    try:
        image = np.subtract(c_frame, r_frame)
        buffer.append(image.astype('int16')) #see "continue" at line 16
        fc = fc + 1
    except :
        continue #stores the dtype for some reason and we don't want that
    r_frame = c_frame
print('The video has been created.'+ '\n entropy = '+ str(entropy1(buffer)))

cap.release()
out.release()
cv2.destroyAllWindows()





