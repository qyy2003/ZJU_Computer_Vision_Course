import cv2
import numpy as np
import matplotlib.pyplot as plt

filepath = "points_py.yml"
# cv_file = cv2.FileStorage(filepath, cv2.FILE_STORAGE_READ)
# points = cv_file.getNode("points").mat()
# gap = cv_file.getNode("gap").real()
import yaml
with open(filepath, 'r') as f:
    file_content = f.read()
    content = yaml.load(file_content, yaml.FullLoader)
# print(content)
points = np.array(content['points'])
gap = content['gap']
x=range(gap,201,gap)
plt.plot(x,points)
plt.xlabel('PC numbers')
plt.ylabel('Rank-1 rate')
# plt.show()
plt.savefig('rank1_pc.png')
# print(points,gap)