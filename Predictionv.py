# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 13:38:35 2020

@author: Dr Clement Etienam
"""

# example of loading a pix2pix model and using it for one-off image translation
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import load
from numpy import expand_dims
import matplotlib.pyplot as plt
import numpy as np
import cv2
# load an image
def load_image(filename, size=(256,256)):
	# load image with the preferred size
	pixels = load_img(filename, target_size=size)
	# convert to numpy array
	pixels = img_to_array(pixels)
	# scale from [0,255] to [-1,1]
	pixels = (pixels - 127.5) / 127.5
	# reshape to 1 sample
	pixels = expand_dims(pixels, 0)
	return pixels

# load source image
src_image = load_image('48.jpg')
print('Loaded', src_image.shape)
# load model
model = load_model('model_005000.h5')
# generate image from source
gen_image = model.predict(src_image)
# scale from [-1,1] to [0,1]
gen_image = (gen_image + 1) / 2.0
# plot the image
plt.figure(figsize =(20,20))
plt.subplot(2,2,3)
plt.imshow(gen_image[0])
plt.title('Generated Thermal', fontsize = 20)
plt.axis('off')


plt.subplot(2,2,1)
im = cv2.imread('48.jpg')
im_resized = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR)
plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
plt.title('RGB', fontsize = 20)
plt.axis('off')


plt.subplot(2,2,2)
im = cv2.imread('48b.jpg')
im_resized = cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR)
plt.imshow(cv2.cvtColor(im_resized, cv2.COLOR_BGR2RGB))
plt.title('True Thermal', fontsize = 20)
plt.axis('off')


plt.show()

plt.figure(figsize =(20,20))
original_image = cv2.imread("48.jpg")
img=cv2.cvtColor(original_image,cv2.COLOR_BGR2RGB)
vectorized = img.reshape((-1,3))
vectorized = np.float32(vectorized)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts=10
ret,label,center=cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2= res.reshape((img.shape))



edges = cv2.Canny(img,150,200)


figure_size = 15
plt.figure(figsize=(figure_size,figure_size))
plt.subplot(2,2,1),plt.imshow(img)
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2,2,2),plt.imshow(res2)
plt.title('Segmented Image when K = %i' % K), plt.xticks([]), plt.yticks([])

plt.subplot(2,2,3),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
plt.show()



