# -*- coding: utf-8 -*-
import skimage.io
import numpy as np
import cv2
import time
import sys, getopt

input_mul = sys.argv[1]
input_pan = sys.argv[2]
output_fus = sys.argv[3]

# El step_1 realiza la multiplicación elemento a elemento de una banda con la pancromática
def step_1(color_matrix, image_matrix):
    input_matrix = np.multiply(color_matrix,image_matrix)
    return input_matrix


# El step_2 determina el valor maximo y minimo de la banda resultante de la transformada de brovey
def step_2(matrix_1):
    mat_max = np.amax(matrix_1)
    mat_min = np.amin(matrix_1)
    return mat_max, mat_min


def step_3(matrix_1, mat_max, mat_min):
    matrix_color = np.empty_like(matrix_1)
    for m in range(matrix_1.shape[0]):
        for n in range(matrix_1.shape[0]):
            matrix_color[m,n] = (((matrix_1[m,n]-mat_min)*255)/(mat_max-mat_min))
    return matrix_color

end = 0
start = 0

# Lee la imagen multiespectral y la pancromatica
m = skimage.io.imread(input_mul, plugin='tifffile')
p = skimage.io.imread(input_pan, plugin='tifffile')

#Verifica que ambas imagenes cumplan con las condiciones
if m.shape[2]:
    print 'la imagen multiespectral tiene '+str(m.shape[2])+' bandas y tamaÃ±o '+str(m.shape[0])+'x'+str(m.shape[1])
else:
    print 'la primera imagen no es multiespectral'

if len(p.shape) == 2:
    print 'la imagen Pancromatica tiene tamaÃ±o '+str(p.shape[0])+'x'+str(p.shape[1])
else:
    print 'la segunda imagen no es pancromatica'


m1 = m.astype(np.float32)
r = m[:,:,0]
r1 = r.astype(np.float32)
g = m[:,:,1]
g1 = g.astype(np.float32)
b = m[:,:,2]
b1 = b.astype(np.float32)
p1 = p.astype(np.float32)

start = time.time()

m33 = step_1(r1, p1)
m44 = step_1(g1, p1)
m55 = step_1(b1, p1)

Amax, Amin = step_2(m33)
br = step_3(m33, Amax, Amin)

Amax, Amin = step_2(m44)
bg = step_3(m44, Amax, Amin)

Amax, Amin = step_2(m55)
bb = step_3(m55, Amax, Amin)

end = time.time()

brr = br.astype(np.uint8)
bgg = bg.astype(np.uint8)
bbb = bb.astype(np.uint8)

brgb = np.stack((brr, bgg, bbb))

tiempo = (end - start)
print tiempo

t = skimage.io.imsave('/home/nvera/andres/multiplicative/'+output_fus+'.tif', brgb, plugin='tifffile')
