# -*- coding: utf-8 -*-
import skimage.io
import numpy as np
import cv2
import pycuda.autoinit
import pycuda.driver as drv
import pycuda.gpuarray as gpuarray
import numpy as np
import skcuda.linalg as linalg
import skcuda.misc as misc
from pycuda.elementwise import ElementwiseKernel
import pycuda.cumath as cumath
import sys, getopt
import time


input_mul = sys.argv[1]
input_pan = sys.argv[2]
output_fus = sys.argv[3]

# Kernel establecido para llevar a cabo el mÃ©todo de rescale global min
lin_comb = ElementwiseKernel(
        "float a, float *x, float b, float *z",
        "z[i] = ((x[i]-a)*255)/(b-a)",
        "linear_combination")


# El step_1 realiza la multiplicaci�n elemento a elemento de una banda con la pancrom�tica
def step_1(color_matrix, image_matrix):
    #La funciÃ³n linalg.mulitply realiza la multiplicaciÃ³n elemento a elemento entre dos matrices
    matrix_sal = linalg.multiply(color_matrix, image_matrix)
    return matrix_sal



# El step_2 determina el valor maximo y minimo de la banda resultante de la transformada de multiplicación
def step_2(matrix_1):
    mat_max = np.amax(matrix_1)
    mat_min = np.amin(matrix_1)
    return mat_max, mat_min


def step_3(matrix_1, matrix_color, mat_max, mat_min):
    #La funciÃ³n lin_comb invoca el kernel elementwise establecido para llevar a cabo el ajuste por rescale global min
    lin_comb(mat_min, matrix_1, mat_max, matrix_color)
    return matrix_color.get()


end = 0
start = 0

# Lee la imagen multiespectral y la pancromatica
m = skimage.io.imread(input_mul, plugin='tifffile')
p = skimage.io.imread(input_pan, plugin='tifffile')

#Verifica que ambas imagenes cumplan con las condiciones
if m.shape[2]:
    print 'la imagen multiespectral tiene '+str(m.shape[2])+' bandas y tamaño '+str(m.shape[0])+'x'+str(m.shape[1])
else:
    print 'la primera imagen no es multiespectral'

if len(p.shape) == 2:
    print 'la imagen Pancromatica tiene tamaño '+str(p.shape[0])+'x'+str(p.shape[1])
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

r1_gpu = gpuarray.to_gpu(r1)
g1_gpu = gpuarray.to_gpu(g1)
b1_gpu = gpuarray.to_gpu(b1)
p1_gpu = gpuarray.to_gpu(p1)

linalg.init()

m33_gpu = step_1(r1_gpu, p1_gpu)
m44_gpu = step_1(g1_gpu, p1_gpu)
m55_gpu = step_1(b1_gpu, p1_gpu)

Amax, Amin = step_2(m33_gpu.get())
br_gpu = gpuarray.empty_like(r1_gpu)
br_host = step_3(m33_gpu, br_gpu, Amax, Amin)

Amax, Amin = step_2(m44_gpu.get())
bg_gpu = gpuarray.empty_like(g1_gpu)
bg_host = step_3(m44_gpu, bg_gpu, Amax, Amin)

Amax, Amin = step_2(m55_gpu.get())
bb_gpu = gpuarray.empty_like(b1_gpu)
bb_host = step_3(m55_gpu, bb_gpu, Amax, Amin)

end = time.time()

brr = br_host.astype(np.uint8)
bgg = bg_host.astype(np.uint8)
bbb = bb_host.astype(np.uint8)

brgb = np.stack((brr, bgg, bbb))

tiempo = (end - start)
print tiempo

t = skimage.io.imsave('/home/nvera/andres/multiplicative/'+output_fus+'.tif', brgb, plugin='tifffile')
