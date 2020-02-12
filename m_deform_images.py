# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:05:34 2020

@author: jcv
"""
import os
from os import listdir
import numpy as np; import math as m
from matplotlib import pyplot as plt
from PIL import Image
import numpy.matlib
from scipy.interpolate import interp2d, griddata

# Test git diff

os.chdir(r"Z:\Experiments\drop_tower\sa5_images")
# varify the path using getcwd() 
cwd = os.getcwd() 

filename = cwd+'/im_test.jpg'

img_ref = Image.open(filename)

imarray = np.asarray(img_ref) # convert image to array

Ny, Nx = imarray.shape # find size of image
imarray = np.reshape(imarray,Nx*Ny)

#%% Up sample image to perform sub-pixel interpolation
sample_factor = 10
x_orig_vec = np.linspace(0,Nx,Nx) # original x coordiantes in vector form
y_orig_vec = np.linspace(0,Ny,Ny) # original y coordiantes in vector form

x_orig_mesh,y_orig_mesh = np.meshgrid(x_orig_vec,y_orig_vec)

x_orig_meshV = np.reshape(x_orig_mesh,Nx*Ny)
y_orig_meshV = np.reshape(y_orig_mesh,Nx*Ny)

x_us_vec = np.linspace(0,Nx, num = Nx*sample_factor)
y_us_vec = np.linspace(0,Ny, num = Ny*sample_factor)

# create mesh of coordinates for plotting
x_us_mesh,y_us_mesh = np.meshgrid(x_us_vec,y_us_vec)

Ny_us, Nx_us = x_us_mesh.shape

x_us_meshV = np.reshape(x_us_mesh,Nx_us*Ny_us)
y_us_meshV = np.reshape(y_us_mesh,Nx_us*Ny_us)

# --- allocate memory for sorted variables ---
#im_array = np.zeros((2*Ny,2*Nx))

# interpolate image to upsampled grid
print('Upsampling image')
img_us_ref = griddata((x_orig_meshV,y_orig_meshV), imarray, (x_us_mesh, y_us_mesh), method ='cubic')

fig1 = plt.figure() # create a figure with the default size 
ax1 = fig1.add_subplot(2,2,1) 
f1 = ax1.pcolor(img_ref)
ax1.set_title('291 X 301')
fig1.colorbar(f1, ax=ax1)

im2 = np.random.rand(100,100)
ax2 = fig1.add_subplot(2,2,2)
f2 = ax2.imshow(img_us_ref, interpolation='none')
ax2.set_title('2910 x 3010')
fig1.colorbar(f2, ax=ax2)

#%%                         
# define displacement fields
#define constant displacement
x_def_const = 20
y_def_const = 0

x_us_mesh_def = x_us_mesh - x_def_const
y_us_mesh_def = y_us_mesh - y_def_const

x_us_mesh_defV = np.reshape(x_us_mesh_def,Nx_us*Ny_us)
y_us_mesh_defV = np.reshape(y_us_mesh_def,Nx_us*Ny_us)

img_us_refV = np.reshape(img_us_ref,Nx_us*Ny_us)

print('Interpolating to deformed positions')
# interpolate image to upsampled grid
img_us_def = griddata((x_us_mesh_defV,y_us_mesh_defV), img_us_refV, (x_us_mesh, y_us_mesh), method ='cubic')

fig1 = plt.figure() # create a figure with the default size 
ax1 = fig1.add_subplot(2,2,1) 
f1 = ax1.pcolor(img_us_ref)
ax1.set_title('291 X 301')
fig1.colorbar(f1, ax=ax1)

im2 = np.random.rand(100,100)
ax2 = fig1.add_subplot(2,2,2)
f2 = ax2.imshow(img_us_def, interpolation='none')
ax2.set_title('2910 x 3010')
fig1.colorbar(f2, ax=ax2)