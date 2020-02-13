# -*- coding: utf-8 -*-
"""
===============================================================================
Numerical Image Deformation - v 1.0
===============================================================================

This script loads in a single speckle pattern image and applies a prescribed
deformation field to the pattern. The images are first super-sampled by a user-
defined amount (integer multiple of original resolution) prior to deforming the
image. The images are then downsampled to the original resolution. 

Dependent files:
This script has no dependency on functions aside from those defined here.

This script requires the following inputs (formats):
    i) speckle pattern image (.jpg)
    ii) user defined deformation field
    
Author: Jared Van Blitterswyk
Other contributors: N/A
Date created: 02.04.2020  

National Institute of Standards and Technology (NIST)
Materials Measurement Science Division
Material Measurement Laboratory  
"""
import os
from os import listdir
import numpy as np; import math as m
from matplotlib import pyplot as plt
from PIL import Image
import numpy.matlib
from scipy.interpolate import interp2d, griddata

os.chdir(r"Z:\Experiments\drop_tower\sa5_images")
# varify the path using getcwd() 
cwd = os.getcwd() 

filename = cwd+'/img_test2.jpg' # hardcode the location of the figure

img_ref = Image.open(filename) # open image

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

imarray = np.asarray(img_ref) # convert image to array

imarray_grey = rgb2gray(imarray)
img_ref = imarray_grey 
   
plt.pcolor(imarray_grey, cmap=plt.get_cmap('gray'))
plt.show()

Ny, Nx = imarray_grey.shape # find size of image
imarray = np.reshape(imarray_grey,Nx*Ny) # reshape to vector for griddata interpolation

#%% Up sample image to perform sub-pixel interpolation
sample_factor = 5 # integer scale factor for upsampled image resolution
x_orig_vec = np.linspace(0,Nx,Nx) # original x coordinates in vector form
y_orig_vec = np.linspace(0,Ny,Ny) # original y coordinates in vector form

x_orig_mesh,y_orig_mesh = np.meshgrid(x_orig_vec,y_orig_vec) # original coordinates in grid form

x_orig_meshV = np.reshape(x_orig_mesh,Nx*Ny) # original x coordinates from mesh in vector form for griddata interpolation
y_orig_meshV = np.reshape(y_orig_mesh,Nx*Ny) # original y coordinates from mesh in vector form for griddata interpolation

x_us_vec = np.linspace(0,Nx, num = Nx*sample_factor) # generate upsampled x coordinates (here based on pixels)
y_us_vec = np.linspace(0,Ny, num = Ny*sample_factor) # generate upsampled y coordinates (here based on pixels)

# create mesh of coordinates for plotting
x_us_mesh,y_us_mesh = np.meshgrid(x_us_vec,y_us_vec) # upsampled original coordinates in grid form

Ny_us, Nx_us = x_us_mesh.shape # shape (rows, vectors) of upsampled image

x_us_meshV = np.reshape(x_us_mesh,Nx_us*Ny_us) # upsampled original x coordinates from mesh in vector form for griddata interpolation
y_us_meshV = np.reshape(y_us_mesh,Nx_us*Ny_us) # upsampled original y coordinates from mesh in vector form for griddata interpolation

# interpolate image to upsampled grid
print('Upsampling image')
# upsample image using cubic grid interpolation
img_us_ref = griddata((x_orig_meshV,y_orig_meshV), imarray, (x_us_mesh, y_us_mesh), method ='cubic')

# plot diagnostic figures - original resolution reference image and upsampled reference image
fig1 = plt.figure() # create a figure with the default size 
plt.pcolor(img_ref, cmap = 'gray')
plt.title('original resolution reference image')
plt.colorbar()
plt.clim([0, 255])

fig2 = plt.figure()
plt.pcolor(img_us_ref, cmap = 'gray')
plt.title('upsampled reference image')
plt.colorbar()
plt.clim([0, 255])

#%%  define displacement fields and interpolate                       
# --------------------- define displacement fields ----------------------------
x_def = 10 #define constant x displacement
y_def = 0 #define constant y displacement

# ----------------------- define distortion field -----------------------------
# d^-1(x,y) = (1+p(x^2 + y^2))(1+pr^2)(x',y')
x_def = np.zeros(img_us_ref.shape)
y_def = np.zeros(img_us_ref.shape)

# move origin to centre of images to fit traditional distortion model
x_us_mesh_c = x_us_mesh-0.5*max(x_us_mesh[0,:])
y_us_mesh_c = y_us_mesh-0.5*max(y_us_mesh[:,0])

# distortion coefficients
# create approx. 1 pixel displacement radial distortion in x and y
k1 = -1.76e-10 # based on Table 1 in B. Pan et al. Systematic errors in two-dimensional digital image correlationdue to lens distortion, 2013 - adjusted for pixels instead of physical dimensions
k2 = 0

for i in range(0,Ny_us):
    for j in range(0,Nx_us):
        
        r = (x_us_mesh_c[i][j]**2 + y_us_mesh_c[i][j]**2) # radius of given point from centre of image
        
        x_def[i,j] = -1*x_us_mesh_c[i][j]*(k1*r**2 + k2*r**4) # inverse x displacement caused by distortion
        y_def[i,j] = -1*y_us_mesh_c[i][j]*(k1*r**2 + k2*r**4) # inverse y displacement caused by distortion         

# diagnostic figure
fig2 = plt.figure()
plt.pcolor(x_def, cmap = 'gray')
plt.title('upsampled x-displacement field')
plt.colorbar()
plt.clim([0, 255])

fig2 = plt.figure()
plt.pcolor(x_def, cmap = 'gray')
plt.title('upsampled y-displacement field')
plt.colorbar()
plt.clim([0, 255])
# -----------------------------------------------------------------------------
# ----------------------------------------------------------------------------- 

# create deformed coordinate matrices based on prescribed displacement fields      
x_us_mesh_def = x_us_mesh - x_def # shift the original upsampled x coordinates by the prescribed deformation
y_us_mesh_def = y_us_mesh - y_def # shift the original upsampled y coordinates by the prescribed deformation

x_us_mesh_defV = np.reshape(x_us_mesh_def,Nx_us*Ny_us) # deformed x coordinates from mesh in vector form for griddata interpolation 
y_us_mesh_defV = np.reshape(y_us_mesh_def,Nx_us*Ny_us) # deformed y coordinates from mesh in vector form for griddata interpolation

img_us_refV = np.reshape(img_us_ref,Nx_us*Ny_us) # reshape upsampled reference image for griddata interpolation

print('Interpolating to deformed positions')
# interpolate image to upsampled grid
img_us_def = griddata((x_us_mesh_defV,y_us_mesh_defV), img_us_refV, (x_us_mesh, y_us_mesh), method ='cubic')

# plot diagnostic figures - upsampled reference image and upsampled deformed image
fig1 = plt.figure() # create a figure with the default size 
ax1 = fig1.add_subplot(2,2,1) 
f1 = ax1.pcolor(img_us_ref)
ax1.set_title('upsampled reference image')
fig1.colorbar(f1, ax=ax1)

fig2 = plt.figure()
plt.pcolor(img_us_def, cmap = 'gray')
plt.title('upsampled deformed image')
plt.colorbar()
plt.clim([0, 255])

#%% down-sample to original image resolution
img_def = np.zeros((Ny,Nx))
for i in range(0,Ny):
    for j in range(0,Nx):
        ind_row1 = i*sample_factor
        ind_row2 = (i+1)*(sample_factor)-1
        ind_col1 = j*sample_factor
        ind_col2 = (j+1)*(sample_factor)-1
        
        if i ==0 and j ==0:
            print(ind_row1)
            print(ind_row2)
            print(ind_col1)
            print(ind_col2)
        elif i ==5 and j ==0:
            print(ind_row1)
            print(ind_row2)
            print(ind_col1)
            print(ind_col2)
        
        # average grey levels in upsampled image over upsampled window size (sample_factor*sample_factor) to return image to original resolution
        img_def[i,j] = np.mean(img_us_def[ind_row1:ind_row2,ind_col1:ind_col2])

# diagnostic figure - downsampled deformed image
fig3 = plt.figure()
plt.pcolor(img_def, cmap = 'gray')
plt.title('original resolution deformed image')
plt.colorbar()
plt.clim([0, 255])
