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
import matplotlib as matplib
from matplotlib import pyplot as plt
from PIL import Image
import pandas as pd
import numpy.matlib
from scipy.interpolate import interp2d, griddata
import scipy
from f_generate_deformed_image_stack import deform_images

os.chdir(r"Z:\Python\image_deformation")
# varify the path using getcwd() 
cwd = os.getcwd() 

image_dir = 'Z:/Python/image_deformation/rigid_body_translation' # directory where main images will be stored

image_name_prefix = 'hc_dic_' # prefix of image name - will be followed by descriptor (i.e: rigid body tranlsation of 1 pixel in x, 0 in y would be image_name_prefix_10x_00y)

image_name_ref = image_name_prefix+'00x_00y.tif' # name of reference image
# ---------------------- load in reference image ------------------------------
#filename = image_dir+'/hc_reference_dic_challenge_1.tif' # hardcode the location of the figure

img_ref = Image.open(image_dir+'/'+image_name_ref) # open image
'''
# for rgb images
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

imarray = np.asarray(img_ref) # convert image to array

imarray_grey = rgb2gray(imarray)
img_ref = imarray_grey 
'''
imarray_grey = np.asarray(img_ref) # greylevel images

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
print('Upsampling reference image...')
# upsample image using cubic grid interpolation
img_us_ref = griddata((x_orig_meshV,y_orig_meshV), imarray, (x_us_mesh, y_us_mesh), method ='cubic')
print('Complete.')
# plot diagnostic figures - original resolution reference image and upsampled reference image
'''
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
'''

#%%  define displacement fields and interpolate                       
# ----------------- define rigid-body displacement fields ---------------------
# sub-pixel shift up to 1 pixel
num_def_steps = 2
rows,cols = img_us_ref.shape
x_def = np.zeros((rows,cols,num_def_steps))
y_def = np.zeros((rows,cols,num_def_steps))

for i in range(0,num_def_steps):
    print((i+1)/num_def_steps)
    x_def[:,:,i] = (i+1)/num_def_steps*np.ones((rows,cols))
    

'''
# ----------------------- define distortion field -----------------------------
# d^-1(x,y) = (1+p(x^2 + y^2))(1+pr^2)(x',y')

x_def = np.zeros(img_us_ref.shape)
y_def = np.zeros(img_us_ref.shape)

# move origin to centre of images to fit traditional distortion model
x_us_mesh_c = x_us_mesh-0.5*max(x_us_mesh[0,:])
y_us_mesh_c = y_us_mesh-0.5*max(y_us_mesh[:,0])

# distortion coefficients
# create approx. 1 pixel displacement radial distortion in x and y
k1 = -1.76e-12 # based on Table 1 in B. Pan et al. Systematic errors in two-dimensional digital image correlation due to lens distortion, 2013 - adjusted for pixels instead of physical dimensions
k2 = 0

for i in range(0,Ny_us):
    for j in range(0,Nx_us):
        
        r = (x_us_mesh_c[i][j]**2 + y_us_mesh_c[i][j]**2) # radius of given point from centre of image
        
        x_def[i,j] = -1*x_us_mesh_c[i][j]*(k1*r**2 + k2*r**4) # inverse x displacement caused by distortion
        y_def[i,j] = -1*y_us_mesh_c[i][j]*(k1*r**2 + k2*r**4) # inverse y displacement caused by distortion     
'''
# ---------------------- define displacement field ----------------------------
'''
x_def = 0.5*np.sin(2*m.pi*x_us_mesh/Nx)
y_def = np.zeros(img_us_ref.shape)

# diagnostic figure
fig2 = plt.figure()
plt.pcolor(x_def, cmap = 'gray')
plt.title('upsampled x-displacement field')
plt.colorbar()

fig2 = plt.figure()
plt.pcolor(y_def, cmap = 'gray')
plt.title('upsampled y-displacement field')
plt.colorbar()
'''
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
class coords_us:
    pass
coords_us = coords_us()

coords_us.x_mesh = x_us_mesh
coords_us.y_mesh = y_us_mesh
coords_us.x = x_us_vec
coords_us.y = y_us_vec
coords_us.Nx = Nx_us
coords_us.Ny = Ny_us

class coords_ref:
    pass

coords_ref = coords_ref()
coords_ref.x_mesh = x_orig_mesh
coords_ref.y_mesh = y_orig_mesh
coords_ref.x = x_orig_vec
coords_ref.y = y_orig_vec
coords_ref.Nx = Nx
coords_ref.Ny = Ny

# an empty class of deformation fields
class disp:
    pass

disp = disp()
disp.x = x_def
disp.y = y_def

print('Numerically deforming images...')
img_us_def = deform_images(img_us_ref,coords_us,coords_ref,disp,num_def_steps)
print('Complete.')

# define strings describing displacements for each image
disp_inc_str_x = ["{0:02}".format(int(np.mean(np.mean(disp.x[:,:,i]*10)))) for i in range(0,num_def_steps)]
disp_inc_str_y = ["{0:02}".format(int(np.mean(np.mean(disp.y[:,:,i]*10)))) for i in range(0,num_def_steps)]

# save images to file
print('Saving images to file...')
for i in range(0,num_def_steps):
    print('Image: '+str(i+1)+' of '+str(num_def_steps)+'...')
    filename = image_dir+'/'+image_name_prefix+disp_inc_str_x[i]+'x_'+disp_inc_str_y[i]+'y.tiff'
    print(filename)
    
    im = Image.fromarray(img_us_def[:,:,i].astype(np.uint8))
    im.save(filename)

'''
# create deformed coordinate matrices based on prescribed displacement fields      
x_us_mesh_def = x_us_mesh - x_def # shift the original upsampled x coordinates by the prescribed deformation
y_us_mesh_def = y_us_mesh - y_def # shift the original upsampled y coordinates by the prescribed deformation

x_us_mesh_defV = np.reshape(x_us_mesh_def,Nx_us*Ny_us) # deformed x coordinates from mesh in vector form for griddata interpolation 
y_us_mesh_defV = np.reshape(y_us_mesh_def,Nx_us*Ny_us) # deformed y coordinates from mesh in vector form for griddata interpolation

img_us_refV = np.reshape(img_us_ref,Nx_us*Ny_us) # reshape upsampled reference image for griddata interpolation

print('Interpolating to deformed positions')
# interpolate image to upsampled grid
img_us_def = griddata((x_us_mesh_defV,y_us_mesh_defV), img_us_refV, (x_us_mesh, y_us_mesh), method ='cubic')
'''
# plot diagnostic figures - upsampled reference image and upsampled deformed image
'''
fig1 = plt.figure()
plt.pcolor(img_us_ref, cmap = 'gray')
plt.title('upsampled reference image')
plt.colorbar()
plt.clim([0, 255])

fig2 = plt.figure()
plt.pcolor(img_us_def, cmap = 'gray')
plt.title('upsampled deformed image')
plt.colorbar()
plt.clim([0, 255])
'''

#%% down-sample to original image resolution
img_def = np.zeros((Ny,Nx))
for i in range(0,Ny):
    for j in range(0,Nx):
        ind_row1 = i*sample_factor
        ind_row2 = (i+1)*(sample_factor)-1
        ind_col1 = j*sample_factor
        ind_col2 = (j+1)*(sample_factor)-1
        
        # average grey levels in upsampled image over upsampled window size (sample_factor*sample_factor) to return image to original resolution
        img_def[i,j] = np.mean(img_us_def[ind_row1:ind_row2,ind_col1:ind_col2])

# diagnostic figure - downsampled deformed image
fig3 = plt.figure()
plt.pcolor(img_def, cmap = 'gray')
plt.title('original resolution deformed image')
plt.colorbar()
plt.clim([0, 255])
