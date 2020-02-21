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


os.chdir(r"Z:\Python\image_deformation")
# varify the path using getcwd() 
cwd = os.getcwd() 

from f_generate_deformed_image_stack import deform_images

image_dir = 'Z:/Python/image_deformation/rigid_body_translation' # directory where main images will be stored

#image_name_prefix = 'hc_dic_' # prefix of image name - will be followed by descriptor (i.e: rigid body tranlsation of 1 pixel in x, 0 in y would be image_name_prefix_10x_00y)
image_name_prefix = 'hc_dic_reu_ref1' # prefix of image name
#image_name_ref = image_name_prefix+'00x_00y.tiff' # name of reference image
image_name_ref = image_name_prefix+'.tiff' # name of reference image
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
sample_factor = 10 # integer scale factor for upsampled image resolution
x_orig_vec = np.linspace(0,Nx-1,Nx) # original x coordinates in vector form
y_orig_vec = np.linspace(0,Ny-1,Ny) # original y coordinates in vector form

x_orig_mesh,y_orig_mesh = np.meshgrid(x_orig_vec,y_orig_vec) # original coordinates in grid form

x_orig_meshV = np.reshape(x_orig_mesh,Nx*Ny) # original x coordinates from mesh in vector form for griddata interpolation
y_orig_meshV = np.reshape(y_orig_mesh,Nx*Ny) # original y coordinates from mesh in vector form for griddata interpolation

x_us_vec = np.linspace(0,Nx-1, num = Nx*sample_factor+1) # generate upsampled x coordinates (here based on pixels)
y_us_vec = np.linspace(0,Ny-1, num = Ny*sample_factor+1) # generate upsampled y coordinates (here based on pixels)

# create mesh of coordinates for plotting
x_us_mesh,y_us_mesh = np.meshgrid(x_us_vec,y_us_vec) # upsampled original coordinates in grid form

Ny_us, Nx_us = x_us_mesh.shape # shape (rows, vectors) of upsampled image

x_us_meshV = np.reshape(x_us_mesh,Nx_us*Ny_us) # upsampled original x coordinates from mesh in vector form for griddata interpolation
y_us_meshV = np.reshape(y_us_mesh,Nx_us*Ny_us) # upsampled original y coordinates from mesh in vector form for griddata interpolation
'''
# interpolate image to upsampled grid
print('Upsampling reference image...')
# upsample image using cubic grid interpolation
img_us_ref = griddata((x_orig_meshV,y_orig_meshV), imarray, (x_us_mesh, y_us_mesh), method ='linear')
print('Complete.')

'''
# -----------------------------------------------------------------------------
# ---------- instead of interpolating, assign values to upsample image --------
# this will work better if shifting pixels by discrete amounts, but not for more general deformations
print('Upsampling reference image...')
img_us_ref = np.zeros((Ny_us,Nx_us))
for i in range(0,Ny):
    for j in range(0,Nx):
        ind_row1 = i*sample_factor
        ind_row2 = (i+1)*sample_factor # indexing: [start:stop] - start through stop -1
        ind_col1 = j*sample_factor
        ind_col2 = (j+1)*sample_factor
        
        # average grey levels in upsampled image over upsampled window size (sample_factor*sample_factor) to return image to original resolution
        img_us_ref[ind_row1:ind_row2,ind_col1:ind_col2] = imarray_grey[i,j]
        
print('Complete.')
'''
# interpolate with zero shift        
print('Numerically deforming images...')
img_us_def = deform_images(img_us_ref,coords_us,coords_ref,disp,num_def_steps)
print('Complete.')
'''
# -----------------------------------------------------------------------------

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
print('Numerically deforming images...')
img_ref = np.zeros((Ny,Nx))
for i in range(0,Ny):
    for j in range(0,Nx):
        ind_row1 = i*sample_factor
        ind_row2 = (i+1)*(sample_factor)
        ind_col1 = j*sample_factor
        ind_col2 = (j+1)*(sample_factor)
        
        # average grey levels in upsampled image over upsampled window size (sample_factor*sample_factor) to return image to original resolution
        img_ref[i,j] = np.mean(img_us_ref[ind_row1:ind_row2,ind_col1:ind_col2])
        
# save reference image to file
print('Saving images to file...')
print('Image: 0...')
filename = image_dir+'/'+image_name_prefix+'_s.tiff'
print(filename)

im = Image.fromarray(img_ref[:,:].astype(np.uint8))
im.save(filename)
#%%  define displacement fields and interpolate                       
# ----------------- define rigid-body displacement fields ---------------------
# sub-pixel shift up to 1 pixel
num_def_steps = sample_factor

rows,cols = img_us_ref.shape
img_us_def = np.zeros((rows,cols,num_def_steps))

print('Numerically deforming images...')
for i in range(0,num_def_steps):
    print('Image: '+str(i+1)+' of '+str(num_def_steps)+'...')
    img_us_def[:,0:Nx_us-i-1,i] = img_us_ref[:,i+1:Nx_us]
    
print('Complete.')    


# define strings describing displacements for each image
disp_inc_str_x = ["{0:02}".format(int(i+1)) for i in range(0,num_def_steps)]
disp_inc_str_y = ["{0:02}".format(int(i+1)) for i in range(0,num_def_steps)]


#%% down-sample to original image resolution
img_def = np.zeros((Ny,Nx,num_def_steps))
for k in range(0,num_def_steps):
    for i in range(0,Ny):
        for j in range(0,Nx):
            ind_row1 = i*sample_factor
            ind_row2 = (i+1)*(sample_factor)
            ind_col1 = j*sample_factor
            ind_col2 = (j+1)*(sample_factor)
            
            # average grey levels in upsampled image over upsampled window size (sample_factor*sample_factor) to return image to original resolution
            img_def[i,j,k] = np.mean(img_us_def[ind_row1:ind_row2,ind_col1:ind_col2,k])
        
       
# save images to file
print('Saving images to file...')
for i in range(0,num_def_steps):
    print('Image: '+str(i+1)+' of '+str(num_def_steps)+'...')
    filename = image_dir+'/'+image_name_prefix+disp_inc_str_x[i]+'x_'+disp_inc_str_y[i]+'y.tiff'
    print(filename)
    
    im = Image.fromarray(img_def[:,:,i].astype(np.uint8))
    im.save(filename)

'''        

# diagnostic figure - downsampled deformed image
fig3 = plt.figure()
plt.pcolor(img_def, cmap = 'gray')
plt.title('original resolution deformed image')
plt.colorbar()
plt.clim([0, 255])
'''
