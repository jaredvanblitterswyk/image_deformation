# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:02:32 2020

@author: jcv
"""
import numpy as np; import math as m
from scipy import interpolate
from scipy.interpolate import interp2d, griddata

# load in displacement fields and generate image stack stored in img_def

def deform_images(img_us_ref,coords_us,coords_ref,disp):
    # create deformed coordinate matrices based on prescribed displacement fields      
    x_us_mesh_def = coords_us.x_mesh - disp.x # shift the original upsampled x coordinates by the prescribed deformation
    y_us_mesh_def = coords_us.y_mesh - disp.y # shift the original upsampled y coordinates by the prescribed deformation
    
    x_us_mesh_defV = np.reshape(x_us_mesh_def,coords_us.Nx*coords_us.Ny) # deformed x coordinates from mesh in vector form for griddata interpolation 
    y_us_mesh_defV = np.reshape(y_us_mesh_def,coords_us.Nx*coords_us.Ny) # deformed y coordinates from mesh in vector form for griddata interpolation
    
    img_us_refV = np.reshape(img_us_ref,coords_us.Nx*coords_us.Ny) # reshape upsampled reference image for griddata interpolation
    
    print('Interpolating to deformed positions')
    # interpolate image to upsampled grid
    img_us_def = griddata((x_us_mesh_defV,y_us_mesh_defV), img_us_refV, (coords_us.x_mesh, coords_us.y_mesh), method ='cubic') # cubic interpolation
    
    return img_us_def