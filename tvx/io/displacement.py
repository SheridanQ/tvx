#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:54:32 2020

@author: xiaoxiaoqi
"""
import SimpleITK as sitk
import numpy as np
import os

def get_displacement(displacement_field):
    """This function get displacement field transform from displacement field image
    
    :type displacement_field:sitkimage
    :rtype: sitktransform
    """
    
    displacementtransform=sitk.DisplacementFieldTransform(3)
    displacementtransform.SetFixedParameters(displacement_field.GetSize() + displacement_field.GetOrigin() + \
                                             displacement_field.GetSpacing() + displacement_field.GetDirection())
    parameters=np.ndarray.tolist(np.ndarray.flatten(sitk.GetArrayFromImage(displacement_field)))
    displacementtransform.SetParameters(parameters)
    return displacementtransform