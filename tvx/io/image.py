#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 17:06:10 2020

@author: xiaoxiaoqi
"""
import SimpleITK as sitk
import numpy as np
import os

def read_image(path):
    """This function only output the header without loading the bulk image
    
    :type path: string
    :rtype: sitkimage
    """
    reader = sitk.ImageFileReader()
    reader.SetImageIO("NiftiImageIO")
    reader.SetFileName(path)
    return reader.Execute()

def write_image(image, path):
    """This function writes image from sitkimage to nifti
    
    :type image: sitkimage
    :type path: string
    :rtype: None
    """
    writer = sitk.ImageFileWriter()
    writer.SetFileName(path)
    writer.Execute(image)
    return

def print_info(image):
    """This function print info of sitkimage
    
    :type image:sitkimage
    :rtype: None
    """
    print("Dimension:{}".format(image.GetDimension()))
    print("Direction:{}".format(image.GetDirection()))
    print("Spacing:{}".format(image.GetSpacing()))
    print("Origin:{}".format(image.GetOrigin()))
    print("Size:{}".format(image.GetSize()))
    print("PixelNo:{}".format(image.GetNumberOfPixels()))

def subsample_image(image, tuple_range, tuple_size):
    i,j,k,t =
    sub_image = np.zeros(tuple_size)
    tdim = sub_dwi_image.shape[0]