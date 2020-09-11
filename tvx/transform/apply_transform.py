#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 18:02:50 2020

@author: xiaoxiaoqi
"""
import SimpleITK as sitk
import numpy as np
import os



def apply_transform(reference, transform, image):
    """This command does the same thing as the antsApplyTransforms does (test passed.).
       The interpolator of the resampler can be changed.
       The reference should be a 3D itk image object.
       
       :type reference: sitkimage
       :type transform: sitktransform
       :type image: sitkimage
       :rtype: sitkimage
    """
    def _resample(reference, transform, image):
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetSize(reference.GetSize())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetTransform(transform)
        resampler.SetOutputPixelType(image.GetPixelID())
        img = resampler.Execute(image)   
        return img

    def _get3dslice(image, slice=0):
        size = list(image.GetSize())
        if len(size) == 4:
            size[3] = 0
            index = [0, 0, 0, slice]

            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex(index)
            image = extractor.Execute(image)
        return image
    # Resample 4D (SITK Doesn't support directly; so iterate through slice and get it done)

    transformed_images = []
    size = list(image.GetSize())
    
    if len(size)==4:
        for s in range(size[3]):
            img = _get3dslice(image, s)
            transformed = _resample(reference, transform, img) # the displacement works. 
            transformed_images.append(transformed)
        
        join = sitk.JoinSeriesImageFilter()
        transformed_final = join.Execute(transformed_images)
    
    elif len(size)==3:
        transformed_final = _resample(reference, transform, image)
    else:
        print("This function only supports 3D or 4D image.")
    
    return transformed_final

### Resample image to a giving spacing

def resample_image(itk_image, out_spacing=[0.5,0.5,0.5], is_label=False):
    """This function resample image
    
    :type itk_image: sitkimage
    :type out_spacing: array (3D or 4D)
    :type is_label: boolean
    :rtype: sitkimage
    """
    
    def _resample(itk_image, out_spacing, out_size, is_label=False):
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(out_spacing)
        resampler.SetSize(out_size)
        resampler.SetOutputOrigin(itk_image.GetOrigin())
        resampler.SetOutputDirection(itk_image.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetOutputPixelType(itk_image.GetPixelID())
        if is_label:
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        else:
            resampler.SetInterpolator(sitk.sitkLinear) # slow, using Linear for testing purpose, otherwise, BSpline
        
        img = resampler.Execute(itk_image)   
        return img
    
    def _get3dslice(image, slice=0):
        size = list(image.GetSize())
        if len(size) == 4:
            size[3] = 0
            index = [0, 0, 0, slice]

            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex(index)
            image = extractor.Execute(image)
        return image
    
    # Resample images to 0.5mm spacing with SimpleITK
    original_spacing = itk_image.GetSpacing()
    original_size = list(itk_image.GetSize())
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]

    resampled_images=[]
    if len(original_size)==4:
        for s in range(original_size[3]):
            img = _get3dslice(itk_image, s)
            resampled = _resample(img, out_spacing, out_size, is_label) # the displacement works. 
            resampled_images.append(resampled)
        
        join = sitk.JoinSeriesImageFilter()
        resampled_final = join.Execute(resampled_images)
    
    elif len(original_size)==3:
        resampled_final = _resample(itk_image, out_spacing, out_size, is_label)
    else:
        print("This function only supports 3D or 4D image.")
    
    
    return resampled_final

def transform_point(dwi_image, dwi_mask, bvec, bval, inv_transform, rotation_matrices, target_image):
    """This function transform point to physical space
    inputs: simple itk image/transform object. bvec and bval are file string. input images are should be in simpleITK image format.
    output: 1. index shows index of the first element in a voxel and number of elements per voxel
            2. the diffusion value image: n*1
            3. the direction image: n*3, array type.
            4. The bvec image: n*1, array type
    
    :type dwi_image: sitkimage
    :type dwi_mask: sitkimage
    :type bvec: string
    :type bval: string
    :type inv_transform: sitktransform
    :type rotation_matrices: sitkimage
    :type target_image: sitktransform
    :type out_file: string
    :rtype: dictionary
    """
    
    def _transform_point_index(i,j,k):
        """
        i,j,k are the x,y,z index of the input moving image
        output:tx,ty,tz are the x,y,z index of the transformed coordinates
        """
        xx,yy,zz,vv = dwi_image.TransformIndexToPhysicalPoint([i,j,k,0])
        px,py,pz = inv_transform.TransformPoint((xx,yy,zz))
        tx,ty,tz = target_image.TransformPhysicalPointToIndex([px,py,pz])
        # deal with nonsense values. Test codes has shown the valuse are at two ends.
        if tx <= 0 or tx >= dimx: tx = 0
        if ty <= 0 or ty >= dimy: ty = 0
        if tz <= 0 or tz >= dimz: tz = 0

        return tx,ty,tz
    
    def _transform_point_values(i,j,k,t, dwi_image_array, rotation_matrices_array):
        """i,j,k are the x,y,z index of the input moving image
        return the expecte values in the new position.
        :type i j k t: int
        :rtype dwibval: float
        :rtype dwibvec: array
        :rtype dwivalue: float
        """
        rotation_matrice = rotation_matrices_array[:,k,j,i]
        rotation = get_rotation_matrix(rotation_matrice)
        # use array to store values and vector.
        
        dwibval = bvals[t]
        dwibvec = rotate_vectors(bvecs[t],rotation)
        dwivalue = dwi_image_array[t,k,j,i]
        
        return dwibval, dwibvec, dwivalue
        
    # borrow function from dipy
    from dipy.io import read_bvals_bvecs  
    # :type:string, :rtype:array
    bvals,bvecs = read_bvals_bvecs(bval, bvec)
    
    # initiate the index_image in the beginning
    dimx, dimy, dimz, dimt = dwi_image.GetSize()
    tdimx,tdimy,tdimz = target_image.GetSize()

    dwi_image_array = sitk.GetArrayFromImage(dwi_image)
    dwi_mask_array = sitk.GetArrayFromImage(dwi_mask)
    rotation_matrices_array = sitk.GetArrayFromImage(rotation_matrices)


    # initiate output array    index_image1 = sitk.Image(tdimx, tdimy, tdimz, sitk.sitkInt16)
    index_image = np.zeros((2,tdimz,tdimy,tdimx))
    dwibvals = np.array([])
    dwibvecs = np.array([]).reshape(0,3)
    dwivalues = np.array([])


    # use a dictionary to store the transformed values.
    big_dict = {}
    for t,k,j,i in np.ndindex(dimt, dimz, dimy, dimx):
        if dwi_mask_array[t,k,j,i] > 0.5: ## a 4D mask array. If array has value, do computation.
            tx,ty,tz = _transform_point_index(i,j,k)
            dwibval, dwibvec, dwivalue = _transform_point_values(i,j,k,t, dwi_image_array, rotation_matrices_array)
            if (tx,ty,tz) not in big_dict:
                big_dict[(tx,ty,tz)] = [[],[],[]]
            voxel = big_dict[(tx,ty,tz)]
            voxel[0].append(dwibval)
            voxel[1].append(dwibvec)
            voxel[2].append(dwivalue)

    start_index = 0
    for k,j,i in np.ndindex(tdimz,tdimy,tdimx):
        if (i,j,k) in big_dict:
            size = len(big_dict[(i,j,k)][0])
            index_image[0,k,j,i] = start_index
            index_image[1,k,j,i] = size
            dwibvals=np.append(dwibvals, big_dict[(i,j,k)][0])
            dwibvecs=np.vstack(dwibvecs, big_dict[(i,j,k)][1])
            dwivalues=np.append(dwivalues, big_dict[(i,j,k)][2])
            start_index += size

    index_img = sitk.JoinSeries([sitk.GetImageFromArray(index_image[0],False), sitk.GetImageFromArray(index_image[1],False)])
    index_img.CopyInformation(target_image)

    # os.mkdir(out_file)
    # sitk.WriteImage(index_img, f"{out_file}/index.nii.gz")
    # np.save(dwibvals, f"{out_file}/dwibvals.npy")
    # np.save(dwibvals, f"{out_file}/dwibvecs.npy")
    # np.save(dwivalues, f"{out_file}/dwivalues.npy")

    return index_img, dwibvals, dwibvecs, dwivalues

def rotate_vectors(vector, rotation_matrix): 
    """This function rotate a vector using an 3x3 rotation_matrix
    
    :type vector: array
    :type rotation_matrix: array[array[]]
    :rtype: array
    """
    return np.dot(rotation_matrix, vector) # order matters
        
def get_rotation_matrix(rotation_matrice):
    """ This function decompose a value in rotation_matrices to an affine rotation matrix
    
    :type rotation_matrice: array 1d
    :rtype: array 2d
    """
    rotation = []
    rotation.append(rotation_matrice[0:3])
    rotation.append(rotation_matrice[3:6])
    rotation.append(rotation_matrice[6:9])
    return np.array(rotation)
        


