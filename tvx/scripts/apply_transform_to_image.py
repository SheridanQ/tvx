#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 14:16:09 2020

@author: xiaoxiaoqi
"""
import os
import sys
import json

import SimpleITK as sitk
import numpy as np

from tvx.io.image import read_image
from tvx.io.displacement import get_displacement
from tvx.transform.apply_transform import transform_point

dwipath = sys.argv[1]
dwimaskpath = sys.argv[2]

bvecpath = sys.argv[3]
bvalpath = sys.argv[4]

inv_dp_path = sys.argv[5]
ref_path = sys.argv[6]
rotationpath = sys.argv[7]

outputdir = sys.argv[8]


dwi_image = read_image(dwipath)
dwi_mask_image = read_image(dwimaskpath)

ref_image = read_image(ref_path)
rotation_matrices = read_image

inv_dp_image = read_image(inv_dp_path)
inv_transform = get_displacement(inv_dp_image)

if not os.path.isdir(outputdir):
    os.mkdir(outputdir)
    
bvals_dict, bvecs_dict, dwivalues_dict, index1, index2 = transform_point(dwi_image, dwi_mask_image, bvecpath, bvalpath, \
                                                                         inv_transform, rotation_matrices, ref_image)
    
sitk.WriteImage(index1, f"{outputdir}/checking_image.nii.gz")
sitk.WriteImage(index2, f"{outputdir}/stats_image.nii.gz")
with open(f'{outputdir}/bvals.json','w') as fp:
    json.dump(bvals_dict, fp)
with open(f'{outputdir}/bvecs.json','w') as fp:
    json.dump(bvals_dict, fp)
with open(f'{outputdir}/dwivalues.json','w') as fp:
    json.dump(bvals_dict, fp)
