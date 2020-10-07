#!/usr/bin/env python3
"""
@author: xiaoxiaoqi
"""
import os

import SimpleITK as sitk
import numpy as np

from tvx.io.image import read_image
from tvx.io.displacement import get_displacement
from tvx.transform.apply_transform import transform_point, resample_image

## Read images
dwi_image = read_image("/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/dwi/babu_8-dwis.nii.gz")
dwi_mask_image = read_image("/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/dwi/babu_8-mask.nii.gz")
dp_image = read_image("/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/displacement_field/1mm/babu_8_fod_composedwarp.nii.gz")
inv_dp_image = read_image("/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/displacement_field/1mm/babu_8_fod_composedinvwarp.nii.gz")
ref_image = read_image("/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/target/new_atlas_our_65_95_static_BrainCerebellum.nii.gz")

## Read bvecs and bvals
bvecs = "/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/dwi/babu_8.bvecs"
bvals = "/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/dwi/babu_8.bvals"

## Read displacement
transform = get_displacement(dp_image)
inv_transform = get_displacement(inv_dp_image)

## Read Rotation
rotation_matrices = read_image("/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/displacement_field/1mm/babu_8_rotation_inv.nii.gz")

outputdir = "/Users/xiaoxiaoqi/PycharmProjects/qball_csd/test_data/test_results/babu_8"
if not os.path.isdir(outputdir):
    os.mkdir(outputdir)
#%%
up_dwi_image = resample_image(dwi_image, out_spacing = [1,1,1])
up_dwi_mask_image = resample_image(dwi_mask_image, out_spacing = [1,1,1], is_label=True)
#%%
#index_img, dwibvals, dwibvecs, dwivalues = transform_point(dwi_image, dwi_mask_image, bvecs, bvals, inv_transform, rotation_matrices, ref_image)
big_dict, index_img = transform_point(dwi_image, dwi_mask_image, bvecs, bvals, inv_transform, rotation_matrices, ref_image)


sitk.WriteImage(index_img, f"{outputdir}/index_inv_dwi.nii.gz")
# np.save(dwibvals, f"{outputdir}/dwibvals.npy")
# np.save(dwibvals, f"{outputdir}/dwibvecs.npy")
# np.save(dwivalues, f"{outputdir}/dwivalues.npy")


# import json
# with open(f'{outputdir}/data_inv_noref.json','w') as fp:
#     json.dump(big_dict, fp)



