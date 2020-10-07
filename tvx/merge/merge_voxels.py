#!/usr/bin/env python3
"""

@author: xiaoxiaoqi
"""
import SimpleITK as sitk
import numpy as np

def split_trarry(index_img, dwibvals, dwibvecs, dwivalues, batch_size, outdir):
    """
    This function merge
    :param trarry: string
    :param batch: int
    :param outdir: string
    :return: a folder
    """

    size = index_img.Get_Size()
    length = size[0] * size[1] * size[2]



    for i in range()
