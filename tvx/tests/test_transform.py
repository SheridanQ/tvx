#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:51:16 2020

@author: xiaoxiaoqi
"""
import numpy as np
from numpy.testing import assert_array_equal

from tvx.transform.apply_transform import get_rotation_matrix

def test_get_rotation_matrix():
    rotation_matrice = np.array([ 0.99755382,-0.01439499,-0.06840445,0.03211438,0.9635597,\
              0.26555859,0.06208906,-0.26710575,0.96166494])
    rotation = get_rotation_matrix(rotation_matrice)
    
    expected_shape = (3,3)
    
    assert_array_equal(rotation.shape, expected_shape)

def test_sort_big_dict():
    d = {(0,0):[1,2],(0,1):[1,2,3],(1,0):[1,2,3],(1,1):[4,5,6]}
    one_list = []
    start_index = 0
    index_list = []
    count_list = []
    for i,j in np.ndindex(2,2):
        index_list.append(start_index)
        count_list.append(len(d[(i,j)]))
        one_list.extend(d[(i,j)])
        start_index+=len(d[(i,j)])

    expected_one_list = [1,2,1,2,3,1,2,3,4,5,6]
    expected_index_list = [0,2,5,8]
    expected_count_list = [2,3,3,3]
    assert_array_equal(expected_one_list, one_list)
    assert_array_equal(expected_index_list, index_list)
    assert_array_equal(expected_count_list, count_list)

def test_append_array():
    vector0 = np.array([]).reshape(0,3)
    vector1 = [np.array([1,2,3]),np.array([4,5,6])]
    vector2 = [np.array([4,5,6])]

    vector1 = np.vstack((vector0,vector1))
    final_array = np.vstack((vector1,vector2))
    size = final_array.shape
    expected_size = (3,3)
    assert_array_equal(size, expected_size)

