#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 23:05:19 2022

@author: ouj
"""

from numba import cuda
import ctypes
import numpy as np
import torch

def devndarray2torch(dev_arr):
    t = torch.empty(size=dev_arr.shape, dtype=torch.float32).cuda()
    ctx = cuda.cudadrv.devices.get_context()
	
    # constant value of #bytes in case of float32 = 4
    mp = cuda.cudadrv.driver.MemoryPointer(ctx, ctypes.c_ulong(t.data_ptr()), t.numel()*4)
    tmp_arr = cuda.cudadrv.devicearray.DeviceNDArray(t.size(), [i*4 for i in t.stride()], np.dtype('float32'), 
                                            gpu_data=mp, stream=torch.cuda.current_stream().cuda_stream)
											
    # To verify whether the data pointer is same or not.
    # print(tmp_arr.__cuda_array_interface__)
    # print(dev_arr.__cuda_array_interface__)
    tmp_arr.copy_to_device(dev_arr)
    return t

# d_arr = cuda.to_device(np.array([[10,20,30],[40,50,60.0]], dtype=np.float32))
# tensor = devndarray2torch(d_arr)
# print(tensor)