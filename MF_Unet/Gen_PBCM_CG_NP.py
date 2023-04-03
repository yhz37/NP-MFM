#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 09:15:52 2022

@author: y
"""

import numpy as np
import matlab.engine

def Gen_PBCM_CG_NP(samples,eng,case):

    Config = {}
    Config['m']=float(100)
    Config['DetectorL']= float(400)
    Config['FS_Num'] = float(150)
    Config['DP'] = float(2)
    
    if '_9' in case:
        Config['ProbType'] = 'mCGG_9'
    if '_7' in case:
        Config['ProbType'] = 'mCGG_7'
        Config['DefP'] = matlab.double([0.758891170957360,0.936400148471394,0.984074836887003]);

    PBCM_samples = eng.Gen_PBCM_CG_NP(matlab.double(samples),Config)
    PBCM_samples = np.asarray(PBCM_samples).transpose()
    
    return PBCM_samples