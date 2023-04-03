#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 23 23:31:14 2022

@author: y
"""

import paramiko
import time
import os
import numpy as np
import matlab.engine


def Gen_CFD_CG_WS16(sample,eng,case):
    global Num,sim_path_timestr,timestr
    if 'Num' in globals():
        Num = Num+1
    else:        
        Num = 1
        timestr = time.strftime('%Y%m%d_%H%M%S')
        sim_path = '/home/y/y/Python/MFNN/Simulation_File'
        sim_path_timestr = sim_path+'/'+timestr
        os.makedirs(sim_path_timestr)
    if '_9' in case:
        Input = eng.Point2input_9_P(matlab.double(sample))
    elif '_7' in case:
        point = np.concatenate((sample[0]*np.array([0.758891170957360,0.936400148471394,0.984074836887003]),sample[1:]), axis=1)
        Input = eng.Point2input_9_P(matlab.double(point))
    Input = np.asarray(Input)
    results = []
    num_processor = 50
    SimCommand ='''
                touch Auto_Boundary_Sim.java
                echo '// STAR-CCM+ macro: Auto_Boundary_Sim.java'>>Auto_Boundary_Sim.java
                echo '// Written by STAR-CCM+ 13.02.013'>>Auto_Boundary_Sim.java
                echo 'package macro;'>>Auto_Boundary_Sim.java
                echo 'import java.util.*;'>>Auto_Boundary_Sim.java
                echo 'import star.common.*;'>>Auto_Boundary_Sim.java
                echo 'import star.base.neo.*;'>>Auto_Boundary_Sim.java
                echo 'import star.passivescalar.*;'>>Auto_Boundary_Sim.java
                echo 'import star.flow.*;'>>Auto_Boundary_Sim.java
                echo 'public class Auto_Boundary_Sim extends StarMacro {'>>Auto_Boundary_Sim.java
                echo '  public void execute() {'>>Auto_Boundary_Sim.java
                echo '    execute0();'>>Auto_Boundary_Sim.java
                echo '  }'>>Auto_Boundary_Sim.java
                echo '  private void execute0() {'>>Auto_Boundary_Sim.java
                echo '    Simulation simulation_0 = '>>Auto_Boundary_Sim.java
                echo '      getActiveSimulation();'>>Auto_Boundary_Sim.java
                echo '    Region region_0 = '>>Auto_Boundary_Sim.java
                echo '      simulation_0.getRegionManager().getRegion("FLUID");'>>Auto_Boundary_Sim.java
                echo '    Boundary boundary_0 = '>>Auto_Boundary_Sim.java
                echo '      region_0.getBoundaryManager().getBoundary("INLET_1");'>>Auto_Boundary_Sim.java
                echo '    PassiveScalarProfile passiveScalarProfile_0 = '>>Auto_Boundary_Sim.java
                echo '      boundary_0.getValues().get(PassiveScalarProfile.class);'>>Auto_Boundary_Sim.java
                echo '    passiveScalarProfile_0.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));'>>Auto_Boundary_Sim.java
                echo '    TotalPressureProfile totalPressureProfile_0 = '>>Auto_Boundary_Sim.java
                echo '      boundary_0.getValues().get(TotalPressureProfile.class);'>>Auto_Boundary_Sim.java
                echo '    totalPressureProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);'>>Auto_Boundary_Sim.java
                echo '    Boundary boundary_1 = '>>Auto_Boundary_Sim.java
                echo '      region_0.getBoundaryManager().getBoundary("INLET_2");'>>Auto_Boundary_Sim.java
                echo '    PassiveScalarProfile passiveScalarProfile_1 = '>>Auto_Boundary_Sim.java
                echo '      boundary_1.getValues().get(PassiveScalarProfile.class);'>>Auto_Boundary_Sim.java
                echo '    passiveScalarProfile_1.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));'>>Auto_Boundary_Sim.java
                echo '    TotalPressureProfile totalPressureProfile_1 = '>>Auto_Boundary_Sim.java
                echo '      boundary_1.getValues().get(TotalPressureProfile.class);'>>Auto_Boundary_Sim.java
                echo '    totalPressureProfile_1.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);'>>Auto_Boundary_Sim.java
                echo '    Boundary boundary_2 = '>>Auto_Boundary_Sim.java
                echo '      region_0.getBoundaryManager().getBoundary("INLET_3");'>>Auto_Boundary_Sim.java
                echo '    PassiveScalarProfile passiveScalarProfile_2 = '>>Auto_Boundary_Sim.java
                echo '      boundary_2.getValues().get(PassiveScalarProfile.class);'>>Auto_Boundary_Sim.java
                echo '    passiveScalarProfile_2.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));'>>Auto_Boundary_Sim.java
                echo '    TotalPressureProfile totalPressureProfile_2 = '>>Auto_Boundary_Sim.java
                echo '      boundary_2.getValues().get(TotalPressureProfile.class);'>>Auto_Boundary_Sim.java
                echo '    totalPressureProfile_2.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);'>>Auto_Boundary_Sim.java
                echo '    Boundary boundary_3 = '>>Auto_Boundary_Sim.java
                echo '      region_0.getBoundaryManager().getBoundary("INLET_4");'>>Auto_Boundary_Sim.java
                echo '    PassiveScalarProfile passiveScalarProfile_3 = '>>Auto_Boundary_Sim.java
                echo '      boundary_3.getValues().get(PassiveScalarProfile.class);'>>Auto_Boundary_Sim.java
                echo '    passiveScalarProfile_3.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));'>>Auto_Boundary_Sim.java
                echo '    TotalPressureProfile totalPressureProfile_3 = '>>Auto_Boundary_Sim.java
                echo '      boundary_3.getValues().get(TotalPressureProfile.class);'>>Auto_Boundary_Sim.java
                echo '    totalPressureProfile_3.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);'>>Auto_Boundary_Sim.java
                echo '    Boundary boundary_4 = '>>Auto_Boundary_Sim.java
                echo '      region_0.getBoundaryManager().getBoundary("INLET_5");'>>Auto_Boundary_Sim.java
                echo '    PassiveScalarProfile passiveScalarProfile_4 = '>>Auto_Boundary_Sim.java
                echo '      boundary_4.getValues().get(PassiveScalarProfile.class);'>>Auto_Boundary_Sim.java
                echo '    passiveScalarProfile_4.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));'>>Auto_Boundary_Sim.java
                echo '    TotalPressureProfile totalPressureProfile_4 = '>>Auto_Boundary_Sim.java
                echo '      boundary_4.getValues().get(TotalPressureProfile.class);'>>Auto_Boundary_Sim.java
                echo '    totalPressureProfile_4.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);'>>Auto_Boundary_Sim.java
                echo '    Boundary boundary_5 = '>>Auto_Boundary_Sim.java
                echo '      region_0.getBoundaryManager().getBoundary("INLET_6");'>>Auto_Boundary_Sim.java
                echo '    PassiveScalarProfile passiveScalarProfile_5 = '>>Auto_Boundary_Sim.java
                echo '      boundary_5.getValues().get(PassiveScalarProfile.class);'>>Auto_Boundary_Sim.java
                echo '    passiveScalarProfile_5.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));'>>Auto_Boundary_Sim.java
                echo '    TotalPressureProfile totalPressureProfile_5 = '>>Auto_Boundary_Sim.java
                echo '      boundary_5.getValues().get(TotalPressureProfile.class);'>>Auto_Boundary_Sim.java
                echo '    totalPressureProfile_5.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);'>>Auto_Boundary_Sim.java
                echo '    Solution solution_0 = '>>Auto_Boundary_Sim.java
                echo '      simulation_0.getSolution();'>>Auto_Boundary_Sim.java
                echo '    solution_0.clearSolution();'>>Auto_Boundary_Sim.java
                echo '    solution_0.initializeSolution();'>>Auto_Boundary_Sim.java
                echo '    ResidualPlot residualPlot_0 = '>>Auto_Boundary_Sim.java
                echo '      ((ResidualPlot) simulation_0.getPlotManager().getPlot("Residuals"));'>>Auto_Boundary_Sim.java
                echo '    residualPlot_0.open();'>>Auto_Boundary_Sim.java
                echo '    simulation_0.getSimulationIterator().runAutomation();'>>Auto_Boundary_Sim.java
                echo '    PlotUpdate plotUpdate_0 = '>>Auto_Boundary_Sim.java
                echo '      residualPlot_0.getPlotUpdate();'>>Auto_Boundary_Sim.java
                echo '    HardcopyProperties hardcopyProperties_0 = '>>Auto_Boundary_Sim.java
                echo '      plotUpdate_0.getHardcopyProperties();'>>Auto_Boundary_Sim.java
                echo '    hardcopyProperties_0.setCurrentResolutionWidth(1292);'>>Auto_Boundary_Sim.java
                echo '    hardcopyProperties_0.setCurrentResolutionHeight(432);'>>Auto_Boundary_Sim.java
                echo '    XYPlot xYPlot_0 = '>>Auto_Boundary_Sim.java
                echo '      ((XYPlot) simulation_0.getPlotManager().getPlot("Detector Conc"));'>>Auto_Boundary_Sim.java
                echo '    xYPlot_0.open();'>>Auto_Boundary_Sim.java
                echo '    hardcopyProperties_0.setCurrentResolutionWidth(1294);'>>Auto_Boundary_Sim.java
                echo '    hardcopyProperties_0.setCurrentResolutionHeight(433);'>>Auto_Boundary_Sim.java
                echo '    PlotUpdate plotUpdate_1 = '>>Auto_Boundary_Sim.java
                echo '      xYPlot_0.getPlotUpdate();'>>Auto_Boundary_Sim.java
                echo '    HardcopyProperties hardcopyProperties_1 = '>>Auto_Boundary_Sim.java
                echo '      plotUpdate_1.getHardcopyProperties();'>>Auto_Boundary_Sim.java
                echo '    hardcopyProperties_1.setCurrentResolutionWidth(1292);'>>Auto_Boundary_Sim.java
                echo '    hardcopyProperties_1.setCurrentResolutionHeight(432);'>>Auto_Boundary_Sim.java
                echo '    Cartesian2DAxisManager cartesian2DAxisManager_0 = '>>Auto_Boundary_Sim.java
                echo '      ((Cartesian2DAxisManager) xYPlot_0.getAxisManager());'>>Auto_Boundary_Sim.java
                echo '    cartesian2DAxisManager_0.setAxesBounds(new Vector(Arrays.asList(new star.common.AxisManager.AxisBounds("Bottom Axis", -4.442899953573942E-4, false, 7.577299838885665E-4, false), new star.common.AxisManager.AxisBounds("Left Axis", -3.409309578684831E-65, false, 9.105821793778221E-54, false))));'
                echo '    xYPlot_0.export(resolvePath("mCGG_Sim_%04d.csv"), ",");'>>Auto_Boundary_Sim.java
                echo '  }'>>Auto_Boundary_Sim.java
                echo '}'>>Auto_Boundary_Sim.java
                /opt/Siemens/15.02.009-R8/STAR-CCM+15.02.009-R8/star/bin/starccm+ -batch Auto_Boundary_Sim.java TriY_mCGG.sim -np %d
                ''' % (Input[:,1],Input[:,0],Input[:,3],Input[:,2],Input[:,5],Input[:,4],Input[:,7],Input[:,6],Input[:,9],Input[:,8],Input[:,11],Input[:,10],Num,num_processor)

    client,ssh_stdin,ssh_stdout,ssh_stderr = ssh_conn(SimCommand)
    get_file(client,Num,sim_path_timestr)
    move_file(client,Num,timestr)
    for i in results:
        print(i.strip())
    CG = extract_CG(Num,sim_path_timestr)
    return CG

def ssh_conn(SimCommand):
    client  = paramiko.SSHClient()
    client.load_system_host_keys()
    client.connect('10.72.3.191', username='y')
    ssh_stdin, ssh_stdout, ssh_stderr = client.exec_command(SimCommand)
    results = []
    for line in ssh_stderr:
        results.append(line.strip('\n'))   
    return client,ssh_stdin,ssh_stdout,ssh_stderr

def get_file(client,Num,sim_path_timestr):
    ftp_client=client.open_sftp()
    ftp_client.get('/home/y/mCGG_Sim_{:04d}.csv'.format(Num),sim_path_timestr+'/mCGG_Sim_{:04d}.csv'.format(Num))
    ftp_client.close()
    
def move_file(client,Num,timestr):
    sim_path_timestr_remote = '/home/y/Simulation_File/'+timestr+'/mCGG_Starccm_Sim_{:04d}'.format(Num)
    
    movecommand ='''
                    mkdir -p {}
                    mv mCGG_Sim_{:04d}.csv {}
                    mv Auto_Boundary_Sim.java {}
                ''' .format(sim_path_timestr_remote,Num,sim_path_timestr_remote,sim_path_timestr_remote)
    ssh_stdin, ssh_stdout, ssh_stderr = client.exec_command(movecommand)
    
def extract_CG(Num,sim_path_timestr):
    with open(sim_path_timestr+'/mCGG_Sim_{:04d}.csv'.format(Num),'r') as csv_file:
        next(csv_file)
        CG = np.loadtxt(csv_file, delimiter=",")

    for i in [0,2,4,6,8,10,12]:
        CG[:,i:i+2] = CG[:,i:i+2][CG[:,i:i+2][:, 0].argsort()]
        
    CG_All = CG[:,[1,3,5,7,9,11,13]]
    CG_All = np.mean(CG_All,axis = 1)
    CG_All = CG_All.reshape([1,-1])

    # xConc = np.linspace(0, 1, num=100)
    # for i in range(7):
    #     plt.plot(xConc,CG[:,2*i+1])
    # plt.show()
    return CG_All    

