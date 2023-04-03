# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 14:51:44 2022

@author: haizhouy
"""

import time
import os
import numpy as np
import matlab.engine
import subprocess


def Gen_CFD_CG_Local(sample,eng,case):
    global Num,sim_path_timestr,timestr
    if 'Num' in globals():
        Num = Num+1
    else:        
        Num = 1
        timestr = time.strftime('%Y%m%d_%H%M%S')
        # sim_path = '/home/y/y/Python/MFNN/Simulation_File'
        # sim_path_timestr = sim_path+'/'+timestr
        sim_path = 'D:\OneDrive - University of South Carolina\Research\Microfluidic Concentration Gradient Generators\CFD_Starccm'
        sim_path_timestr = sim_path+'\\'+timestr
        os.makedirs(sim_path_timestr)
    os.chdir('D:\OneDrive - University of South Carolina\Research\Microfluidic Concentration Gradient Generators\CFD_Starccm')
    sim_path_timestr_num = sim_path_timestr+'\\''mCGG_Starccm_Sim_{:04d}'.format(Num)
    os.makedirs(sim_path_timestr_num)
    
    if '_9' in case:
        Input = eng.Point2input_9_P(matlab.double(sample))
    elif '_7' in case:
        point = np.concatenate((sample[0]*np.array([0.758891170957360,0.936400148471394,0.984074836887003]),sample[1:]), axis=1)
        Input = eng.Point2input_9_P(matlab.double(point))
    Input = np.asarray(Input)
    num_processor = 10
    
    fp = open('Auto_Boundary_Sim.java', 'w')
    Javascript ='''// STAR-CCM+ macro: Auto_Boundary_Sim.java
    // Written by STAR-CCM+ 13.02.013
    package macro;
    import java.util.*;
    import star.common.*;
    import star.base.neo.*;
    import star.passivescalar.*;
    import star.flow.*;
    public class Auto_Boundary_Sim extends StarMacro {
      public void execute() {
        execute0();
      }
      private void execute0() {
        Simulation simulation_0 = 
          getActiveSimulation();
        Region region_0 = 
          simulation_0.getRegionManager().getRegion("FLUID");
        Boundary boundary_0 = 
          region_0.getBoundaryManager().getBoundary("INLET_1");
        PassiveScalarProfile passiveScalarProfile_0 = 
          boundary_0.getValues().get(PassiveScalarProfile.class);
        passiveScalarProfile_0.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));
        TotalPressureProfile totalPressureProfile_0 = 
          boundary_0.getValues().get(TotalPressureProfile.class);
        totalPressureProfile_0.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);
        Boundary boundary_1 = 
          region_0.getBoundaryManager().getBoundary("INLET_2");
        PassiveScalarProfile passiveScalarProfile_1 = 
          boundary_1.getValues().get(PassiveScalarProfile.class);
        passiveScalarProfile_1.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));
        TotalPressureProfile totalPressureProfile_1 = 
          boundary_1.getValues().get(TotalPressureProfile.class);
        totalPressureProfile_1.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);
        Boundary boundary_2 = 
          region_0.getBoundaryManager().getBoundary("INLET_3");
        PassiveScalarProfile passiveScalarProfile_2 = 
          boundary_2.getValues().get(PassiveScalarProfile.class);
        passiveScalarProfile_2.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));
        TotalPressureProfile totalPressureProfile_2 = 
          boundary_2.getValues().get(TotalPressureProfile.class);
        totalPressureProfile_2.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);
        Boundary boundary_3 = 
          region_0.getBoundaryManager().getBoundary("INLET_4");
        PassiveScalarProfile passiveScalarProfile_3 = 
          boundary_3.getValues().get(PassiveScalarProfile.class);
        passiveScalarProfile_3.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));
        TotalPressureProfile totalPressureProfile_3 = 
          boundary_3.getValues().get(TotalPressureProfile.class);
        totalPressureProfile_3.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);
        Boundary boundary_4 = 
          region_0.getBoundaryManager().getBoundary("INLET_5");
        PassiveScalarProfile passiveScalarProfile_4 = 
          boundary_4.getValues().get(PassiveScalarProfile.class);
        passiveScalarProfile_4.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));
        TotalPressureProfile totalPressureProfile_4 = 
          boundary_4.getValues().get(TotalPressureProfile.class);
        totalPressureProfile_4.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);
        Boundary boundary_5 = 
          region_0.getBoundaryManager().getBoundary("INLET_6");
        PassiveScalarProfile passiveScalarProfile_5 = 
          boundary_5.getValues().get(PassiveScalarProfile.class);
        passiveScalarProfile_5.getMethod(ConstantArrayProfileMethod.class).getQuantity().setArray(new DoubleVector(new double[] {%.15e}));
        TotalPressureProfile totalPressureProfile_5 = 
          boundary_5.getValues().get(TotalPressureProfile.class);
        totalPressureProfile_5.getMethod(ConstantScalarProfileMethod.class).getQuantity().setValue(%.15e);
        Solution solution_0 = 
          simulation_0.getSolution();
        solution_0.clearSolution();
        solution_0.initializeSolution();
        ResidualPlot residualPlot_0 = 
          ((ResidualPlot) simulation_0.getPlotManager().getPlot("Residuals"));
        residualPlot_0.open();
        simulation_0.getSimulationIterator().runAutomation();
        PlotUpdate plotUpdate_0 = 
          residualPlot_0.getPlotUpdate();
        HardcopyProperties hardcopyProperties_0 = 
          plotUpdate_0.getHardcopyProperties();
        hardcopyProperties_0.setCurrentResolutionWidth(1292);
        hardcopyProperties_0.setCurrentResolutionHeight(432);
        XYPlot xYPlot_0 = 
          ((XYPlot) simulation_0.getPlotManager().getPlot("Detector Conc"));
        xYPlot_0.open();
        hardcopyProperties_0.setCurrentResolutionWidth(1294);
        hardcopyProperties_0.setCurrentResolutionHeight(433);
        PlotUpdate plotUpdate_1 = 
          xYPlot_0.getPlotUpdate();
        HardcopyProperties hardcopyProperties_1 = 
          plotUpdate_1.getHardcopyProperties();
        hardcopyProperties_1.setCurrentResolutionWidth(1292);
        hardcopyProperties_1.setCurrentResolutionHeight(432);
        Cartesian2DAxisManager cartesian2DAxisManager_0 = 
          ((Cartesian2DAxisManager) xYPlot_0.getAxisManager());
        cartesian2DAxisManager_0.setAxesBounds(new Vector(Arrays.asList(new star.common.AxisManager.AxisBounds("Bottom Axis", -4.442899953573942E-4, false, 7.577299838885665E-4, false), new star.common.AxisManager.AxisBounds("Left Axis", -3.409309578684831E-65, false, 9.105821793778221E-54, false))));
        xYPlot_0.export(resolvePath("mCGG_Sim_%04d.csv"), ",");
        simulation_0.saveState(resolvePath("TriY_mCGG_%04d.sim"));
      }
    }
    ''' % (Input[:,1],Input[:,0],Input[:,3],Input[:,2],Input[:,5],Input[:,4],Input[:,7],Input[:,6],Input[:,9],Input[:,8],Input[:,11],Input[:,10],Num,Num)

    fp.write(Javascript)
    fp.close()
    sysout= subprocess.check_output('starccm+ -batch Auto_Boundary_Sim.java TriY_mCGG.sim -np %d'%(num_processor), shell=True)

    move_file(Num,sim_path_timestr_num)

    CG = extract_CG(Num,sim_path_timestr_num)
    return CG




    
def move_file(Num,sim_path_timestr_num):
    
    os.replace("mCGG_Sim_{:04d}.csv".format(Num), sim_path_timestr_num+'\\'+"mCGG_Sim_{:04d}.csv".format(Num))
    
    os.replace("Auto_Boundary_Sim.java", sim_path_timestr_num+'\\'+ 'Auto_Boundary_Sim.java')
    
    
def extract_CG(Num,sim_path_timestr):
    with open(sim_path_timestr+'\mCGG_Sim_{:04d}.csv'.format(Num),'r') as csv_file:
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

