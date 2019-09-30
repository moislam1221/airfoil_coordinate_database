#!usr/bin/env python

import os
import sys
import time
import itertools
import traceback
import subprocess
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pylab

basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def extract_coordinates(coordPath, fname):
    xProfile = []
    yProfile = []
    with open(coordPath, 'r') as f:
        next(f)
        for line in f:
            coordinates = line.split()
            if (fname == 's1223rtl'):
                print(coordinates)
            x = float(coordinates[0])
            y = float(coordinates[1])
            xProfile.append(x) 
            yProfile.append(y)
    
    return xProfile, yProfile

def plot_airfoil(xProfile, yProfile, imagePath, fname):
    pylab.figure()
    pylab.plot(xProfile, yProfile)
    pylab.title(fname + ' Profile', fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.axis('equal')
    pylab.savefig(imagePath)
    pylab.close()

def plot_profile_failing(fname):
    if fname.endswith('.dat'): fname = fname[:-4]
    coordPath = os.path.join(basepath, 'problematic_coordinates', fname + '.dat')
    imagePath = os.path.join(basepath, 'images', 'problematic_airfoils', fname + '.png')
    xProfile, yProfile = extract_coordinates(coordPath, fname)
    plot_airfoil(xProfile, yProfile, imagePath, fname)

def plot_profile_working(fname):
    if fname.endswith('.dat'): fname = fname[:-4]
    coordPath = os.path.join(basepath, 'coordinates', fname + '.dat')
    imagePath = os.path.join(basepath, 'images', 'working_airfoils', fname + '.png')
    xProfile, yProfile = extract_coordinates(coordPath, fname)
    plot_airfoil(xProfile, yProfile, imagePath, fname)
    
if __name__ == '__main__':
    # Plot failing airfoils in the coordinates directory
    filesProblematic = sorted(os.listdir(os.path.join(basepath, 'problematic_coordinates')))
    nAirfoilsProblematic = 35
    for i in range(0, nAirfoilsProblematic):
        plot_profile_failing(filesProblematic[i])
    # Plot working airfoils in the coordinates directory
    filesWorking = sorted(os.listdir(os.path.join(basepath, 'coordinates')))
    nAirfoilsWorking = 1516
    for i in range(0, nAirfoilsWorking):
        plot_profile_working(filesWorking[i])

    #pool = multiprocessing.Pool()
    #pool.map(plot_profile, itertools.product(files[0:2]))
