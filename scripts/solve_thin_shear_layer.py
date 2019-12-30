import os
import itertools
import multiprocessing
from numpy import *
import scipy.interpolate
from scipy import integrate
import pylab

basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

## Setup ## 
def preprocess_args(args):
    fname, Re = args
    if fname.endswith('.dat'): fname = fname[:-4]
    subpath = os.path.join(basepath, 'profiles', fname, str(Re))
    return fname, Re, subpath

## Pre-processing functions for U(y) ##
def extract_stagnation_point(s, Ue):
   
    # Identify the index of the points right before/after the stagnation point 
    idx_stagnation = 0 
    for idx in range(0, len(s)-1):
        if Ue[idx] * Ue[idx+1] < 0:
            idx_before = idx
            idx_after = idx+1
    
    # Identify stagnation coordinate s_stag and ds of adjacent points closest to stagantion points (useful for PDE solve)
    s_before = s[idx_before]
    s_after = s[idx_after]
    factor = Ue[idx_before] / (Ue[idx_before] + abs(Ue[idx_after]))
    s_stagnation = s_before + factor * abs(s_after - s_before)
    ds_to_before = abs(s_stagnation - s_before)

    return s_before, idx_before, ds_to_before

def compute_constant_term_1storder(Ue_curr, Ue_ahead, ds):

    # Obtain adjacent points velocities and positions (in preparation for derivatives)
    dUe_dx = Ue_curr *(Ue_ahead - Ue_curr) / (-ds)

    return dUe_dx

def compute_constant_term(idx, s, Ue):

    # Obtain adjacent points velocities and positions (in preparation for derivatives)
    U_0 = Ue[idx]
    U_plus = Ue[idx+1]
    U_minus = Ue[idx-1]
    s_0 = s[idx] 
    s_plus = s[idx+1] 
    s_minus = s[idx-1]
    ds_to_plus = abs(s_0 - s_plus)
    ds_to_minus = abs(s_0 - s_minus)

    # Coefficients for 1D central difference
    coeff_plus = 1/(ds_to_plus + ds_to_minus)
    coeff_minus = -coeff_plus

    # Central difference for dUedx term on non-uniform 1D grid
    dUedx = coeff_plus * U_plus + coeff_minus * U_minus

    return U_0 * dUedx

## Functions for Newton-Raphson iterations ##
def compute_residual(u_current, u_prev_dx, Ue_dUedx, nu, v, dy, ds):

    # Obtain grid length 
    nGrids = len(u_current)
    # Initialize residual
    R = zeros(nGrids)
    # Construct all terms in the PDE: R(u) = -u*du/dx-v*du/dy+ue*due/dx+nu*d^2u/dy^2
    # term 1 u*du/dx
    term1 = -u_current[1:-1] * (u_prev_dx[1:-1] - u_current[1:-1])/(-ds) 
    # term 2 v*du/dy
    term2 = - (v[1:-1]/(2*dy))*(u_current[2:] - u_current[0:-2])
    # term 3 ue*due/dx
    term3 = Ue_dUedx 
    # term 4 nu*d^2u/dy^2
    term4 = (nu/(dy*dy))* (u_current[2:] - 2*u_current[1:-1] + u_current[0:-2])
    # combine into final residual 
    R[1:-1] = term1 + term2 + term3 + term4

    return R

def compute_jacobian(u_current, u_prev_dx, nu, v, dy, ds):
  
    # Obtain grid length 
    nGrids = len(u_current)

    # Construct arrays for subdiagonal, superdiagonal, and diagonal 
    diagonal = -2*nu/(dy*dy) * ones(nGrids) + (-u_prev_dx + 2*u_current)/(-ds)
    sub_diagonal = (nu/(dy*dy)) * ones(nGrids-1) + v[1:]/(2*dy) 
    super_diagonal = (nu/(dy*dy)) * ones(nGrids-1) - v[0:-1]/(2*dy) 

    # Construct the jacobian
    dRdu = diag(diagonal) + diag(super_diagonal, 1) + diag(sub_diagonal, -1)
            
    # Enforce no update on the boundary points (the BCs are already set correctly on these points)
    # No update on u(y=0)
    dRdu[0][0] = 1
    dRdu[0][1] = 0
    # No update on u(y=L)
    dRdu[-1][-1] = 1
    dRdu[-1][-2] = 0

    # return it
    return dRdu

def construct_linear_system_and_solve(Ue_dUedx, Re, Ue_curr, ds, Dstar_curr, u_prev_dx, v, nyGrids):
    
    # Establish vertical grid
    ymin = 0
    ymax = 10*Dstar_curr
    y = linspace(ymin, ymax, nyGrids)
    dy = (ymax - ymin) / (nyGrids - 1)
    
    # Define necessary constants (velocity, viscosity at point adjacent to stagnation point)
    nu = 1.0/Re

    # # PERFORM LINEAR CASE WHERE u - ue on lhs
    # # Construct matrix A
    # off_diagonal_constant = nu/(dy*dy)
    # diagonal_constant = -2*nu/(dy*dy) +  U/(-ds)
    # diagonal = diagonal_constant * ones(nGrids)
    # off_diagonal = off_diagonal_constant * ones(nGrids-1)
    # A = diag(diagonal) + diag(off_diagonal, 1) + diag(off_diagonal, -1)
    # # Construct rhs b
    # b = -Ue_dUedx * ones(nGrids)
    # # Apply boundary conditions u(y = 0) = 0 and u(y = L) = Ue
    # # Apply BC (u(y=0) = 0)
    # A[0][0] = 1
    # A[0][1] = 0
    # b[0] = 0
    # # Apply edge BC (u(y=L) = Ue
    # A[-1][-1] = 1
    # A[-1][-2] = 0
    # b[-1] = Ue_curr
    # # Solve system for vertical velocity profile
    # Uy = linalg.solve(A, b)
    
    # PERFORM NONLINEAR CASE WHERE u on lhs is not simplified
    # Set tolerances and initial guess
    current_residual = 100.00
    TOL = 1e-8
    # Establish the initial guess as a scaled version of the U(y) at previous x
    if u_prev_dx[-1] == 0: # Set the scaling to 1 if our adjacent point is the stagnation vector (just handling this special case)
        scaling = 1
    else:
        scaling = Ue_curr/u_prev_dx[-1] # A scaling so that the U(y = L) at the new location is the new Ue value by default
    u_init = scaling * u_prev_dx
    u_init[-1] = Ue_curr
    u_current = u_init
    # Perform Newton-Raphson updates
    while current_residual > TOL:
        # Construct Jacobian matrix
        dRdu = compute_jacobian(u_current, u_prev_dx, nu, v, dy, ds)
        # Obtain residual vector
        R = compute_residual(u_current, u_prev_dx, Ue_dUedx, nu, v, dy, ds)
        # Compute Jacobian matrix numerically
        # Solve system for increment on vertical velocity profile
        du = linalg.solve(dRdu, -R)
        # Update
        u_next = u_current + du
        u_current = u_next
        # Compute the updated residual norm
        R = compute_residual(u_current, u_prev_dx, Ue_dUedx, nu, v, dy, ds)
        current_residual = linalg.norm(R)
        # print(current_residual)
        Uy = u_current
            
    return y, Uy

## Post-processing function ##
def compute_v(Uy, u_prev_dx, ds, y):
    dy = y[2] - y[1]
    f = -(u_prev_dx - Uy)/(-ds)
    v = zeros(len(Uy))
    v[1:] = integrate.cumtrapz(f, dx=dy)
    return v

## Plotting routines ##
def plot_BL_all_alphas(subpath, airfoil, Re, iters):
   
    for i in range(0, iters):
        pylab.figure()
        for alpha in range(-4, 9): # 9): # -4, 9
	    # Extract the Uy profile
            UyDirectory = os.path.join(subpath, 'Uy.' + str(i) + '.{}.npy'.format(alpha))
            velocity_profile = load(UyDirectory)
            y = velocity_profile[0,:]
            Uy = velocity_profile[1,:]
            # Plot the figure
            pylab.plot(Uy, y, linewidth=2, label='alpha = ' + str(alpha))

	# Plot boundary layers for all alpha (for specific airfoil and Re number)
        print("Plotting for point " + str(i))
        plotpath = os.path.join(basepath, 'profiles', airfoil, 'boundaryLayers')
        if not os.path.exists(plotpath): os.system('mkdir ' + plotpath)
        plotting_directory = os.path.join(plotpath, 'BL.' + str(Re) + '.point.{}..png'.format(i))
        pylab.grid()
        pylab.legend()
        pylab.xlabel('$U_{y}$', fontsize = 20)
        pylab.ylabel('$y$', fontsize = 20)
        pylab.title('Re = ' + str(Re) + ', point = ' + str(i), fontsize = 20)
        pylab.xticks(fontsize = 16)
        pylab.yticks(fontsize = 16)
        pylab.tight_layout()
        pylab.savefig(plotting_directory)
        pylab.close() 
    
    return 0

def plot_BL_all_points(subpath, airfoil, Re, iters):
   
    for alpha in range(-4, 9): # 9): # -4, 9
        pylab.figure()
        for i in range(0, iters):
	    # Extract the Uy profile
            UyDirectory = os.path.join(subpath, 'Uy.' + str(i) + '.{}.npy'.format(alpha))
            velocity_profile = load(UyDirectory)
            y = velocity_profile[0,:]
            Uy = velocity_profile[1,:]
            # Plot the figure
            pylab.plot(Uy, y, linewidth=2, label='i = ' + str(i))

	# Plot boundary layers for all x-points (for specific airfoil and Re number and alpha)
        print("Plotting for alpha " + str(alpha))
        plotpath = os.path.join(basepath, 'profiles', airfoil, 'boundaryLayers')
        if not os.path.exists(plotpath): os.system('mkdir ' + plotpath)
        plotting_directory = os.path.join(plotpath, 'BL.' + str(Re) + '.alpha.{}.png'.format(alpha))
        pylab.grid()
        pylab.legend()
        pylab.xlabel('$U_{y}$', fontsize = 20)
        pylab.ylabel('$y$', fontsize = 20)
        # pylab.title(airfoil + ' at Re = ' + str(Re), fontsize = 20)
        pylab.title('Re = ' + str(Re) + ', alpha = ' + str(alpha), fontsize = 20)
        pylab.xticks(fontsize = 16)
        pylab.yticks(fontsize = 16)
        pylab.tight_layout()
        pylab.savefig(plotting_directory)
        pylab.close() 
    
    return 0

## Main operation ## 
def solve_and_plot_BL(args):
    
    airfoil, Re, subpath = preprocess_args(args)

    for alpha in range(-4, 9):
        # Extract data about boundary layer from correct file 
        dirname = os.path.join(subpath, 'laminar.{}.npy'.format(alpha))
        s, Ue, Dstar, Theta = load(dirname)
        s0, idx, ds = extract_stagnation_point(s, Ue) 
        # For every x-point, compute the boundary layer profile U(y)
        # while idx > 0:
        nyGrids = 100
        nxGrids = 10
        for i in range(0,nxGrids): 
            if i == 0:
                u_prev_dx = zeros(nyGrids)
                v = zeros(nyGrids)
                Ue_ahead = 0
            else:
                u_prev_dx = Uy
                Ue_ahead = Ue[idx+1]
                ds = abs(s[idx+1] - s[idx]) # recompute distance between points at every step
            # Compute Ue*dUedx with finite difference
            Ue_dUedx = compute_constant_term_1storder(Ue[idx], Ue_ahead, ds)
            # Compute U(y) from TSL equations
            velocity_profile = construct_linear_system_and_solve(Ue_dUedx, Re, Ue[idx], ds, Dstar[idx], u_prev_dx, v, nyGrids)
            y, Uy = velocity_profile
            # Compute v(y)
            v = compute_v(Uy, u_prev_dx, ds, y)
            # Save results and decrement idx
            save(os.path.join(subpath, 'Uy.' + str(i) + '.{}.npy'.format(alpha)), velocity_profile)
            # Update index of point in direction going towards the trailing edge
            idx = idx-1
        
    plot_BL_all_alphas(subpath, airfoil, Re, nxGrids)
    plot_BL_all_points(subpath, airfoil, Re, nxGrids)
    
if __name__ == '__main__':
    Nfiles = 1516
    files = sorted(os.listdir(os.path.join(basepath, 'coordinates')))
    Res = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,
           500000, 1000000, 2000000, 5000000, 10000000, 20000000,
           50000000, 100000000, 200000000, 500000000, 1000000000]
    solve_and_plot_BL((files[0], 1000))
    # pool = multiprocessing.Pool()
    # pool.map(solve_and_plot_BL, itertools.product(files[0:1], Res))
