import os
import itertools
import multiprocessing
from numpy import *
import scipy.interpolate
import pylab

basepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

def plot_BL(subpath, airfoil, Re):
   
    pylab.figure()
    for alpha in range(-4, 9):
        # Extract the Uy profile
        UyDirectory = os.path.join(subpath, 'Uy.{}.npy'.format(alpha))
        velocity_profile = load(UyDirectory)
        y = velocity_profile[0,:]
        Uy = velocity_profile[1,:]
		# Plot the figure
        pylab.plot(Uy, y, linewidth=2, label='alpha = ' + str(alpha))

    # Plot boundary layers for all alpha (for specific airfoil and Re number)
    plotpath = os.path.join(basepath, 'profiles', airfoil, 'boundaryLayers')
    if not os.path.exists(plotpath): os.system('mkdir ' + plotpath)
    plotting_directory = os.path.join(plotpath, 'BL.{}.10dstar.nonlinear.png'.format(Re))
    pylab.grid()
    pylab.legend()
    pylab.xlabel('$U_{y}$', fontsize = 20)
    pylab.ylabel('$y$', fontsize = 20)
    pylab.title(airfoil + ' at Re = ' + str(Re), fontsize = 20)
    pylab.xticks(fontsize = 16)
    pylab.yticks(fontsize = 16)
    pylab.tight_layout()
    pylab.savefig(plotting_directory)
    pylab.close() 
    return 0

def residual_calculation(y, Uy, Re, ds, Ue, Ue_dUedx, non_linear):

    # Define necessary constants
    N = len(y)
    nu = 1.0/Re
    dy = y[1] - y[0]
    
    # Create residual vector containing contribution from each grid point j in U[y], y in [0,L]
    residual = zeros(N)
    for j in range(1, N-1):
        if non_linear == 0:
            residual[j] = Ue*Uy[j]/(-ds) + Ue_dUedx + nu*(Uy[j+1]-2*Uy[j]+Uy[j-1])/(dy*dy)
        else:
            residual[j] = Uy[j]*Uy[j]/(-ds) + Ue_dUedx + nu*(Uy[j+1]-2*Uy[j]+Uy[j-1])/(dy*dy)
        # residual[j] = - Ue_dUedx - nu*(Uy[j+1]-2*Uy[j]+Uy[j-1])/(dy*dy)
        # print("The " + str(j) + " residual component is " + str(residual[j]))
    r = linalg.norm(residual)
	
    return r
    
def construct_linear_system_and_solve(Ue_dUedx, Re, idx, ds, Ue, Dstar, non_linear):
    
    # Establish vertical grid
    nGrids = 1000
    ymin = 0
    ymax = 10*Dstar[idx]
    y = linspace(ymin, ymax, nGrids)
    dy = (ymax - ymin) / (nGrids - 1)
    
    # Define necessary constants (velocity, viscosity at point adjacent to stagnation point)
    U = Ue[idx]
    nu = 1.0/Re

    if (non_linear == 0):
		# PERFORM LINEAR CASE WHERE u - ue on lhs
		# Construct matrix A
        off_diagonal_constant = nu/(dy*dy)
        diagonal_constant = -2*nu/(dy*dy) +  U/(-ds)
        diagonal = diagonal_constant * ones(nGrids)
        off_diagonal = off_diagonal_constant * ones(nGrids-1)
        A = diag(diagonal) + diag(off_diagonal, 1) + diag(off_diagonal, -1)
		# Construct rhs b
        b = -Ue_dUedx * ones(nGrids)
		# Apply boundary conditions u(y = 0) = 0 and u(y = L) = Ue
		# Apply BC (u(y=0) = 0)
        A[0][0] = 1
        A[0][1] = 0
        b[0] = 0
		# Apply edge BC (u(y=L) = Ue
        A[-1][-1] = 1
        A[-1][-2] = 0
        b[-1] = U
		# Solve system for vertical velocity profile
        Uy = linalg.solve(A, b)
    else:
        # PERFORM NONLINEAR CASE WHERE u on lhs is not simplified
        # Set tolerances and initial guess
        current_residual = 10.00
        TOL = 1e-8
        u_current = zeros(nGrids)
        u_current[0] = 0
        u_current[-1] = U
        R = zeros(nGrids)
        off_diagonal = nu/(dy*dy) * ones(nGrids-1)
        while current_residual > TOL:
    		# Construct Jacobian matrix dRdu
            diagonal = -2*nu/(dy*dy) + 2*u_current/(-ds)
            dRdu = diag(diagonal) + diag(off_diagonal, 1) + diag(off_diagonal, -1)
            # Construct rhs R(u)
            #print('Term 1 in R is', u_current[1:-2]*u_current[1:-2]/(-ds))
            #print('Term 2 in R is', Ue_dUedx)
            #print('Term 3 in R is', (nu/(dy*dy))*(u_current[2:] - 2*u_current[1:-1] + u_current[0:-2]))
            R[1:-1] = (u_current[1:-1]*u_current[1:-1])/(-ds) + Ue_dUedx + (nu/(dy*dy))* (u_current[2:] - 2*u_current[1:-1] + u_current[0:-2])
            # Enforce no update on the boundary points (the BCs are already set correctly on these points)
		    # No update on u(y=0)
            dRdu[0][0] = 1
            dRdu[0][1] = 0
		    # No update on u(y=L)
            dRdu[-1][-1] = 1
            dRdu[-1][-2] = 0
    		# Solve system for increment on vertical velocity profile
            #print('dRdu is ', dRdu)
            #print('R is ', R)
            du = linalg.solve(dRdu, -R)
            # print('du is', du)
            # Update
            u_next = u_current + du
            u_current = u_next
            Uy = u_current
            R[1:-1] = (u_current[1:-1]*u_current[1:-1])/(-ds) + Ue_dUedx + (nu/(dy*dy))* (u_current[2:] - 2*u_current[1:-1] + u_current[0:-2])
            current_residual = linalg.norm(R)
            print(current_residual)
            
    return y, Uy

def compute_constant_term_1storder(idx, ds, Ue):

    # Obtain adjacent points velocities and positions (in preparation for derivatives)
    U_0 = Ue[idx]

    return -U_0 * U_0 / (-ds)

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
    ds_to_after = abs(s_stagnation - s_after)

    return s_before, idx_before, ds_to_before

def preprocess_args(args):
    fname, Re = args
    if fname.endswith('.dat'): fname = fname[:-4]
    subpath = os.path.join(basepath, 'profiles', fname, str(Re))
    return fname, Re, subpath

def solve_and_plot_BL(args):
    airfoil, Re, subpath = preprocess_args(args)
    nonlinear_FLAG = 1 # Use linear approximation
    for alpha in range(-4, 9): 
        dirname = os.path.join(subpath, 'laminar.{}.npy'.format(alpha))
        s, Ue, Dstar, Theta = load(dirname)
        s_before, idx_before, ds_to_before = extract_stagnation_point(s, Ue)
        # Ue_dUedx = compute_constant_term(idx_before, s, Ue)
        Ue_dUedx = compute_constant_term_1storder(idx_before, ds_to_before, Ue)
        velocity_profile = construct_linear_system_and_solve(Ue_dUedx, Re, idx_before, ds_to_before, Ue, Dstar, nonlinear_FLAG)
        print('Subpath complete ' + subpath)
        y, Uy = velocity_profile
        r = residual_calculation(y, Uy, Re, ds_to_before, Ue[idx_before], Ue_dUedx, nonlinear_FLAG)
        print("The residual is " + str(r))
        save(os.path.join(subpath, 'Uy.{}.npy'.format(alpha)), velocity_profile)
    
    plot_BL(subpath, airfoil, Re)
    
if __name__ == '__main__':
    Nfiles = 1516
    files = sorted(os.listdir(os.path.join(basepath, 'coordinates')))
    Res = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000,
           500000, 1000000, 2000000, 5000000, 10000000, 20000000,
           50000000, 100000000, 200000000, 500000000, 1000000000]
    solve_and_plot_BL((files[0], 1000))
    #pool = multiprocessing.Pool()
    #pool.map(solve_and_plot_BL, itertools.product(files[0:1], Res))
