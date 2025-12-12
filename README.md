This is going to be a long one...

Testing
  Testing samplers:
  * `optimise_step_H2`: Plots for finding the optimum acceptance step size for H2 MCMC.
  * `test_thinning_plots`: Shows autocorrelation with and without thinned samples (unused).
  * `test_MALA_H2`: Tests Metropolis-Adjusted Langevin Algorithm (MALA) sampling for the H2 molecule.
  * `test_samples`: Compares MALA, sMALA, and Metropolis-Hastings (MH) sampling for QHO states.
  * `test_sampling_3d`: Tests 3D sampling (MALA) for the Hydrogen atom.
  * `test_samples_hydrogen`: Plots the radial distribution for hydrogen samples.
  
  Testing differentiators:
  * `test_diffrentiators_3d`: show the laplacian for a random function.
  * `test_second_order_diffrentiators`: Plots RMS error vs. step size for QHO second derivatives (various orders).
  * `test_hydrogen_laplacian`: Plots RMS error vs. step size for the numerical Laplacian of the Hydrogen 1s wavefunction.
  * `test_h2_parameter_gradient_error_analysis`: Plots RMS error vs. step size for the H2 log-wavefunction parameter gradient.

  Testing minimisers: (mostly unused) 
  * `test_minimiser`: Compares the final convergence point and value for 1D optimization algorithms (GD, SGD, RMSProp, QN) on a simple parabola.
  * `test_minimiser_3d`: Visualizes and compares the paths and final convergence points of 2D optimization algorithms (GD, SGD, RMSProp, QN) on a random function.

Harmonic Oscillator: 
  * `test_localenergy`: finds the expected energies for QHO states 1-4

Hydrogen:
  * `ground_state`: Finds the ground state using GD
  * `plot_ground_state`: Samples the electron's position using MH, plots 2D electron density distribution.

Hydrogen Molecule: 
  * `test_samples`: plots the 2D probability density of the two electrons
  * `test_SR`: Tests Stochastic Reconfiguration 
  * `test_QN`: Tests Quasi-Newton 
  * `test_minimisers_convergence`: Compares the energy convergence traces of six different optimisation algorithms (SR, QN, GD, RMSProp, SA) and saves  
  * `parallel_ground_state_energies`: Calculates the potential energy curve and saves it

Other: 
  plot_morse: run this file to plot the energy curve and thetas over the range of r_12 based on saved data
  plot_convergence: run this file to plot the convergence of different methods based on saved data

Thank you :)
    
    
