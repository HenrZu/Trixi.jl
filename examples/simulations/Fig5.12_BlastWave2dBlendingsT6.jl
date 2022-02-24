
# AT FIRST CLEAR OUT DIRECTORY

using Downloads: download
using Flux
using NNlib
using BSON: load
# network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.904-0.0005.bson", network)
# model2d = load(network, @__MODULE__)[:model2d]


model2d = load("2d_indicator.bson")[:model2d]
using OrdinaryDiffEq
using Trixi

# This elixir was one of the setups used in the following master thesis:
# - Julia Odenthal (2021)
#   Shock capturing with artifical neural networks
#   University of Cologne, advisors: Gregor Gassner, Michael Schlottke-Lakemper
# This motivates the particular choice of fluxes, mesh resolution etc.

###############################################################################
# semidiscretization of the compressible Euler equations
function initial_condition_blast_wav(x, t, equations::CompressibleEulerEquations2D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)
    phi = atan(y_norm, x_norm)
    sin_phi, cos_phi = sincos(phi)
  
    # Calculate primitive variables
    rho = r > 0.5 ? 1.0 : 1.1691
    v1  = 0 #r > 0.5 ? 0.0 : 0.1882 * cos_phi
    v2  = 0 #r > 0.5 ? 0.0 : 0.1882 * sin_phi
    p   = r > 0.5 ? 1.0E-3 : 1.245
  
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

equations = CompressibleEulerEquations2D(1.4)

"""
    initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)

A medium blast wave taken from
- Sebastian Hennemann, Gregor J. Gassner (2020)
  A provably entropy stable subcell shock capturing approach for high order split form DG
  [arXiv: 2008.12044](https://arxiv.org/abs/2008.12044)
"""
function initial_condition_blast_wave(x, t, equations::CompressibleEulerEquations2D)
  # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
  # Set up polar coordinates
  inicenter = SVector(0.0, 0.0)
  x_norm = x[1] - inicenter[1]
  y_norm = x[2] - inicenter[2]
  r = sqrt(x_norm^2 + y_norm^2)
  phi = atan(y_norm, x_norm)
  sin_phi, cos_phi = sincos(phi)

  # Calculate primitive variables
  rho = r > 0.5 ? 1.0 : 1.1691
  v1  = r > 0.5 ? 0.0 : 0.1882 * cos_phi
  v2  = r > 0.5 ? 0.0 : 0.1882 * sin_phi
  p   = r > 0.5 ? 1.0E-3 : 1.245

  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_blast_wav

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(4)


# GH 
## !!!! Need to deactivate the network in indicator_2d !!!
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure,  network=model2d)
# NeuralNetworkRayHesthaven
# network = joinpath(@__DIR__, "modelnnrhs-0.973-0.001.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnrhs-0.973-0.001.bson", network)
# model2d = load(network, @__MODULE__)[:model2d]
# indicator_sc = IndicatorNeuralNetwork(equations, basis,
#                                       indicator_type=NeuralNetworkRayHesthaven(),
#                                       alpha_max=0.5,
#                                       alpha_min=0.001,
#                                       alpha_smooth=true,
#                                       alpha_continuous=true,
#                                       alpha_amr=false,
#                                       variable=density_pressure,
#                                       network=model2d)

# NNPP
# network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.904-0.0005.bson", network)
# model2d = load(network, @__MODULE__)[:model2d]
# indicator_sc = IndicatorNeuralNetwork(equations, basis,
#                                       indicator_type=NeuralNetworkPerssonPeraire(),
#                                       alpha_max=0.5,
#                                       alpha_min=0.001,
#                                       alpha_smooth=true,
#                                       alpha_continuous=true,
#                                       alpha_amr=false,
#                                       variable=density_pressure,
#                                       network=model2d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2.0, -2.0)
coordinates_max = ( 2.0,  2.0)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 6)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=50,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


# For Vizualisation
# using Trixi2Vtk
# trixi2vtk("out/solution_*.h5", output_directory="out")