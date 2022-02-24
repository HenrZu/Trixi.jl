using Downloads: download
using Flux
using NNlib
using BSON: load
# network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.904-0.0005.bson", network)
# model2d = load("datasets//2d_indicator.bson")[:model2d]
model1d = load("1d_indicator.bson")[:model1d]

# !!!!!! deactivate Limiter in indicator_2d !!!!!!!!!!

using OrdinaryDiffEq
using Trixi

###############################################################################
"""
    initial_condition_sedov_positivity(x, t, equations::CompressibleEulerEquations1D)

A version of the Sedov blast based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_sedov_positivity(x, t, equations::CompressibleEulerEquations1D)
  # Set up polar coordinates
  inicenter = SVector(0.0)
  x_norm = x[1] - inicenter[1]
  r = abs(x_norm)

  # Ambient values
  rho_0 = 1
  p_0  = 1.0e-5

  # density
  σ_rho = 0.25
  rho = rho_0 + 1 / (4 * pi * σ_rho^2) * exp(-0.5 * r^2 / σ_rho^2)
  
  # velocity
  v1  = 0.0
  
  # pressure
  σ_p = 0.15
  p   = p_0 + (equations.gamma - 1) / (4 * pi * σ_p^2) * exp(-0.5 * r^2 / σ_p^2)
  
  return prim2cons(SVector(rho, v1, p), equations)
end
initial_condition = initial_condition_sedov_positivity

equations = CompressibleEulerEquations1D(1.4)


surface_flux = flux_hllc 
volume_flux  = flux_ranocha 
basis = LobattoLegendreBasis(7)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure, network = model1d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.5,)
coordinates_max = ( 1.5,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 1.05)
ode = semidiscretize(semi, tspan)


summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)


stepsize_callback = StepsizeCallback(cfl=0.45)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution, 
                        stepsize_callback)



###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54( williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary


# to save Data (possible solution -  may exist better ones)
pd = PlotData1D(sol)
x = pd.x
y = pd.data
density = y[:,1]
velocity = y[:,2]
pressure = y[:,3]

using DelimitedFiles
open("without_limiter.txt", "w") do io
 	writedlm(io, [x density velocity pressure])
end