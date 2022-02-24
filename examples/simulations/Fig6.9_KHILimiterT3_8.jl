using Downloads: download
using Flux
using NNlib
using BSON: load
# network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.904-0.0005.bson", network)
# model2d = load("datasets//2d_indicator.bson")[:model2d]
model2d = load("2d_indicator_unlimited.bson")[:model2d]

using OrdinaryDiffEq
using Trixi

"""
    initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)

A version of the classical Kelvin-Helmholtz instability based on
- Andrés M. Rueda-Ramírez, Gregor J. Gassner (2021)
  A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations
  of the Euler Equations
  [arXiv: 2102.06017](https://arxiv.org/abs/2102.06017)
"""
function initial_condition_kelvin_helmholtz_instability(x, t, equations::CompressibleEulerEquations2D)
  # change discontinuity to tanh
  # typical resolution 128^2, 256^2
  # domain size is [-1,+1]^2
  slope = 15
  amplitude = 0.02
  B = tanh(slope * x[2] + 7.5) - tanh(slope * x[2] - 7.5)
  rho = 0.5 + 0.75 * B
  v1 = 0.5 * (B - 1)
  v2 = 0.1 * sin(2 * pi * x[1])
  p = 1.0
  return prim2cons(SVector(rho, v1, v2, p), equations)
end


equations = CompressibleEulerEquations2D(1.4)

initial_condition = initial_condition_kelvin_helmholtz_instability 

surface_flux =  flux_lax_friedrichs
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=1,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure,  network=model2d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1.5, -1.5)
coordinates_max = ( 1.5,  1.5)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=100_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)



###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 3.8) 
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 200


analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, 
                        save_solution,
                        stepsize_callback,
                        )

# Positivity limiter as stage Callback    
limiter! = PositivityPreservingLimiterRuedaGassner(semi, beta = 0.6)
stage_limiter! = limiter!
step_limiter! = limiter!
                        
                        
                        ###############################################################################
                        # run the simulation
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);

summary_callback() # print the timer summary


# For Vizualisation (Fig 6.7)
using Trixi2Vtk
trixi2vtk("out/solution_*.h5", output_directory="out")