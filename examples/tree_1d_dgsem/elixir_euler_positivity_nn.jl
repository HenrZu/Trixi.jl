using Flux
using NNlib
using BSON: load
model1d = load("erstes_modell_scheintgut.bson")[:model1d]

using OrdinaryDiffEq
using Trixi
###############################################################################
# semidiscretization of the compressible Euler equations
function initial_condition_s(x, t, equations::CompressibleEulerEquations1D)
    # Set up polar coordinates
    inicenter = SVector(0.0)
    x_norm = x[1] - inicenter[1]
    r = abs(x_norm)

    rho_0 = 1
    p_0  = 1.0e-5 # = true Sedov setup

    σ_rho = 0.25
    σ_p = 0.15
    # p0_outer = 1.0e-3 # = more reasonable setup

  
    # Calculate primitive variables
    rho = rho_0 + 1 / (4 * pi * σ_rho^2) * exp(-0.5 * r^2 / σ_rho^2)
    v1  = 0.0
    p   = p_0 + (equations.gamma - 1) / (4 * pi * σ_p^2) * exp(-0.5 * r^2 / σ_p^2)
  
    return prim2cons(SVector(rho, v1, p), equations)
  end

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_s

surface_flux = flux_hllc # flux_lax_friedrichs# 
volume_flux  = flux_ranocha 
basis = LobattoLegendreBasis(7)

indicator_sc = IndicatorNeuralNetwork(equations, basis,
                                      indicator_type=NeuralNetworkPerssonPeraire(),
                                      alpha_max=0.5,
                                      alpha_min=0.001,
                                      alpha_smooth=true,
                                      alpha_continuous=false,
                                      alpha_amr=false,
                                      variable=density_pressure,
                                      network=model1d)

# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max=0.5,
#                                          alpha_min=0.001,
#                                          alpha_smooth=true,
#                                          variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2,)
coordinates_max = ( 2,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10)
ode = semidiscretize(semi, tspan)


summary_callback = SummaryCallback()

analysis_interval = 500
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
                        stepsize_callback)

limiter! = PositivityPreservingLimiterRuedaGassner(semi, beta = 0.1)
stage_limiter! = limiter!
step_limiter! = limiter!


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary

# Ideen: Step out -> Stages prüfen