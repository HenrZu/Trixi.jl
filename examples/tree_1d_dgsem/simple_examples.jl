using Downloads: download
using Flux
using NNlib
using BSON: load

network = joinpath(@__DIR__, "modelnnrh-0.95-0.009.bson")
download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnrh-0.95-0.009.bson", network)
model1d = load(network, @__MODULE__)[:model1d]


# network = joinpath(@__DIR__, "modelnnpp-0.97-0.0001.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.97-0.0001.bson", network)
# model1d = load(network, @__MODULE__)[:model1d]


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
    rho = 2 + sin(4 * pi * x[1])
    v1  = 0.0
    p   = 1.0
  
    return prim2cons(SVector(rho, v1, p), equations)
  end

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_s

surface_flux = flux_lax_friedrichs# 
volume_flux  = flux_ranocha 
basis = LobattoLegendreBasis(3)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max=1.0,
#                                          alpha_min=0.001,
#                                          alpha_smooth=true,
#                                          variable=density_pressure)


# indicator_sc = IndicatorNeuralNetwork(equations, basis,
#                                       indicator_type=NeuralNetworkPerssonPeraire(),
#                                       alpha_max=1.0,
#                                       alpha_min=0.001,
#                                       alpha_smooth=true,
#                                       alpha_continuous=false,
#                                       alpha_amr=false,
#                                       variable=density_pressure,
#                                       network=model1d)


indicator_sc = IndicatorNeuralNetwork(equations, basis,
                                         indicator_type=NeuralNetworkRayHesthaven(),
                                         alpha_max=1.0,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         alpha_continuous=false,
                                         alpha_amr=false,
                                         variable=density_pressure,
                                         network=model1d)  

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-1,)
coordinates_max = ( 1,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.0005)
ode = semidiscretize(semi, tspan)



summary_callback = SummaryCallback()

analysis_interval = 500
# analysis_callback = AnalysisCallback(semi, interval=analysis_interval)
analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)


stepsize_callback = StepsizeCallback(cfl=0.2)

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

# Ideen: Step out -> Stages prüfen