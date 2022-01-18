using Downloads: download
using Flux
using NNlib
using BSON: load


# network = joinpath(@__DIR__, "modelnnpp-0.97-0.0001.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.97-0.0001.bson", network)
# model1d = load(network, @__MODULE__)[:model1d]

model1d = load("1d_indicator.bson")[:model1d]
# model1d = load("erster_limiter.bson")[:model1d]



using OrdinaryDiffEq
using Trixi

# This elixir was one of the setups used in the following master thesis:
# - Julia Odenthal (2021)
#   Shock capturing with artifical neural networks
#   University of Cologne, advisors: Gregor Gassner, Michael Schlottke-Lakemper
# This motivates the particular choice of fluxes, mesh resolution etc.

# positvity set up
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
  p   = 1#p_0 + (equations.gamma - 1) / (4 * pi * σ_p^2) * exp(-0.5 * r^2 / σ_p^2)

  return prim2cons(SVector(rho, v1, p), equations)
end

function initial_condition_blast_wav(x, t, equations::CompressibleEulerEquations1D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
  #   inicenter = SVector(0.0)
  #   x_norm = x[1] - inicenter[1]
  #   r = abs(x_norm)
  #   # The following code is equivalent to
  #   # phi = atan(0.0, x_norm)
  #   # cos_phi = cos(phi)
  #   # in 1D but faster
  #   cos_phi = x_norm > 0 ? one(x_norm) : -one(x_norm)
  
    # Calculate primitive variables
    rho = x[1] > 0 ? 1 : 0.5
    v1  = 0
    p   = 1
  
    return prim2cons(SVector(rho, v1, p), equations)
end
  
###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_blast_wave

surface_flux = flux_lax_friedrichs #flux_hllc
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorNeuralNetwork(equations, basis,
                                      indicator_type=NeuralNetworkPerssonPeraire(),
                                      alpha_max=0.5,
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

coordinates_min = (-2.0,)
coordinates_max = ( 2.0,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

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

limiter! = PositivityPreservingLimiterRuedaGassner(semi, beta = 0.7)
stage_limiter! = limiter!
step_limiter! = limiter!
                        
                        
                        ###############################################################################
                        # run the simulation
sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),                        
# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
                                    dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
                                    save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary                        


###############################################################################
# run the simulation

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks, maxiters=1e5);
# summary_callback() # print the timer summary

# pd=PlotData1D(sol)
# plot(pd["rho"])