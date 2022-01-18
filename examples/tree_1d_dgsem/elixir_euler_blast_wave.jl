using Downloads: download
using Flux
using NNlib
using BSON: load


# network = joinpath(@__DIR__, "modelnnpp-0.97-0.0001.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.97-0.0001.bson", network)
# model1d = load(network, @__MODULE__)[:model1d]

model1d = load("1d_indicator.bson")[:model1d]

using OrdinaryDiffEq
using Trixi
using Plots
using DelimitedFiles


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

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_blast_wave

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure,  network = model1d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2,)
coordinates_max = ( 2,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 10)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100

# analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

analysis_callback = AnalysisCallback(semi, interval=analysis_interval,
                                     extra_analysis_integrals=(entropy,))

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


###############################################################################
# run the simulation
limiter! = PositivityPreservingLimiterRuedaGassner(semi, beta = 0.3)
stage_limiter! = limiter!
step_limiter! = limiter!


###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, step_limiter!, williamson_condition=false),
#sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);
summary_callback() # print the timer summary

# pd=PlotData1D(sol)
# plot(pd["rho"])

# A =  stepsize_callback.condition.Data_alpha
# time = A[2:3:end]
# alpha = A[3:3:end]

# open("Gh_alphas_1d_32elements.txt", "w") do io
# 	writedlm(io, [x time  alpha])
# end



#plot!(getmesh(pd["rho"]))