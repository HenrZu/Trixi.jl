using Downloads: download
using Flux
using NNlib
using BSON: load
using Plots


using OrdinaryDiffEq
using Trixi


N = 3                
tspan = (0.0, 0.001)
CFL = 0.5

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
  rho = 2 + sin(4 * π * x[1])
  v1  = 0
  p   = 1

  return prim2cons(SVector(rho, v1, p), equations)
end

###############################################################################
# semidiscretization of the compressible Euler equations

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_blast_wav

boundary_conditions = BoundaryConditionDirichlet(initial_condition)

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(N)


coordinates_min = (-1,)
coordinates_max = ( 1,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=0,
                n_cells_max=10_000)


                
                
                ###############################################################################
                # ODE solvers, callbacks etc.



stepsize_callback = StepsizeCallback(cfl=CFL)

callbacks = CallbackSet(stepsize_callback)


### indiviudal set up

# GH
model1d = load("erster_limiter.bson")[:model1d]
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure, network = model1d)

volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
                                                 
solver = DGSEM(basis, surface_flux, volume_integral)

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, tspan)

println("-------------- GH -----------------------------")

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);


            # pd = PlotData1D(sol)
            # b=plot(pd["rho"])


## NeuralNetworkPerssonPeraire


# network = joinpath(@__DIR__, "modelnnpp-0.97-0.0001.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.97-0.0001.bson", network)
# model1d = load(network, @__MODULE__)[:model1d]
model1d = load("erster_limiter.bson")[:model1d]


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

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
ode = semidiscretize(semi, tspan)

println("-------------- New Approach -----------------------------")

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);

pd = PlotData1D(sol)
 b=plot(pd["rho"])



#             # NNPPh
# network = joinpath(@__DIR__, "modelnnpph-0.9382341028756581-0.9606943550118051-1.0e-5.bson")
# model1d = load(network, @__MODULE__)[:model1d]


# indicator_sc = IndicatorNeuralNetwork(equations, basis,
#                                       indicator_type=NeuralNetworkPerssonPeraire(),
#                                       alpha_max=0.5,
#                                       alpha_min=0.001,
#                                       alpha_smooth=true,
#                                       alpha_continuous=false,
#                                       alpha_amr=false,
#                                       variable=density_pressure,
#                                       network=model1d)

# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg=volume_flux,
#                                                  volume_flux_fv=surface_flux)
                                                 
# solver = DGSEM(basis, surface_flux, volume_integral)

# semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)
# ode = semidiscretize(semi, tspan)

# println("-------------- NNPPh -----------------------------")

# sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
#             dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
#             save_everystep=false, callback=callbacks, maxiters=1e5);

