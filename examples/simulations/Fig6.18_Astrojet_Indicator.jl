using Downloads: download
using Flux
using NNlib
using BSON: load
# network = joinpath(@__DIR__, "modelnnpp-0.904-0.0005.bson")
# download("https://gist.github.com/JuliaOd/97728c2c15d6a7255ced6e46e3a605b6/raw/modelnnpp-0.904-0.0005.bson", network)
# model2d = load("datasets//2d_indicator.bson")[:model2d]
model2d = load("2d_indicator_astrojet.bson")[:model2d]

using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 5/3
equations = CompressibleEulerEquations2D(gamma)

# Initial condition adopted from 
# - Yong Liu, Jianfang Lu, and Chi-Wang Shu
#   An oscillation free discontinuous Galerkin method for hyperbolic systems
#   https://tinyurl.com/c76fjtx4
# Mach = 2000 jet
function initial_condition_astro_jet(x, t, equations::CompressibleEulerEquations2D)
  @unpack gamma = equations 
  rho = 0.5
  v1 = 0
  v2 = 0
  p =  0.4127
  # add inflow for t>0 at x=-0.5
  # domain size is [-0.5,+0.5]^2
  if (t > 0) && (x[1] ≈ -0.5) && (abs(x[2]) < 0.05)
    rho = 5
    v1 = 800 # about Mach number Ma = 2000
    v2 = 0
    p = 0.4127
  end
  return prim2cons(SVector(rho, v1, v2, p), equations)
end
initial_condition = initial_condition_astro_jet

boundary_conditions = (
                       x_neg=BoundaryConditionDirichlet(initial_condition_astro_jet),
                       x_pos=BoundaryConditionDirichlet(initial_condition_astro_jet),
                       y_neg=boundary_condition_periodic,
                       y_pos=boundary_condition_periodic,
                      )

surface_flux = flux_lax_friedrichs # HLLC needs more shock capturing (alpha_max)
volume_flux  = flux_ranocha # works with Chandrashekar flux as well
polydeg = 3
basis = LobattoLegendreBasis(polydeg)

# shock capturing necessary for this tough example
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.3, 
                                         alpha_min=0.0001,
                                         alpha_smooth=true,
                                         variable=density_pressure, network = model2d)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)

solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-0.5, -0.5)
coordinates_max = ( 0.5,  0.5)

mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
                periodicity=(false,true),
                n_cells_max=100_000)
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver, boundary_conditions=boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 0.001)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 5000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval= 100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorHennemannGassner(semi,                                                     
                                          alpha_max=1.0,                                            
                                          alpha_min=0.0001,                                         
                                          alpha_smooth=false,                                       
                                          variable=Trixi.density, network = model2d)                                 
                                                                                                    
amr_controller = ControllerThreeLevelCombined(semi, amr_indicator, indicator_sc,                    
                                              base_level=2,                                         
                                              med_level =0, med_threshold=0.0003, # med_level = current level
                                              max_level =8, max_threshold=0.003,                    
                                              max_threshold_secondary=indicator_sc.alpha_max)       
                                                                                                    
amr_callback = AMRCallback(semi, amr_controller,                                                    
                           interval=1,                                                              
                           adapt_initial_condition=true,                                            
                           adapt_initial_condition_only_refine=true) 

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, 
                        amr_callback, save_solution)


###############################################################################
# run the simulation
# use adaptive time stepping based on error estimates, time step roughly dt = 1e-7
sol = solve(ode, SSPRK43(),
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
