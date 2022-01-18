
using OrdinaryDiffEq
using Trixi

function initial_condition_sedov(x, t, equations::CompressibleEulerEquations2D)
    # Set up polar coordinates
    inicenter = SVector(0.0, 0.0)
    x_norm = x[1] - inicenter[1]
    y_norm = x[2] - inicenter[2]
    r = sqrt(x_norm^2 + y_norm^2)

    rho_0 = 1
    p_0  = 1.0e-5 # = true Sedov setup

    σ_rho = 0.25
    σ_p = 0.15
    # p0_outer = 1.0e-3 # = more reasonable setup
  
    # Calculate primitive variables
    rho = rho_0 + 1 / (4 * pi * σ_rho^2) * exp(-0.5 * r^2 / σ_rho^2)
    v1  = 0.0
    v2  = 0.0
    p   = p_0 + (equations.gamma - 1) / (4 * pi * σ_p^2) * exp(-0.5 * r^2 / σ_p^2)
  
    return prim2cons(SVector(rho, v1, v2, p), equations)
end

###############################################################################
# semidiscretization of the compressible Euler equations
gamma = 1.4
equations = CompressibleEulerEquations2D(gamma)

initial_condition = initial_condition_sedov

surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(4)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
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

limiter! = PositivityPreservingLimiterRuedaGassner(semi, beta = 0.1)
stage_limiter! = limiter!


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 4)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 100
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)
visualization = VisualizationCallback(interval=20)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.5)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback, 
                        save_solution,
                        stepsize_callback,
                        #visualization
                        )
###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks, maxiters=1e5);
            
# sol = solve(ode, SSPRK43(stage_limiter!),
#             save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
