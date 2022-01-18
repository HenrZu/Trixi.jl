
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

surface_flux = flux_lax_friedrichs #flux_hllc # 
volume_flux  = flux_ranocha #flux_chandrashekar
basis = LobattoLegendreBasis(5)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)

coordinates_min = (-2,)
coordinates_max = ( 2,)
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=4,
                n_cells_max=10_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 17)
ode = semidiscretize(semi, tspan)


summary_callback = SummaryCallback()

analysis_interval = 500
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

amr_indicator = IndicatorLöhner(semi,
                                variable=density_pressure)
amr_controller = ControllerThreeLevel(semi, amr_indicator,
                                      base_level=4,
                                      med_level =0, med_threshold=0.1, # med_level = current level
                                      max_level =6, max_threshold=0.3)
amr_callback = AMRCallback(semi, amr_controller,
                           interval=2,
                           adapt_initial_condition=true,
                           adapt_initial_condition_only_refine=true)

stepsize_callback = StepsizeCallback(cfl=0.6)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        #amr_callback, 
                        stepsize_callback)


#limiter! = PositivityPreservingLimiterZhangShu(thresholds=(5.0e-6, 5.0e-6),
 #                                              variables=(Trixi.density, pressure))
limiter! = PositivityPreservingLimiterRuedaGassner(semi, beta = 0.1)
stage_limiter! = limiter!



###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(stage_limiter!, williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
