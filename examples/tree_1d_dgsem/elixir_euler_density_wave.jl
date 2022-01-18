
using OrdinaryDiffEq
using Trixi

###############################################################################
# semidiscretization of the compressible Euler equations

function initial_condition_bl(x, t, equations::CompressibleEulerEquations1D)
    # Modified From Hennemann & Gassner JCP paper 2020 (Sec. 6.3) -> "medium blast wave"
    # Set up polar coordinates
    inicenter = SVector(0.0)
    x_norm = x[1] - inicenter[1]
    if x_norm[1] <= 4 && x_norm[1] >= 2
        rho = 3
    else
        rho = 2.5
    end

  
    return prim2cons(SVector(rho, 0, 1), equations)
end

equations = CompressibleEulerEquations1D(1.4)

initial_condition = initial_condition_bl


surface_flux = flux_lax_friedrichs
volume_flux  = flux_ranocha
basis = LobattoLegendreBasis(3)
indicator_sc = IndicatorHennemannGassner(equations, basis,
                                         alpha_max=0.5,
                                         alpha_min=0.001,
                                         alpha_smooth=true,
                                         variable=density_pressure)
volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
                                                 volume_flux_dg=volume_flux,
                                                 volume_flux_fv=surface_flux)
solver = DGSEM(basis, surface_flux, volume_integral)


# surface_flux = flux_lax_friedrichs
# basis = LobattoLegendreBasis(3)
# indicator_sc = IndicatorHennemannGassner(equations, basis,
#                                          alpha_max=0.5,
#                                          alpha_min=0.001,
#                                          alpha_smooth=true,
#                                          variable=density_pressure)
# volume_integral = VolumeIntegralShockCapturingHG(indicator_sc;
#                                                  volume_flux_dg=flux_ranocha,
#                                                  volume_flux_fv=surface_flux)

# # solver = DGSEM(polydeg=5, surface_flux=flux_lax_friedrichs)

# solver = DGSEM(basis, surface_flux, volume_integral) #=VolumeIntegralPureLGLFiniteVolume(flux_hllc))

coordinates_min = 0
coordinates_max =  6
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=2,
                n_cells_max=30_000)


semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver)


###############################################################################
# ODE solvers, callbacks etc.

tspan = (0.0, 2.0)
ode = semidiscretize(semi, tspan)

summary_callback = SummaryCallback()

analysis_interval = 2000
analysis_callback = AnalysisCallback(semi, interval=analysis_interval)

alive_callback = AliveCallback(analysis_interval=analysis_interval)

save_solution = SaveSolutionCallback(interval=100,
                                     save_initial_solution=true,
                                     save_final_solution=true,
                                     solution_variables=cons2prim)

stepsize_callback = StepsizeCallback(cfl=0.8)

callbacks = CallbackSet(summary_callback,
                        analysis_callback, alive_callback,
                        save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=stepsize_callback(ode), # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);
summary_callback() # print the timer summary
