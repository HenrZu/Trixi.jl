
using OrdinaryDiffEq
using Trixi
using Plots

###############################################################################
# semidiscretization of the linear advection equation

surface_flux = flux_lax_friedrichs
function initial_condition_shock(x, t, equation::LinearScalarAdvectionEquation1D)
    inicenter = SVector(0.0)
    x_norm = x[1] - inicenter[1]
    if x_norm[1] <= 0.6  && x_norm[1] >= 0.3
      u = 1
    else
      u = 0 
    end
    return SVector(u)
end

advectionvelocity = 1.0
equations = LinearScalarAdvectionEquation1D(advectionvelocity)

# function boundary_condition_convergence_test(u_inner, orientation, direction, x, t,
#   surface_flux_function,
#   equation::LinearScalarAdvectionEquation1D)
# u_boundary = initial_condition_convergence_test(x, t, equation)

# # Calculate boundary flux
# if direction == 2  # u_inner is "left" of boundary, u_boundary is "right" of boundary
# flux = surface_flux_function(u_inner, u_boundary, orientation, equation)
# else # u_boundary is "left" of boundary, u_inner is "right" of boundary
# flux = surface_flux_function(u_boundary, u_inner, orientation, equation)
# end

# return flux
# end

# boundary_conditions = (
#                        x_neg=SVector(1),
#                        x_pos=SVector(0)
#                       )
# boundary_conditions = BoundaryConditionDirichlet(initial_condition_shock)

boundary_conditions =  BoundaryConditionDirichlet(initial_condition_shock)


# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
# solver = DGSEM(polydeg=5, surface_flux = surface_flux)

solver = DGSEM(polydeg=1, surface_flux=surface_flux,
               volume_integral=VolumeIntegralPureLGLFiniteVolume(surface_flux))

coordinates_min = 0 # minimum coordinate
coordinates_max = 1 # maximum coordinat

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=3,
                n_cells_max=30_000) # set maximum capacity of tree data structure

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_shock, solver, boundary_conditions = boundary_conditions)


###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0.0 to 1.0
ode = semidiscretize(semi, (0.0, 0.01));

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation setup
# and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the results
analysis_callback = AnalysisCallback(semi, interval=100)

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval=100,
                                     solution_variables=cons2prim)

# The StepsizeCallback handles the re-calculcation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl=0.5)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution, stepsize_callback)


###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition=false),
            dt=1.0, # solve needs some value here but it will be overwritten by the stepsize_callback
            save_everystep=false, callback=callbacks);

# Print the timer summary
summary_callback()

plot(sol)