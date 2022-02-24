advection_velocity = 1.0
equations = LinearScalarAdvectionEquation1D(advection_velocity)

# Inital condition simple shock

function initial_condition_shock(x, t, equation::LinearScalarAdvectionEquation1D)

    x_trans = x - equation.advection_velocity * t
    if x_trans[1] >= 0.3 && x_trans[1] <= 0.6
        scalar = 1
    else
        scalar = 0
    end
    return SVector(scalar)
  end

boundary_conditions =  BoundaryConditionDirichlet(initial_condition_shock)


# Create DG solver with polynomial degree = 3 and (local) Lax-Friedrichs/Rusanov flux as surface flux
surface_flux        = flux_lax_friedrichs

# For pure FV
# volume_integral = VolumeIntegralPureLGLFiniteVolume(flux_hllc)
# solver = DGSEM(basis, surface_flux, volume_integral)

solver = DGSEM(polydeg=5, surface_flux = surface_flux)

coordinates_min = 0.0 # minimum coordinate
coordinates_max =  1.0 # maximum coordinate

# Create a uniformly refined mesh with periodic boundaries
mesh = TreeMesh(coordinates_min, coordinates_max,
                initial_refinement_level=6,
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

# to save Data (possible solution -  may exist better ones)
# pd = PlotData1D(sol)
# t = pd.x
# y = pd.data
# using DelimitedFiles
# open("n3_exact.txt", "w") do io
# 	writedlm(io, [t y])
# end