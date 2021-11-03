# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


  """
  A FV positivity limiter for the DGSEM

  Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations
  - Rueda-Ramirez, Gassner (2021),
    A Subcell Finite Volume Positivity-Preserving Limiter for DGSEM Discretizations of the Euler Equations
    [doi: 10.23967/wccm-eccomas.2020.038](https://doi.org/10.23967/wccm-eccomas.2020.038)

  """
  mutable struct PositivityPreservingLimiterRuedaGassner{N, Variables<:NTuple{N,Any}}
    semi_fv::AbstractSemidiscretization
    semi_dg::AbstractSemidiscretization
    beta::Float64
    variables::Variables
    s::Integer
    tolerance::Float64
    n_iterations::Integer
    u_safe
    node_dg
    node_tmp
    du_dα
    dp_du
  end
  

  function PositivityPreservingLimiterRuedaGassner(semi::AbstractSemidiscretization; beta, variables)
    @unpack volume_flux_fv = semi.solver.volume_integral 
    @unpack basis = semi.solver 
    @unpack surface_flux = semi.solver.surface_integral
    @unpack mesh, equations, initial_condition = semi 

    volume_integral = VolumeIntegralPureLGLFiniteVolume(volume_flux_fv)
    solver_fv = DGSEM(basis, volume_flux_fv, volume_integral)
    semi_fv = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_fv)

    solver_dg = DGSEM(basis, surface_flux)
    semi_dg = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_dg)
    num_vars = 2 + ndims(mesh)
    n_nodes = nnodes(solver_dg)
    n_elements = nelements(semi.solver, semi.cache)
    u_safe = zeros(num_vars * n_nodes^(ndims(mesh)) * n_elements)

    node_dg = zeros(2 + ndims(mesh))
    node_tmp = zeros(2 + ndims(mesh))
    du_dα = zeros(2 + ndims(mesh))
    dp_du = zeros(2 + ndims(mesh))

    # PositivityPreservingLimiterRuedaGassner(semi_fv, semi_dg, beta, variables, 1)
    PositivityPreservingLimiterRuedaGassner(semi_fv, semi_dg, beta, variables, 1, 1e-15, 10, u_safe, node_dg, node_tmp, du_dα, dp_du)
  end

  function (limiter!::PositivityPreservingLimiterRuedaGassner)(
      u_ode, integrator, semi::AbstractSemidiscretization, t)

      #integrator_save = deepcopy(integrator)
      u = wrap_array(u_ode, semi)
      limiter_rueda_gassner!(u, integrator, semi, limiter!)

      #integrator = deepcopy(integrator_save)

      # increase stage number in Limiter
      if limiter!.s == 4
        limiter!.s = 1
      else
        limiter!.s += 1
      end
      return nothing
  end
 
  
  
  include("positivity_alpha_rueda_gassner_dg1d.jl")
  include("positivity_alpha_rueda_gassner_dg2d.jl")
  
  
  end # @muladd
  