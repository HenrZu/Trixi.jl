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
  mutable struct PositivityPreservingLimiterRuedaGassner{RealT<:Real, Cache}
    beta::RealT
    stage::Int32
    cache::Cache
  end
  

  function PositivityPreservingLimiterRuedaGassner(semi::AbstractSemidiscretization; beta)
    cache = create_cache(semi)
    PositivityPreservingLimiterRuedaGassner{typeof(beta), typeof(cache)}(beta, 1, cache)

  end


  function create_cache(semi::AbstractSemidiscretization)
    @unpack volume_flux_fv = semi.solver.volume_integral 
    @unpack basis = semi.solver 
    @unpack surface_flux = semi.solver.surface_integral
    @unpack mesh, equations, initial_condition = semi 

    volume_integral = VolumeIntegralPureLGLFiniteVolume(volume_flux_fv)
    solver_fv = DGSEM(basis, volume_flux_fv, volume_integral)
    semi_fv = SemidiscretizationHyperbolic(mesh, equations, initial_condition, solver_fv)

    tolerance = 1e-15
    iterations_newton = 10
    num_vars = 2 + ndims(mesh)
    n_nodes = nnodes(solver_fv)
    n_elements = nelements(solver_fv, semi.cache)

    # conflicts when using AMR?
    u_safe = zeros(num_vars * n_nodes^(ndims(mesh)) * n_elements)

    node_dg = zeros(2 + ndims(mesh))
    node_tmp = zeros(2 + ndims(mesh))
    du_dα = zeros(2 + ndims(mesh))
    dp_du = zeros(2 + ndims(mesh))

    return (; semi_fv, tolerance, iterations_newton,
               u_safe, node_dg, node_tmp, du_dα, dp_du)
  end

  function (limiter!::PositivityPreservingLimiterRuedaGassner)(
      u_ode, integrator, semi::AbstractSemidiscretization, t)
      @unpack alpha = semi.solver.volume_integral.indicator.cache
      @unpack mesh = semi
      
      u = wrap_array(u_ode, semi)
      limiter_rueda_gassner!(u, alpha, mesh, integrator, semi, limiter!)

      # increase stage number in Limiter
      limiter!.stage == 5 ? limiter!.stage = 1 : limiter!.stage+= 1
      return nothing
  end
 
  
  
  include("positivity_alpha_rueda_gassner_dg1d.jl")
  include("positivity_alpha_rueda_gassner_dg2d.jl")
  
  
  end # @muladd
  