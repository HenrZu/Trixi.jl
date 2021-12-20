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
    dt::Float64
    t::Float64
    data:: Array{Float64, 1}
    labels:: Array{Float64, 1}
    list_element:: Array{Float64, 1}
  end
  

  function PositivityPreservingLimiterRuedaGassner(semi::AbstractSemidiscretization; beta)
    cache = create_cache(semi)
    data = Float64[]
    labels = Float64[]
    list_element = Float64[]
    dt = 2.0
    t = 3.0
    PositivityPreservingLimiterRuedaGassner{typeof(beta), typeof(cache)}(beta, 1, cache, dt, t, data, labels, list_element)

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
    u_latest_stage = similar(u_safe)
    tmp_lates_stage = similar(u_safe)

    node_dg = zeros(2 + ndims(mesh))
    node_tmp = zeros(2 + ndims(mesh))
    du_dα = zeros(2 + ndims(mesh))
    dp_du = zeros(2 + ndims(mesh))

    return (; semi_fv, tolerance, iterations_newton, tmp_lates_stage,
               u_safe, u_latest_stage, node_dg, node_tmp, du_dα, dp_du)
  end

  function (limiter!::PositivityPreservingLimiterRuedaGassner)(
      u_ode, integrator, semi::AbstractSemidiscretization, t)
      @unpack alpha = semi.solver.volume_integral.indicator.cache
      @unpack mesh = semi
      
      
      u = wrap_array(u_ode, semi)
      x = getX(limiter!.data,limiter!.labels, limiter!.list_element, u, semi)
      limiter_rueda_gassner!(u, alpha, mesh, integrator, semi, limiter!)

      # save solution o_ode and tmp for next stage
      for idx in eachindex(u_ode)
        limiter!.cache.u_latest_stage[idx] = u_ode[idx]
        limiter!.cache.tmp_lates_stage[idx] = integrator.cache.tmp[idx]        
      end

      # brauche ich nur für stage 5
      limiter!.dt = integrator.dt
      limiter!.t = integrator.t


      #upper_bound!(u, alpha, mesh, integrator, semi, limiter!)

      
      for α in limiter!.list_element
        push!(limiter!.labels, alpha[convert(Int64, α)])
      end
      limiter!.list_element = Float64[]
      if t > 25
          h5open("datasets//train_elixir_weak_n3_g5.h5", "w") do file
            write(file, "X", limiter!.data)
            write(file, "Y", limiter!.labels)
          end
        println("Length von X $(length(limiter!.data))")
        println("Length von X $(length(limiter!.labels))")
        error("Ziel erreicht!!")
      end

      # increase stage number in Limiter
      limiter!.stage == 5 ? limiter!.stage = 1 : limiter!.stage+= 1
      return nothing
  end
 
  
  
  include("positivity_alpha_rueda_gassner_dg1d.jl")
  include("positivity_alpha_rueda_gassner_dg2d.jl")



  function getX(data, labels, list_element,  u::AbstractArray{<:Any,3},semi)
    @unpack mesh, equations, solver, cache = semi
    @unpack alpha_max, alpha_min, alpha_smooth, variable = semi.solver.volume_integral.indicator
    @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded = semi.solver.volume_integral.indicator.cache
    # TODO: Taal refactor, when to `resize!` stuff changed possibly by AMR?
    #       Shall we implement `resize!(semi::AbstractSemidiscretization, new_size)`
    #       or just `resize!` whenever we call the relevant methods as we do now?
    resize!(alpha, nelements(solver, cache))
    if alpha_smooth
    resize!(alpha_tmp, nelements(solver, cache))
    end

    # magic parameters
    threshold = 0.5 * 10^(-1.8 * (nnodes(solver))^0.25)
    parameter_s = log((1 - 0.0001)/0.0001)

    @threaded for element in eachelement(solver, cache)
      indicator  = indicator_threaded[Threads.threadid()]
      modal      = modal_threaded[Threads.threadid()]

      # Calculate indicator variables at Gauss-Lobatto nodes
      for i in eachnode(solver)
        u_local = get_node_vars(u, equations, solver, i, element)
        indicator[i] = semi.solver.volume_integral.indicator.variable(u_local, equations)
      end

      # Convert to modal representation
      multiply_scalar_dimensionwise!(modal, solver.basis.inverse_vandermonde_legendre, indicator)

      # Calculate total energies for all modes, without highest, without two highest
      total_energy = zero(eltype(modal))
      for i in 1:nnodes(solver)
        total_energy += modal[i]^2
      end
      total_energy_clip1 = zero(eltype(modal))
      for i in 1:(nnodes(solver)-1)
        total_energy_clip1 += modal[i]^2
      end
      total_energy_clip2 = zero(eltype(modal))
      for i in 1:(nnodes(solver)-2)
        total_energy_clip2 += modal[i]^2
      end
      if length(data) == 0
        push!(data,  0.0)
        push!(data,  0.0)
        push!(labels,  0.0)
      end
        
      if (total_energy - total_energy_clip1) / total_energy > 0 || (total_energy_clip1 - total_energy_clip2)/total_energy_clip1 > 0
        push!(data,  (total_energy - total_energy_clip1)/total_energy)
        push!(data,  (total_energy_clip1 - total_energy_clip2)/total_energy_clip1)
        push!(list_element, element)
      end
    end
  end
  
  
end # @muladd
  