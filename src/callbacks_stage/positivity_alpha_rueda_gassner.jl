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
      
      # alpha_copy = deepcopy(alpha)
      u = wrap_array(u_ode, semi)


      u_copy = deepcopy(u)
      # semi_copy = deepcopy(semi)
      # if t > 0.0
      #   x = getX(limiter!, u_copy, semi)
      # end

      

      limiter_rueda_gassner!(u, alpha, mesh, integrator, semi, limiter!)

      if t > 0.0
        x = getX(limiter!, u_copy, semi)
      end

      # upper_bound!(u, alpha, mesh, integrator, semi, limiter!)

      # save solution o_ode and tmp for next stage
      for idx in eachindex(u_ode)
        limiter!.cache.u_latest_stage[idx] = u_ode[idx]
        limiter!.cache.tmp_lates_stage[idx] = integrator.cache.tmp[idx]        
      end

      # brauche ich nur fÃ¼r stage 5
      limiter!.dt = integrator.dt
      limiter!.t = integrator.t

      # if limiter!.stage == 1
      #   limiter!.list_element = Float64[]
      #   limiter!.data = Float64[]
      #   push!(limiter!.labels , t)
      # end

      # # maximal coefficient
      # max_a = maximum(alpha)
      # # mean_a = mean(alpha)
      # A = alpha
      # amount_fo = length(A[A.>0]) / length(alpha)


      # push!(limiter!.data , max_a)
      # # push!(limiter!.list_element , mean_a)
      # push!(limiter!.list_element , amount_fo)
    

      # if limiter!.stage == 5
      #   ave = mean(limiter!.list_element)
      #   maxi = maximum(limiter!.data)
      #   push!(limiter!.labels , ave)
      #   push!(limiter!.labels , maxi)
      # end


      #upper_bound!(u, alpha, mesh, integrator, semi, limiter!)

      
      for a in limiter!.list_element
        push!(limiter!.labels, alpha[convert(Int64, a)])
      end
      limiter!.list_element = Float64[]
      # if t > 3
      #     h5open("datasets//train_shock_06.h5", "w") do file
      #       write(file, "X", limiter!.data)
      #       write(file, "Y", limiter!.labels)
      #     end
      #   println("Length von X $(length(limiter!.data))")
      #   println("Length von Labels $(length(limiter!.labels))")
      #   error("Ziel erreicht!!")
      # end

      # increase stage for Limiter
      limiter!.stage == 5 ? limiter!.stage = 1 : limiter!.stage+= 1
      return nothing
  end
 
  
  
  include("positivity_alpha_rueda_gassner_dg1d.jl")
  include("positivity_alpha_rueda_gassner_dg2d.jl")

  # 2d case
  function getX(limiter!,  u::AbstractArray{<:Any,4},semi)
    @unpack mesh, equations, solver, cache = semi
    @unpack alpha_max, alpha_min, alpha_smooth, variable = semi.solver.volume_integral.indicator
    @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded, modal_tmp1_threaded = semi.solver.volume_integral.indicator.cache
    @unpack data,labels,list_element = limiter!


    # magic parameters
    threshold = 0.5 * 10^(-1.8 * (nnodes(solver))^0.25)
    parameter_s = log((1 - 0.0001)/0.0001)

    for element in eachelement(solver, cache)
      if alpha[element] < 0.001
        indicator  = indicator_threaded[Threads.threadid()]
        modal      = modal_threaded[Threads.threadid()]
        modal_tmp1 = modal_tmp1_threaded[Threads.threadid()]

        # Calculate indicator variables at Gauss-Lobatto nodes
        for j in eachnode(solver), i in eachnode(solver)
          u_local = get_node_vars(u, equations, solver, i, j, element)
          indicator[i, j] = semi.solver.volume_integral.indicator.variable(u_local, equations)
        end

        # Convert to modal representation
        multiply_scalar_dimensionwise!(modal, solver.basis.inverse_vandermonde_legendre, indicator, modal_tmp1)

        # Calculate total energies for all modes, without highest, without two highest
        total_energy = zero(eltype(modal))
        for j in 1:nnodes(solver), i in 1:nnodes(solver)
          total_energy += modal[i, j]^2
        end
        total_energy_clip1 = zero(eltype(modal))
        for j in 1:(nnodes(solver)-1), i in 1:(nnodes(solver)-1)
          total_energy_clip1 += modal[i, j]^2
        end
        total_energy_clip2 = zero(eltype(modal))
        for j in 1:(nnodes(solver)-2), i in 1:(nnodes(solver)-2)
          total_energy_clip2 += modal[i, j]^2
        end
        total_energy_clip3 = zero(eltype(modal))
        for j in 1:(nnodes(solver)-3), i in 1:(nnodes(solver)-3)
          total_energy_clip3 += modal[i, j]^2
        end

        X1 = (total_energy - total_energy_clip1)/total_energy
        X2 = (total_energy_clip1 - total_energy_clip2)/total_energy_clip1
        X3 = (total_energy_clip2 - total_energy_clip3)/total_energy_clip2
        X4 = nnodes(solver)
        network_input = SVector(X1, X2, X3, X4)

        # Scale input data
        network_input = network_input / max(maximum(abs, network_input), one(eltype(network_input)))
            
        if network_input[1] > 0 || network_input[2] > 0 || network_input[3] > 0 
          push!(data,  network_input[1])
          push!(data,  network_input[2])
          push!(data,  network_input[3])
          push!(data,  network_input[4])
          push!(list_element, element)
        end
      end
    end
    return 1
  end
  

  #1d case
  function getX(limiter!,  u::AbstractArray{<:Any,3},semi)
    @unpack mesh, equations, solver, cache = semi
    @unpack alpha_max, alpha_min, alpha_smooth, variable = semi.solver.volume_integral.indicator
    @unpack alpha, alpha_tmp, indicator_threaded, modal_threaded = semi.solver.volume_integral.indicator.cache
    @unpack data,labels,list_element = limiter!

    # magic parameters
    threshold = 0.5 * 10^(-1.8 * (nnodes(solver))^0.25)
    parameter_s = log((1 - 0.0001)/0.0001)

    for element in eachelement(solver, cache)
      if alpha[element] < 0.0003
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

        x1 = (total_energy - total_energy_clip1)/total_energy
        x2 = (total_energy_clip1 - total_energy_clip2)/total_energy_clip1

        # directly normalize
        norm = max(maximum(abs.([x1,x2,nnodes(solver)])),1)
        x1 = x1 / norm
        x2 = x2 / norm
        x3 = nnodes(solver) / norm

        # if length(data) == 0
        #   push!(data,  0.0)
        #   push!(data,  0.0)
        #   push!(data,  1.0)
        #   push!(labels,  0.0)
        # end
          
        if (total_energy - total_energy_clip1) / total_energy > 0 || (total_energy_clip1 - total_energy_clip2)/total_energy_clip1 > 0
          push!(data,  x1)
          push!(data,  x2)
          push!(data,  x3)
          push!(list_element, element)
        end
      end
    end
    return 1
  end
  
end # @muladd
  