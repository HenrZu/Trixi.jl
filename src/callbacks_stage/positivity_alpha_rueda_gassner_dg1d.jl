# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin

    function limiter_rueda_gassner!(u, alpha, mesh::AbstractMesh{1}, integrator, semi, limiter!)
        @unpack solver,  equations = semi 
        @unpack element_ids_dgfv = semi.cache
        @unpack alpha_max = semi.solver.volume_integral.indicator
        @unpack beta, stage = limiter!
        @unpack semi_fv, tolerance, iterations_newton, u_safe, node_dg, node_tmp, du_dα, dp_du  = limiter!.cache
        
        Δts = get_Δts(stage, integrator) #ssprk_get_Δts(stage, integrator)

        # calc pure FV solution to stage s
        get_usafe!(u_safe, integrator, semi_fv, stage) #ssprk_usafe!(u_safe, integrator, semi_fv, stage)
        u_safe = wrap_array(u_safe, semi_fv)
        # pure DG
        # no need to calculate pure dg sol:
        #        u = (1-α) u_dg + α u_FV
        # <-> u_dg = (u- α u_FV) / (1-α)

        # loop through elements      
        for element in element_ids_dgfv
          # if alpha is already max, theres no need to correct
            if abs(alpha[element] - alpha_max) < tolerance
                continue
            end

            # density 
            cor = 0.0
            # Iterate through all nodes and check for correction
            for j in eachnode(solver)
                ρ_safe = u_safe[1,j,element]
                if ρ_safe < 0
                    error("safe value for density not safe")
                end
                α_p = beta * ρ_safe - u[1,j,element]
                # detect if limiting is necessary
                # only correct if ap <= 0 is not fulfilled 
                if (α_p > tolerance)
                    println("Try to correct Density")
                    p_dg = (u[1,j,element] - alpha[element] * u_safe[1,j,element]) / (1 - alpha[element])
                    # avoid a divison by 0
                    if abs(u_safe[1,j,element] - p_dg) < tolerance
                        continue
                    end
                    tmp = α_p / ((u_safe[1,j,element] - p_dg))
                    cor = max(cor, tmp)
                    end
                
            end

          # Correct density
            if cor > 0          
            # check if alpha + correction is below alpha_max
                println("Stage $stage Korrigiert Element $element Dichte:  delta = $cor")
                println("alpha vorher $(alpha[element])")
                if alpha[element] + cor > alpha_max
                    cor =  alpha_max - alpha[element]
                    alpha[element] = alpha_max
                else
                    alpha[element] += cor 
                end
                println("alpha nachher $(alpha[element])")
                correct_u!(u, semi, element, u_safe, alpha, cor)      
            end


            cor = 0.0

            # pressure
            for j in eachnode(solver)

                tmp = 0.0
                _, _, p_safe = cons2prim(u_safe[:,j,element], equations)
                if p_safe < 0
                  error("safe value for pressure not safe")
                end

                _, v1, p_newton = cons2prim(u[:,j,element], equations) 
                α_p = beta * p_safe - p_newton
                if (α_p >  1e-11)
                    # Newton's method
                    # for Newton we need ∂p/∂α which can be calculated using the chain rule
                    # ∂p/∂α =  ∂p/∂u * ∂u/∂α
                    for newton_stage in 1:iterations_newton
                        println("Try to correct pressure")
                        # compute  ∂u/∂α
                        for vars in eachvariable(equations)
                            node_dg[vars] = (u[vars,j,element] - alpha[element] * u_safe[vars,j,element]) / (1 - alpha[element])
                            du_dα[vars] =  (u_safe[vars,j,element] - node_dg[vars])
                        end
                            
                        # compute  ∂p/∂u 
                        dp_du[1] = (equations.gamma - 1) * (0.5 * v1^2) 
                        dp_du[2] = (equations.gamma - 1) * (-v1)
                        dp_du[3] = (equations.gamma - 1)

                        dp_dα = dot(dp_du, du_dα)

                        if abs(dp_dα) < tolerance
                            break # continue
                        end

                        tmp +=  α_p / dp_dα

                        # calc new u 
                        for vars in eachvariable(equations)
                            node_tmp[vars]  = u[vars,j,element] + tmp * (u_safe[vars,j,element] - node_dg[vars])
                        end
                        # get pressure value
                        _, v1, p_newton = cons2prim(node_tmp, equations) 
                        α_p = beta * p_safe - p_newton

                        if α_p <= tolerance
                            break
                        end

                        if newton_stage == iterations_newton
                            error("Number of Iterations ($iterations_newton) not enough to correct pressure")
                        end
                    end
                    cor = max(cor, tmp)
                end
                
            end

            # Correct pressure
            if cor > 0
                println("Stage $stage Korrigiert Element $element Druck:  delta = $cor")
                println("alpha vorher $(alpha[element])")
                if alpha[element] + cor > alpha_max
                    cor =  alpha_max - alpha[element]
                    alpha[element] = alpha_max
                else
                    alpha[element] += cor 
                end
                println("alpha nachher $(alpha[element])")
                
                correct_u!(u, semi, element, u_safe, alpha, cor)

            end                   
        end
        return nothing
    end



    function upper_bound!(u, alpha, mesh::AbstractMesh{1}, integrator, semi, limiter!)
        @unpack solver,  equations = semi 
        @unpack element_ids_dgfv = semi.cache
        @unpack alpha_max = semi.solver.volume_integral.indicator
        @unpack beta, stage = limiter!
        @unpack semi_fv, tolerance, iterations_newton, u_safe, node_dg, node_tmp, du_dα, dp_du  = limiter!.cache

        beta = 1 + (1 - beta)
        
        Δts = get_Δts(stage, integrator) #ssprk_get_Δts(stage, integrator)

        # calc pure FV solution to stage s
        get_usafe!(u_safe, integrator, semi_fv, stage) #ssprk_usafe!(u_safe, integrator, semi_fv, stage)
        u_safe = wrap_array(u_safe, semi_fv)
        # pure DG
        # no need to calculate pure dg sol:
        #        u = (1-α) u_dg + α u_FV
        # <-> u_dg = (u- α u_FV) / (1-α)

        # loop through elements      
        for element in element_ids_dgfv
          # if alpha is already max, theres no need to correct
            if abs(alpha[element] - alpha_max) < tolerance
                continue
            end

            # density 
            cor = 0.0
            # Iterate through all nodes and check for correction
            for j in eachnode(solver)
                ρ_safe = u_safe[1,j,element]
                if ρ_safe < 0
                    error("safe value for density not safe")
                end
                α_p = beta * ρ_safe - u[1,j,element]
                # detect if limiting is necessary
                # only correct if ap <= 0 is not fulfilled 
                if (α_p < tolerance)
                    println("Try to correct Density")
                    p_dg = (u[1,j,element] - alpha[element] * u_safe[1,j,element]) / (1 - alpha[element])
                    # avoid a divison by 0
                    if abs(u_safe[1,j,element] - p_dg) < tolerance
                        continue
                    end
                    tmp = α_p / ((u_safe[1,j,element] - p_dg))
                    cor = max(cor, tmp)
                    end
                
            end

          # Correct density
            if cor > 0          
            # check if alpha + correction is below alpha_max
                println("Stage $stage Korrigiert Element $element Dichte:  delta = $cor")
                println("alpha vorher $(alpha[element])")
                if alpha[element] + cor > alpha_max
                    cor =  alpha_max - alpha[element]
                    alpha[element] = alpha_max
                else
                    alpha[element] += cor 
                end
                println("alpha nachher $(alpha[element])")
                correct_u!(u, semi, element, u_safe, alpha, cor)      
            end


            cor = 0.0

            # pressure
            for j in eachnode(solver)

                tmp = 0.0
                _, _, p_safe = cons2prim(u_safe[:,j,element], equations)
                if p_safe < 0
                  error("safe value for pressure not safe")
                end

                _, v1, p_newton = cons2prim(u[:,j,element], equations) 
                α_p = beta * p_safe - p_newton
                if (α_p <  1e-13)
                    # Newton's method
                    # for Newton we need ∂p/∂α which can be calculated using the chain rule
                    # ∂p/∂α =  ∂p/∂u * ∂u/∂α
                    for newton_stage in 1:iterations_newton
                        println("Try to correct pressure")
                        # compute  ∂u/∂α
                        for vars in eachvariable(equations)
                            node_dg[vars] = (u[vars,j,element] - alpha[element] * u_safe[vars,j,element]) / (1 - alpha[element])
                            du_dα[vars] =  (u_safe[vars,j,element] - node_dg[vars])
                        end
                            
                        # compute  ∂p/∂u 
                        dp_du[1] = (equations.gamma - 1) * (0.5 * v1^2) 
                        dp_du[2] = (equations.gamma - 1) * (-v1)
                        dp_du[3] = (equations.gamma - 1)

                        dp_dα = dot(dp_du, du_dα)

                        if abs(dp_dα) < tolerance
                            break # continue
                        end

                        tmp +=  α_p / dp_dα

                        # calc new u 
                        for vars in eachvariable(equations)
                            node_tmp[vars]  = u[vars,j,element] + tmp * (u_safe[vars,j,element] - node_dg[vars])
                        end
                        # get pressure value
                        _, v1, p_newton = cons2prim(node_tmp, equations) 
                        α_p = beta * p_safe - p_newton

                        if α_p >= tolerance
                            break
                        end

                        if newton_stage == iterations_newton
                            error("Number of Iterations ($iterations_newton) not enough to correct pressure")
                        end
                    end
                    cor = max(cor, tmp)
                end
                
            end

            # Correct pressure
            if cor > 0
                println("Stage $stage Korrigiert Element $element Druck:  delta = $cor")
                println("alpha vorher $(alpha[element])")
                if alpha[element] + cor > alpha_max
                    cor =  alpha_max - alpha[element]
                    alpha[element] = alpha_max
                else
                    alpha[element] += cor 
                end
                println("alpha nachher $(alpha[element])")
                
                correct_u!(u, semi, element, u_safe, alpha, cor)

            end                   
        end
        return nothing
    end

    
    function get_Δts(s, integrator)
        @unpack dt = integrator
        @unpack B1, B2end = integrator.cache.tab
        if s == 1
            Δts = B1 * dt
        else 
            Δts = B2end[s-1] * dt
        end
        return Δts
    end

    function ssprk_get_Δts(stage, integrator)
        @unpack dt = integrator
        #@unpack B1, B2end = integrator.cache.tab
        b = (1/6, 1/6, 1/6, 1/2)
        
        Δts = b[stage] * dt
        
        return Δts
    end

    # corresponds to b = (1/6, 1/6, 1/6, 1/2)

    function get_usafe!(u_safe, integrator, semi_fv, stage)
        @unpack t, dt, uprev, f = integrator
        @unpack k, tmp, williamson_condition, stage_limiter!, step_limiter! = integrator.cache
        @unpack A2end, B1, B2end, c2end = integrator.cache.tab

        # u1
        f(k, uprev, semi_fv, t)
        tmp = dt * k
        for i in 1:length(uprev)
            u_safe[i] = uprev[i] + B1 * tmp[i]
        end

        # other stages
        if stage > 1
            for i in 1:stage-1
                tmp = A2end[i] * tmp
                f(k, u_safe, semi_fv, t + c2end[i] * dt)
                tmp += dt * k
                for d in 1:length(uprev)
                    u_safe[d] = u_safe[d] + B2end[i] * tmp[d]
                end
            end
        end
    end

    function ssprk_usafe!(u_safe, integrator, semi_fv, stage)
        @unpack t,dt,uprev,f, = integrator
        @unpack k,fsalfirst,utilde,atmp,stage_limiter!,step_limiter! = cache
        @unpack one_third_u, two_thirds_u, half_u, half_t = cache.tab
        dt_2 = half_t * dt
      
        # u1
        f( fsalfirst,  uprev, semi_fv, t)
        for d in 1:length(uprev)
            u_safe[d] = uprev[d] + dt_2*fsalfirst[d]
        end
        if stage > 1 
            f( k,  u_safe, semi_fv, t+dt_2)
            # u2
            for d in 1:length(uprev)
                u_safe[d] = u_safe[d] + dt_2*k[d]
            end
            if stage > 2
                f( k, u_safe, semi_fv, t+dt)
                #
                for d in 1:length(uprev)
                    u_safe[d] = u_safe[d] + dt_2*k[d]
                end
                if stage > 3
                    if integrator.opts.adaptive
                        utilde = one_third_u * uprev + two_thirds_u * u # corresponds to bhat = (1/3, 1/3, 1/3, 0)
                    end
                    # u3
                    for d in 1:length(uprev)
                        u_safe[d] = two_thirds_u * uprev[d] + one_third_u * u_safe[d]
                    end
                    f( k,  u_safe, semi_fv, t+dt_2)
                    # corresponds to b = (1/6, 1/6, 1/6, 1/2)
                    for d in 1:length(uprev)
                        u_safe[d] = u_safe[d] + dt_2*k[d]
                    end 
                end
            end
        end
    end


    function correct_u!(u::AbstractArray{<:Any,3}, semi, element, u_safe, alpha, cor)
        @unpack solver, equations = semi
        for i in eachnode(solver)
            for vars in eachvariable(equations)
                sol_dg = (u[vars,i,element] - alpha[element] * u_safe[vars,i,element]) / (1 - alpha[element])
                u[vars,i,element] = u[vars,i,element] + cor * (u_safe[vars,i,element] - sol_dg)
            end            
        end 
    end

end # @muladd

