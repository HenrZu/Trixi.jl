# By default, Julia/LLVM does not use fused multiply-add operations (FMAs).
# Since these FMAs can increase the performance of many numerical algorithms,
# we need to opt-in explicitly.
# See https://ranocha.de/blog/Optimizing_EC_Trixi for further details.
@muladd begin


    function limiter_rueda_gassner!(u, integrator, semi, limiter!)
        @unpack volume_flux_fv = semi.solver.volume_integral 
        @unpack basis = semi.solver 
        @unpack mesh, equations, initial_condition = semi 
        @unpack surface_flux = semi.solver.surface_integral
      # @unpack one_third_u, two_thirds_u, half_t =  integrator.cache.tab
        @unpack element_ids_dgfv = semi.cache
        @unpack alpha = semi.solver.volume_integral.indicator.cache
        @unpack alpha_max = semi.solver.volume_integral.indicator
        @unpack semi_fv, semi_dg, beta, s, tolerance, n_iterations, u_safe, node_dg, node_tmp, du_dα, dp_du  = limiter!

        @unpack t, dt, uprev, f = integrator
        @unpack k, tmp, williamson_condition, stage_limiter!, step_limiter! = integrator.cache
        @unpack A2end, B1, B2end, c2end = integrator.cache.tab

      
      
      
        if s == 1
            Δts = B1 * dt
        else 
            Δts = B2end[s] * dt
        end
      
      # pure FV
      
        arr = size(u)
      # u_safe = sol_stage(u_safe, integrator, semi_fv, s)
      # u_safe = wrap_array(u_safe, semi_fv)
      # u_safe = zeros(arr[1],arr[2],arr[3],arr[4])
      # u1
        f(k, uprev, semi, t)
        tmp = dt * k
        for i in 1:length(uprev)
            u_safe[i] = uprev[i] + B1 * tmp[i]
        end
      # u_safe   = uprev + B1*tmp
      # other stages
        if s > 1
            for i in 1:s-1
                tmp = A2end[i] * tmp
                f(k, uprev, semi, t + c2end[i] * dt)
                tmp += dt * k
                for d in 1:length(uprev)
                    u_safe[d] = u_safe[d] + B2end[i] * tmp[d]
                end
            # u_safe   = u_safe + B2end[i]*tmp
            end
        end

      # print(u_safe)
      # error("u printed")
      

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
            cor = 0
          # Iterate through all LGL Points and check for correction
            for j in 1:arr[2]
                for k in 1:arr[3]
            # get safe density value 
                    ρ_safe = u_safe[1,j,k,element]
                    if ρ_safe < 0
                        error("safe value for density not safe")
                    end
                    α_p = beta * ρ_safe - u[1,j,k,element]
                    if (α_p > 1e-11)
                    # avoid a divison by 0
                        p_dg = (u[1,j,k,element] - alpha[element] * u_safe[1,j,k,element]) / (1 - alpha[element])
                        if abs(u_safe[1,j,k,element] - p_dg) < tolerance
                            continue
                        end

                        tmp = α_p / (Δts * (u_safe[1,j,k,element] - p_dg))
                        cor = max(cor, tmp)
                    end
                end
            end

          # Correct density
            if cor > 0          
            # check if alpha + correction is below alpha_max
                if alpha[element] + cor > alpha_max
                    cor =  alpha_max - alpha[element]
                    alpha[element] = alpha_max
                end
                # println("Stage $s Korrigiert Dichte:  delta = $cor")
                for j in 1:arr[2]
                    for k in 1:arr[3]
                        println("--------------Dichte korrigieren-----------------------")
                        # println("--------------------------------------")
                        # println("Dichte vorher $(u[1,j,k,element])")
                        # println("Safe density; $(u_safe[1,j,k,element])")

                    # u[:,k,element] += cor * dt_2 * (u_safe[:,k,element]- u_dg[:,k,element]),
                    # u_dg = (u- α u_FV) / (1-α)
                        for vars = 1:4
                            node_dg[vars] = (u[vars,j,k,element] - alpha[element] * u_safe[vars,j,k,element]) / (1 - alpha[element])
                            u[vars,j,k,element] = u[vars,j,k,element] + cor * (u_safe[vars,j,k,element] - node_dg[vars])
                        end
                    


                        # println("Dichte nachher $(u[1,j,k,element])")
                        println("ap neu =  $(beta * u_safe[1,j,k,element] - u[1,j,k,element] )")
                        # println("--------------------------------------")
                        # println("")
                    end 
                end          
            end


            cor = 0.0
          # pressure
            for j in 1:arr[2]
                for k in 1:arr[3] # eachnode
                    tmp = 0.0
                    

                    _, _, _, p_safe = cons2prim(u_safe[:,j,k,element], equations)
                    # if p_safe < 0
                    #     for vars = 1:4
                    #         node_dg[vars] = (u[vars,j,k,element] - alpha[element] * u_safe[vars,j,k,element]) / (1 - alpha[element])
                    #     end
                    #   _, _, _, p_blend = cons2prim(u[:,j,k,element], equations)
                    #   println("aktuelle (Druck = $p_blend) lsg u")
                    #   print(u[:,j,k,element])
                    #   println("U Safe (Druck = $p_safe) = ")
                    #   print(u_safe[:,j,k,element])
                    #   println("DG LSG = $(node_dg) ")
                    #   println("alpha = $(alpha[element]) ")
                    #   print("Zeit $t")
                    #   print("Stage $s")
                    #   error("safe value for pressure not safe")
                    # end

                    _, v1, v2, p_newton = cons2prim(u[:,j,k,element], equations) 
                    α_p = beta * p_safe - p_newton
                    if (α_p >  1e-11)
                        # Newton's method
                        # for Newton we need ∂p/∂α which can be calculated using the chain rule
                        # ∂p/∂α =  ∂p/∂u * ∂u/∂α
                        # println("----------------------------------------")
                        # println("p vor newton $p_newton")
                        # println("ap vor newton $α_p")
                        # println("Δts =  $(Δts)")
                        # println("----------------------------------------")
                        for newton_stage in 1:n_iterations
                            # du_dα = dt_2 * (u_safe[:,k,element]- u_dg[:,k,element])

                            if newton_stage > 1
                                _, v1, v2, p_newton = cons2prim(node_tmp, equations) 
                            end

                            for vars = 1:4
                                node_dg[vars] = (u[vars,j,k,element] - alpha[element] * u_safe[vars,j,k,element]) / (1 - alpha[element])
                                du_dα[vars] =  Δts * (u_safe[vars,j,k,element] - node_dg[vars])
                            end
                            
                            
                            
                            # println("element $(u[:,j,k,element])")
                            dp_du[1] = (equations.gamma - 1) * (0.5 * sum(v1^2 + v2^2))
                            dp_du[2] = (equations.gamma - 1) * (-v1)
                            dp_du[3] = (equations.gamma - 1) * (-v2)
                            dp_du[4] = (equations.gamma - 1)

                            dp_dα = dot(dp_du, du_dα)

                            if abs(dp_dα) < tolerance
                                break # continue
                            end

                            tmp +=  α_p / dp_dα

                            # calc new u 
                            for vars = 1:4
                                node_tmp[vars]  = u[vars,j,k,element] + tmp * (u_safe[vars,j,k,element] - node_dg[vars])
                            end
                            # get pressure value
                            α_p = beta * p_safe - p_newton
                            # println("psafe =  $p_safe mit goal $(beta * p_safe)")
                            # println("p nach newton step ($newton_stage) = $p_newton")
                            # println("ap = $α_p")
                            # println("----------------------------------------")
                            if α_p <= 0
                                break
                            end
                            if newton_stage == n_iterations
                                error("Number of Iterations ($n_iterations) not enough to correct pressure")
                            end
                        end
                        cor = max(cor, tmp)
                    end
                end
            end

            # FEHLER MIT END

            # Correct pressure
            if cor > 0
                if alpha[element] + cor > alpha_max
                    cor =  alpha_max - alpha[element]
                    alpha[element] = alpha_max
                end
                for j in 1:arr[2]
                    for k in 1:arr[3]
                        for vars = 1:4
                          node_dg[vars] = (u[vars,j,k,element] - alpha[element] * u_safe[vars,j,k,element]) / (1 - alpha[element])
                          u[vars,j,k,element] += cor * (u_safe[vars,j,k,element] - node_dg[vars])
                        end
                        println("correct pressure")
                        _, _, _, p_safe = cons2prim(u_safe[:,j,k,element], equations)
                        # println("Safe value; $p_safe")
                        _, _, _, p_new = cons2prim(u[:,j,k,element], equations) 
                        # println("Druck nachher $p_new")
                        println("Nachher ap =  $(beta * p_safe - p_new )")
                        # println("--------------------------------------")
                        # println("")
                    end   
                end
            end                   
        end
        return nothing
    end

    


    function sol_stage(u, integrator, semi, s, a)
        @unpack t, dt, uprev, f = integrator
        @unpack k, fsalfirst, utilde, atmp, stage_limiter!, step_limiter! = integrator.cache
        @unpack one_third_u, two_thirds_u, half_u, half_t = integrator.cache.tab
        dt_2 = half_t * dt

  # u1
        f(fsalfirst,  uprev, semi, t)
        u = uprev + dt_2 * fsalfirst
        if s > 1
            f(k,  u, semi, t + dt_2)
    # u2
            u = u + dt_2 * k
            if s > 2
                f(k,  u, semi, t + dt)
      # 
                u = u + dt_2 * k
                if s > 3
        # u3
                    u = two_thirds_u * uprev + one_third_u * u
                    f(k,  u, semi, t + dt_2)
                    u = u + dt_2 * k # corresponds to b = (1/6, 1/6, 1/6, 1/2)   
                end
            end
        end
        return u
    end

    function sol_stage(u, integrator, semi, s)
        @unpack t, dt, uprev, f = integrator
        @unpack k, tmp, williamson_condition, stage_limiter!, step_limiter! = integrator.cache
        @unpack A2end, B1, B2end, c2end = integrator.cache.tab

  # u1
        f(k, uprev, semi, t)
        tmp = dt * k
        u   = uprev + B1 * tmp
  # other stages
        if s > 1
            for i in 1:s - 1
                if williamson_condition
                    f(ArrayFuse(tmp, u, (A2end[i], dt, B2end[i])), u, semi, t + c2end[i] * dt)
                else
                    tmp = A2end[i] * tmp
                    f(k, u, semi, t + c2end[i] * dt)
                    tmp += dt * k
                    u   = u + B2end[i] * tmp
                end
            end
        end
        return u
    end 

end # @muladd



# integrator.f(integrator.k[1], integrator.uprev, semi_dg, integrator.t + dt)
      # if s == 4
      #   u_prev_tmp = wrap_array(integrator.uprev,semi_fv)
      #   k_tmp = wrap_array(integrator.k[1],semi)
      #   u_dg = (two_thirds_u * u_prev_tmp + one_third_u * u) + dt_2 * k_tmp
      # else
      #   u_dg  = wrap_array(integrator.uprev + dt_2*integrator.k[1], semi_dg)
      # end
      # u_dg = zeros(arr[1],arr[2],arr[3])
      # u_dg = sol_stage(u_dg, integrator, semi_dg, s)
      # u_dg = wrap_array(u_dg, semi_dg)


  # dt_2 = half_t * integrator.dt
      
      # if s == 1
      #   dt = 0
      #   x = B1*tmp
      # elseif s == 2 || s == 4
      #   dt = dt_2
      # else 
      #   dt = integrator.dt
      # end


      # FV
       # integrator.f(integrator.k[1], integrator.uprev, semi_fv, integrator.t + dt)
      # if s == 4
      #   u_prev_tmp = wrap_array(integrator.uprev,semi_fv)
      #   k_tmp = wrap_array(integrator.k[1],semi)
      #   u_safe = (two_thirds_u * u_prev_tmp + one_third_u * u) + dt_2 * k_tmp
      # else
      #   u_safe = wrap_array(integrator.uprev + dt_2*integrator.k[1],semi_fv)
      # end