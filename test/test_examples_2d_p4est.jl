module TestExamples2dP4est

using Test
using Trixi

include("test_trixi.jl")

# pathof(Trixi) returns /path/to/Trixi/src/Trixi.jl, dirname gives the parent directory
EXAMPLES_DIR = joinpath(pathof(Trixi) |> dirname |> dirname, "examples", "p4est_2d_dgsem")

@testset "P4estMesh" begin
  @trixi_testset "elixir_advection_basic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_basic.jl"),
      l2   = [9.14468177884088e-6],
      linf = [6.437440532947036e-5])
  end

  @trixi_testset "elixir_advection_nonconforming.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming.jl"),
      l2   = [2.58174252995893e-5],
      linf = [2.65115939055204e-4])
  end

  @trixi_testset "elixir_advection_nonconforming_unstructured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_nonconforming_unstructured.jl"),
      l2   = [3.038384623519386e-3],
      linf = [6.324792487776842e-2])
  end

  @trixi_testset "elixir_advection_amr_solution_independent.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_solution_independent.jl"),
      l2   = [7.945073295691892e-5],
      linf = [0.0007454287896710293])
  end

  @trixi_testset "elixir_advection_amr_unstructured_flag.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_amr_unstructured_flag.jl"),
      l2   = [1.29561551e-03],
      linf = [3.92061383e-02])
  end

  @trixi_testset "elixir_advection_restart.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_advection_restart.jl"),
      l2   = [6.398955192910044e-6],
      linf = [3.474337336717426e-5])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic.jl"),
      l2   = [2.3653424742684444e-6, 2.1388875095440695e-6, 2.1388875095548492e-6, 6.010896863397195e-6],
      linf = [1.4080465931654018e-5, 1.7579850587257084e-5, 1.7579850592586155e-5, 5.956893531156027e-5])
  end

  @trixi_testset "elixir_euler_source_terms_nonperiodic_unstructured.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_source_terms_nonperiodic_unstructured.jl"),
      l2   = [0.005689496253354319, 0.004522481295923261, 0.004522481295922983, 0.009971628336802528],
      linf = [0.05125433503504517, 0.05343803272241532, 0.053438032722404216, 0.09032097668196482])
  end

  @trixi_testset "elixir_euler_free_stream.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_free_stream.jl"),
      l2   = [2.063350241405049e-15, 1.8571016296925367e-14, 3.1769447886391905e-14, 1.4104095258528071e-14],
      linf = [1.9539925233402755e-14, 2e-12, 4.8e-12, 4e-12],
      atol = 1.5e-12, # required to make CI tests pass on macOS
    )
  end

  @trixi_testset "elixir_euler_shockcapturing_ec.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_shockcapturing_ec.jl"),
      l2   = [9.53984675e-02, 1.05633455e-01, 1.05636158e-01, 3.50747237e-01],
      linf = [2.94357464e-01, 4.07893014e-01, 3.97334516e-01, 1.08142520e+00],
      tspan = (0.0, 1.0))
  end

  @trixi_testset "elixir_euler_sedov.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_euler_sedov.jl"),
      l2   = [3.70025991e-01, 2.43469627e-01, 2.43469627e-01, 1.28658401e+00],
      linf = [1.10363921e+00, 1.18898311e+00, 1.18898311e+00, 6.19710862e+00],
      tspan = (0.0, 0.3))
  end

  @trixi_testset "elixir_eulergravity_convergence.jl" begin
    @test_trixi_include(joinpath(EXAMPLES_DIR, "elixir_eulergravity_convergence.jl"),
      l2   = [0.0002456507718735638, 0.0003374236900407321, 0.00033742369004074064, 0.0007198398465056179],
      linf = [0.0013653498212873316, 0.0018996803803537077, 0.0018996803803517093, 0.004384818684607161],
      tspan = (0.0, 0.1))
  end
end

end # module
