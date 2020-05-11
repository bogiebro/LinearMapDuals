module Tests
using Test
using LinearMapDuals
using LinearMaps
using ForwardDiff
using ReTester
using ToeplitzMatrices

function compare_vals_and_grads(f, lm_f, params)
  result = LinearMapResult{Float64}()
  ForwardDiff.jacobian!(result, lm_f, params)
  @test all(Array(result.val) .≈ f(params))
  fd_jacobian = ForwardDiff.jacobian(f, params)
  for (i, grad)  in enumerate(result.jacobians)
    @test all(fd_jacobian[:, i] .≈ reshape(Array(grad), :))
  end
end

function test()
  break_on(:error)

  params = randn(6);
  @testset "wrapper" begin
    f(x) = x * x'
    compare_vals_and_grads(f, LinearMap ∘ f, params)
  end

  @testset "kronecker" begin
    function f(x)
      y = reshape(x, (2,3))
      kron(y,y)
    end
    function lm_f(x)
      y = LinearMap(reshape(x, (2,3)))
      kron(y, y)
    end
    compare_vals_and_grads(f, lm_f, params)
  end

  @testset "multi-kronecker" begin
  break_on(:error)
    function f(x)
      y = reshape(x, (2,3))
      kron(kron(y,y), y)
    end
    function lm_f(x)
      y = LinearMap(reshape(x, (2,3)))
      @run kron(y, y, y)
    end
    compare_vals_and_grads(f, lm_f, params)
  end

  @testset "toeplitz" begin
    function lm_f(x)
      SymmetricToeplitz(x)
    end
    result = LinearMapResult{Float64}()
    ForwardDiff.jacobian!(result, lm_f, params)
    # TODO: compare to manual Toeplitz construction
  end

end

end # module
