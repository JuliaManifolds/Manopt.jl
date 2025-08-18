"""
Example demonstrating the RDoG (Riemannian Distance over Gradients) stepsize schedule.

This learning-rate-free optimization method adapts automatically without hyperparameter tuning.
"""

using Manopt
using Manifolds
using LinearAlgebra
using Random

maxiter = 1000
write_every = 50

Random.seed!(42)

# Setup: Rayleigh quotient minimization on the sphere
# We want to find the dominant eigenvector of a symmetric matrix
d = 30
M = Sphere(d - 1)

# Create a symmetric matrix with known eigenvalues
A = randn(d, d)
A = A' * A  # Make it symmetric positive definite

# Compute the true optimal value (negative of largest eigenvalue)
eigenvalues = eigvals(A)
true_optimal = -maximum(eigenvalues)
println("True optimal value (negative largest eigenvalue): ", true_optimal)

# Define the cost function (negative Rayleigh quotient)
f(M, p) = -p' * A * p

# Define the Riemannian gradient
function grad_f(M, p)
    euc_grad = -2 * A * p
    return euc_grad - dot(euc_grad, p) * p  # Project to tangent space
end

# Initial point
p0 = rand(M)
initial_cost = f(M, p0)
initial_gap = initial_cost - true_optimal

println("RDoG Example: Rayleigh Quotient Minimization")
println("=" ^ 50)
println("Manifold: Sphere($(d-1))")
println("True optimal value: ", true_optimal)
println("Initial cost: ", initial_cost)
println("Initial optimality gap: ", initial_gap)

# Example 1: Basic RDoG without curvature
println("\n1. Basic RDoG (no curvature)")
println("-" ^ 30)

result1 = gradient_descent(
    M, f, grad_f, p0;
    stepsize=RDoG(M; initial_distance=1e-2, use_curvature=false),
    stopping_criterion=StopAfterIteration(maxiter) | StopWhenGradientNormLess(1e-6),
    debug=[
        :Iteration,
        " | ",
        (:Cost, "f(x) = %.12f"),
        " | ",
        (:GradientNorm, "||grad f|| = %.8e"),
        " | ",
        (:Stepsize, "step = %.8e\n"),
        write_every,  # Print every 20 iterations
    ]
)
final_cost1 = f(M, result1)
final_gap1 = final_cost1 - true_optimal
println("Final cost: ", final_cost1)
println("Final optimality gap: ", final_gap1)
println("Gap reduction factor: ", abs(initial_gap / final_gap1))

# Example 2: RDoG with curvature
println("\n2. RDoG with curvature")
println("-" ^ 30)

# The sphere has constant sectional curvature κ = 1
result2 = gradient_descent(
    M, f, grad_f, p0;
    stepsize=RDoG(M; initial_distance=1e-2, use_curvature=true, sectional_curvature_bound=1.0),
    stopping_criterion=StopAfterIteration(maxiter) | StopWhenGradientNormLess(1e-6),
    debug=[
        :Iteration,
        " | ",
        (:Cost, "f(x) = %.12f"),
        " | ",
        (:GradientNorm, "||grad f|| = %.8e"),
        " | ",
        (:Stepsize, "step = %.8e\n"),
        write_every,  # Print every 20 iterations
    ]
)
final_cost2 = f(M, result2)
final_gap2 = final_cost2 - true_optimal
println("Final cost: ", final_cost2)
println("Final optimality gap: ", final_gap2)
println("Gap reduction factor: ", abs(initial_gap / final_gap2))

# Example 3: Comparison with fixed stepsize
println("\n3. Fixed stepsize for comparison")
println("-" ^ 30)

result3 = gradient_descent(
    M, f, grad_f, p0;
    stepsize=ConstantLength(M, 0.01),
    stopping_criterion=StopAfterIteration(maxiter) | StopWhenGradientNormLess(1e-6),
    debug=[
        :Iteration,
        " | ",
        (:Cost, "f(x) = %.12f"),
        " | ",
        (:GradientNorm, "||grad f|| = %.8e"),
        " | ",
        (:Stepsize, "step = %.8e\n"),
        write_every,  # Print every 20 iterations
    ]
)
final_cost3 = f(M, result3)
final_gap3 = final_cost3 - true_optimal
println("Final cost: ", final_cost3)
println("Final optimality gap: ", final_gap3)
println("Gap reduction factor: ", abs(initial_gap / final_gap3))

# Example 4: RDoG with different initial distances
println("\n4. Sensitivity to initial distance")
println("-" ^ 30)

println("Running shorter tests (200 iterations) with different initial distances:")
for eps in [1e-5, 1e-3, 1e-1]
    result = gradient_descent(
        M, f, grad_f, p0;
        stepsize=RDoG(M; initial_distance=eps, use_curvature=false),
        stopping_criterion=StopAfterIteration(200) | StopWhenGradientNormLess(1e-6),
        debug=[]
    )
    final_cost = f(M, result)
    final_gap = final_cost - true_optimal
    println("ε = $eps: Final gap = $(final_gap), Reduction = $(abs(initial_gap / final_gap))x")
end
