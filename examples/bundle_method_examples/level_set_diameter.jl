"""
	level_set_diameter(M, f, ∂f, p0; sub_solver::Function=augmented_Lagrangian_method, iter_cap::Int=60, random_seed::Int=42, distr_var::Real=1., show_err::Bool=true)

	estimates the diameter of the level set of f at p0 on the manifold M.

    # Arguments
    * `M` - the manifold on which to optimize.
    * `f` - the function whose level set has to be estimated.
    * `∂f` - a (sub)gradient of the function `f`.
    * `p0` - the point that defines the level set of `f`.
    * `sub_solver` - solver for the constrained optimization subprblem on the manifold `M×M`. The second option is exact_penalty_method.
    * `iter_cap` - maximum number of iterations for the constrained optimization subprblem on the manifold `M×M`.
    * `random_seed` - random seed for generating the data sample on `M×M` (for reproducibility reasons).
    * `distr_var` - the variance used in generating the data sample on `M×M` (for reproducibility reasons).
    * `show_err` - a boolean input to decide whether to print the error signaling the fallback to running the constrained optimization with `iter_cap`.
	* `debug_var` - a boolean input to decide whether to print the debug output of the subsolver
"""
function level_set_diameter(
    M,
    f,
    ∂f,
    p0;
    sub_solver::Function=augmented_Lagrangian_method,
    iter_cap::Int=60,
    random_seed::Int=42,
    distr_var::Real=1.0,
    show_err::Bool=false,
    debug_var::Bool=false,
)
    N = PowerManifold(M, NestedPowerRepresentation(), 2)
    Random.seed!(random_seed)
    initial_product_point = rand(N)
	set_component!(N, initial_product_point, p0, 1)
	set_component!(N, initial_product_point, p0, 2)
    G(N, q) = -distance(M, q[N, 1], q[N, 2])
    function gradG(N, q)
		if q[N, 1] ≈ q[N, 2]
			X = rand(N; vector_at = q)
			set_component!(N, X, -normal_cone_vector(M, q[N, 1]), 1)
			set_component!(N, X, -normal_cone_vector(M, q[N, 1]), 2)
			return X
		else
			Y = rand(N; vector_at = q)
			set_component!(N, Y, -grad_distance(M, q[N, 2], q[N, 1]), 1)
			set_component!(N, Y, -grad_distance(M, q[N, 1], q[N, 2]), 2)
			return Y
		end
	end
    H1(N, q) = f(M, q[N, 1]) - f(M, p0)
    function gradH1(N, q)
        r = rand(N)
        set_component!(N, r, ∂f(M, q[N, 1]), 1)
        set_component!(N, r, zero_vector(M, q[N, 1]), 2)
        return r
    end
    #gradH1(N, q) = hcat(∂f(M, q[N, 1]), zeros(representation_size(M)))
    H2(N, q) = f(M, q[N, 2]) - f(M, p0)
    function gradH2(N, q)
        r = rand(N)
        set_component!(N, r, zero_vector(M, q[N, 2]), 1)
        set_component!(N, r, ∂f(M, q[N, 2]), 2)
        return r
    end
    #gradH2(N, q) = hcat(zeros(representation_size(M)), ∂f(M, q[N, 2]))
    H(N, q) = [H1(N, q), H2(N, q)]
    gradH(N, q) = [gradH1(N, q), gradH2(N, q)]
    if debug_var
        pts = sub_solver(
            N,
            G,
            gradG,
            initial_product_point;
            g=H,
            grad_g=gradH,
            record=[:Iterate, :Cost],
            return_state=true,
            debug=[:Iteration, :Cost, :Stop, "\n"],
            stopping_criterion=StopWhenAny(
                StopWhenCostNan(),
                #StopWhenIterNan(),
                StopAfterIteration(iter_cap) |
                (StopWhenSmallerOrEqual(:ϵ, 1e-6) & StopWhenChangeLess(1e-6)),
            ),
        )
    else
        pts = sub_solver(
            N,
            G,
            gradG,
            initial_product_point;
            g=H,
            grad_g=gradH,
            record=[:Iterate, :Cost],
            return_state=true,
            # debug=[:Iteration, :Cost, :Stop, "\n"],
            stopping_criterion=StopWhenAny(
                StopWhenCostNan(),
                #StopWhenIterNan(),
                StopAfterIteration(iter_cap) |
                (StopWhenSmallerOrEqual(:ϵ, 1e-6) & StopWhenChangeLess(1e-6)),
            ),
        )
    end
    # end
    if show_err
        println("\n $err")
    end
    # Get the iterate corresponding to the last non-NaN value of the cost function
    if length(get_record_action(pts)[:Iterate]) > 1
        p_diam = get_record_action(pts)[:Iterate][end - 1]
    elseif !isnan(get_record_action(pts)[:Cost][1])
        p_diam = get_record_action(pts)[:Iterate][1]
    else
        p_diam = initial_product_point
    end
    return -G(N, p_diam)
end