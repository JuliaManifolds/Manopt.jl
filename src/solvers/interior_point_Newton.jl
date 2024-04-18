mutable struct ConstrainedProblem
    M::AbstractManifold

    # TO DO: combine constrained and Hessian objective into one
    co::ConstrainedManifoldObjective
    ho::ManifoldHessianObjective
    dims::Tuple{Int,Int}
end

function get_manifold(Pr::ConstrainedProblem)
    return Pr.M
end

function get_product_manifold(Pr::ConstrainedProblem)
    M, (m, n) = Pr.M, Pr.dims
    return M × ℝ^m × ℝ^n × ℝ^m
end

function get_subsolver_manifold(Pr::ConstrainedProblem)
    M, n = Pr.M, Pr.dims[2]
    return M × ℝ^n
end

function evaluate_cost_function(Pr::ConstrainedProblem, p)
    return get_cost(Pr.M, Pr.co, p)
end

function evaluate_cost_gradient(Pr::ConstrainedProblem, p)
    return get_gradient(Pr.M, Pr.co, p)
end

function set_cost_gradient!(Pr::ConstrainedProblem, p, X)
    get_gradient!(Pr.M, Pr.co, p, X)
end

function evaluate_cost_Hessian(Pr::ConstrainedProblem, p, X)
    return get_hessian(Pr.M, Pr.ho, p, X)
end

function set_cost_Hessian!(Pr::ConstrainedProblem, p, X, Y)
    get_hessian!(Pr.M, Pr.ho, p, X, Y)
end

function evaluate_inequality_function(Pr::ConstrainedProblem, p)
    return get_inequality_constraints(Pr.M, Pr.co, p)
end

function evaluate_equality_function(Pr::ConstrainedProblem, p)
    return get_equality_constraints(Pr.M, Pr.co, p)
end

function evaluate_inequality_function_differential(Pr::ConstrainedProblem, p)
    return get_grad_inequality_constraints(Pr.M, Pr.co, p)
end

function evaluate_equality_function_differential(Pr::ConstrainedProblem, p)
    return get_grad_equality_constraints(Pr.M, Pr.co, p)
end

function get_dims(Pr::ConstrainedProblem)
    return Pr.dims
end

function set_dims!(Pr::ConstrainedProblem, m, n)
    Pr.dims = (m, n)
end

mutable struct InteriorPointState

    # point on product manifold
    q::ArrayPartition

    # product of ρ and σ will be the barrier parameter b
    ρ::Real
    σ::Real

    stop::StoppingCriterion
    retr::AbstractRetractionMethod
    step::Stepsize

end

function get_variables(state::InteriorPointState)
    return state.q.x
end

function get_auxillary_variables(state::InteriorPointState)
    return state.ρ, state.σ
end

function calculate_σ(Pr::ConstrainedProblem, q)

    p, μ, λ, s = q

    g_p = evaluate_inequality_function(Pr, p)
    h_p = evaluate_equality_function(Pr, p)
    dg_p = evaluate_inequality_function_differential(Pr, p) 
    dh_p = evaluate_equality_function_differential(Pr, p) 

    F_p = evaluate_cost_gradient(Pr, p)
    F_μ = Float64[]
    F_λ = Float64[]
    F_s = Float64[]

    if m > 0
        F_p += dg_p'μ
        F_μ  = g_p + s
        F_s  = μ.*s
    end

    if n > 0
        F_p += dh_p'λ
        F_λ  = h_p
    end
    
    F_q = [F_p; F_μ; F_λ; F_s]

    return min(0.5, sqrt(norm(F_q)))
end
    
function InteriorPointState(Pr::ConstrainedProblem)

    m, n = get_dims(Pr) 

    # initial point, temporarily fixed to ensure feasibility
    p_1, p_2 = 2*rand(2).-1
    p_3 = rand()
    p = [p_1, p_2, p_3]
    p /= norm(p)

    μ = rand(m) 
    λ = rand(n) * (n > 0)
    s = rand(m)

    q = ArrayPartition(p, μ, λ, s)

    ρ = μ's / m
    σ = calculate_σ(Pr, [p, μ, λ, s])

    # product manifold
    N = get_product_manifold(Pr)

    stop = StopAfterIteration(100)
    retr = default_retraction_method(N)
    step = ArmijoLinesearch(N; retraction_method=retr, initial_stepsize=0.01)

    return InteriorPointState(q, ρ, σ, stop, retr, step)
end

function RHS(   Pr::ConstrainedProblem, 
             state::InteriorPointState)

    p, μ, λ, s = get_variables(state)

    ρ, σ = get_auxillary_variables(state)
    
    g_p = evaluate_inequality_function(Pr, p)
    dg_p = evaluate_inequality_function_differential(Pr, p) 
    dh_p = evaluate_equality_function_differential(Pr, p) 
  
    X_1 = evaluate_cost_gradient(Pr, p)
    X_2 = Float64[]


    m, n = get_dims(Pr)

    if m > 0
        X_1 += dg_p' * ((g_p + s).*μ .+ (ρ*σ) - μ) ./ s
    end
    

    if n > 0
        X_2 += dh_p'λ
    end

    return ArrayPartition(X_1, X_2)
end

function LHS(   Pr::ConstrainedProblem, 
             state::InteriorPointState,
               X_p)

    p, μ, λ, s = get_variables(state) 
          
    dg_p = evaluate_inequality_function_differential(Pr, p) 
    dh_p = evaluate_equality_function_differential(Pr, p)

    Y_1 = evaluate_cost_Hessian(Pr, p, X_p)
    Y_2 = Float64[]

    m, n = get_dims(Pr)

    if m > 0
        Y_1 += (dg_p' * dg_p * X_p) .* μ ./ s # plus sum_i μ_i Hess g_i(p)
    end

    if n > 0
        Y_2  = dh_p*X_p
    end
    
    return ArrayPartition(Y_1, Y_2)
end
    
function is_feasible(Pr::ConstrainedProblem, p)
    # evaluate constraint functions at p
    g_p = evaluate_inequality_function(Pr, p)
    h_p = evaluate_equality_function(Pr, p)

    # check feasibility
    return is_point(Pr.M, p) && all(g_p .<= 0) && all(h_p .== 0)
end

function step_solver(   Pr::ConstrainedProblem,
                     state::InteriorPointState)

    p, μ, λ, s = get_variables(state)
    ρ, σ = get_auxillary_variables(state) 

    N_sub = get_subsolver_manifold(Pr)
    q_sub = ArrayPartition(p, λ)

    rhs = RHS(Pr, state)
    lhs = X -> LHS(Pr, state, X)

    X_init = rand(N_sub, vector_at = q_sub)
    X_p, X_λ = subsolver(N_sub, q_sub, lhs, X_init, -rhs).x
    
    g_p = evaluate_inequality_function(Pr, p)
    dg_p = evaluate_inequality_function_differential(Pr, p)
    
    X_μ = ((dg_p*X_p + g_p).*μ .+ (ρ*σ) - μ)./s
    X_s = ((ρ*σ) .- μ.*s - s.*X_μ) ./ μ

    N = get_product_manifold(Pr)
    X = ArrayPartition(X_p, X_μ, X_λ, X_s)

    # update iterate 
    state.q = retract(N, state.q, X, state.retr)
    state.ρ = μ's / m
    state.σ = calculate_σ(Pr, state.q.x)

    return state
end

# obnjugate residudal method, 
function subsolver(M, p, A, x_0, b)

    # iteration count
    k = 0

    #initial residual
    r_0 = b - A(x_0)

    q_0 = r_0
    Ar_0 = A(r_0)
    Aq_0 = A(q_0)
    
    r = r_0
    q = q_0
    Ar = Ar_0
    Aq = Aq_0
    
    x = x_0

    metric = (X, Y) -> inner(M, p, X, Y) 
    
    while metric(r,r) > 1e-5

        α = metric(r, Ar) / metric(Aq, Aq)
        x += α*q
        
        r_next = r - α*Aq
        Ar_next = A(r_next)
        β = metric(r_next, Ar_next) / metric(r, Ar)
        q_next = r_next + β*q
        Aq_next = Ar_next + β*Aq

        r  = r_next
        q  = q_next
        Ar = Ar_next
        Aq = Aq_next

        k += 1
    end
    return x
end

function interior_point_Newton(Pr::ConstrainedProblem)

    # initial state
    state = InteriorPointState(Pr)
    print("Point 0: ", state.q.x[1], '\n')
    for i in range(1, 20)
        state = step_solver(Pr, state)
        print("Point ", i, ": ", state.q.x[1], '\n')
    end
end



