mutable struct InteriorPointProblem
    M::AbstractManifold
    ob::ConstrainedManifoldObjective
    dims::Tuple{Int,Int}
end

mutable struct InteriorPointParams{Q}
    q::Q    
    b::Real
end

function InteriorPointParams(problem::InteriorPointProblem)
    M = problem.M
    m, n = problem.dims
    q = rand(M × ℝ^m × ℝ^n × ℝ^m)
    b = rand()
    return InteriorPointParams(q, b)
end

mutable struct InteriorPointLagrangian
    problem::InteriorPointProblem
    params::InteriorPointParams
end

function grad(L::InteriorPointLagrangian)

    M = L.problem.M
    ob = L.problem.ob

    p, μ, λ, s = L.params.q.x
    b = L.params.b

    gp = get_inequality_constraints(M, ob, p)
    hp = get_equality_constraints(M, ob, p)
    Jgp = get_grad_inequality_constraints(M, ob, p) 
    Jhp = get_grad_equality_constraints(M, ob, p)

    m, n = L.problem.dims

    Xp = get_gradient(M, ob, p)
    Xμ = Float64[]
    Xλ = Float64[]
    Xs = Float64[]

    if m > 0
        Xp += Jgp'μ
        Xμ = gp + s
        Xs = μ - b*ones(m)./s
    end
    if n > 0
        Xp += Jhp'λ
        Xλ = hp
    end

    return ArrayPartition(Xp, Xμ, Xλ, Xs)
end

function Hess(L::InteriorPointLagrangian, Xq)
    ob = L.problem.ob
    p, μ, λ, s = L.params.q.x
    b = L.params.b
    Xp, Xμ, Xλ, Xs = Xq.x
    Jgp = get_grad_inequality_constraints(M,ob, p) 
    Jhp = get_grad_equality_constraints(M,ob, p)

    m = length(μ)
    n = length(λ)

    Yp = zero_vector(M, p)
    Yμ = Float64[]
    Yλ = Float64[]
    Ys = Float64[]

    if m > 0
        Yp += Jgp'Xμ
        Yμ = Jgp*Xp + Xs
        Ys = Xμ + b*Xs ./ (s.*s)
    end
    if n > 0
        Yp += Jhp'Xλ
        Yλ = Jhp*Xp
    end
    
    return ArrayPartition(Yp, Yμ, Yλ, Ys)
end

mutable struct InteriorPointState
    L::InteriorPointLagrangian
    stopping_criterion::StoppingCriterion
    retraction_method::AbstractRetractionMethod
    step_size::Stepsize
end

function InteriorPointState(L::InteriorPointLagrangian)
    M = L.problem.M
    p, μ, λ, s = L.params.q.x
    m = length(μ)
    n = length(λ)
    N = M × ℝ^m × ℝ^n × ℝ^m
    stopping_criterion = StopAfterIteration(100)
    retraction_method = default_retraction_method(N)
    step_size = ArmijoLinesearch(N; retraction_method=retraction_method, initial_stepsize=1.0)

    return InteriorPointState(L, stopping_criterion, retraction_method, step_size)
end
    
function is_feasible(
    M::AbstractManifold,
   ob::ConstrainedManifoldObjective,
    p
)
    # evaluateobnstraint functions at p
    gp = get_inequality_constraints(M,ob, p)
    hp = get_equality_constraints(M,ob, p)

    # check feasibility
    return all(gp .<= 0) && all(hp .== 0)
end

function step_solver(state::InteriorPointState)

    # get Lagrangian
    L = state.L

    # get manifold
    M = L.problem.M

    # get current iterate
    q = L.params.q

    # build product manifold
    m, n = L.problem.dims
    N = M × ℝ^m × ℝ^n × ℝ^m

    # random initial search direction
    X = rand(N, vector_at = q) 

    # get search direction
    X = subsolver(Hess, X, .-grad(L))

    R = state.retraction_method
    α = state.step_size

    # Get next iterate
    q = retract(N, q, X, R)

    state.L.params.q = q
end

#obnjugate residudal method, 
function subsolver(A, X_0, b)

    k = 0
    r_0 = b - A(L, X_0)
    p_0 = r_0
    Hr_0 = A(L, r_0)
    Hp_0 = A(L, p_0)
    
    r = r_0
    p = p_0
    Hr = Hr_0
    Hp = Hp_0
    
    X = X_0
    
    while inner(N, q, r, r) > 1e-5

        α = inner(N, q, r, Hr) / inner(N, q, Hp, Hp)
        X += α*p
        r_next = r - α*Hp
        Hr_next = A(L, r_next)
        β = inner(N, q, r_next, Hr_next) / inner(N, q, r, Hr)
        p_next = r_next + β*p
        Hp_next = A(L, r_next) + β*Hp
        r = r_next
        p = p_next
        Hr = Hr_next
        Hp = Hp_next

        k += 1
    end
    return X
end

function interior_point_Newton(M::AbstractManifold,ob::ConstrainedManifoldObjective)
    p = rand(M)
    m = length(get_inequality_constraints(M,ob, p))
    n = length(get_equality_constraints(M,ob, p))
    problem = InteriorPointProblem(M,ob, (m,n))
    params = InteriorPointParams(problem)
    L = InteriorPointLagrangian(problem, params)

    # initial state
    state = InteriorPointState(L)

    for i in range(1, 10)
        step_solver(state)
    end

    return state.L.params.q[N,1]
end



