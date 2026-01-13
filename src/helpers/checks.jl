@doc """
    check_differential(M, F, dF, p=rand(M), X=rand(M; vector_at=p); kwargs...)

Check numerically whether the differential `dF(M,p,X)` of `F(M,p)` is correct.

This implements the method described in [Boumal:2023; Section 4.8](@cite).

Note that if the errors are below the given tolerance and the method is exact,
no plot is generated,

# Keyword arguments

* `exactness_tol=1e-12`: if all errors are below this tolerance,
  the differential is considered to be exact
* `io=nothing`: provide an `IO` to print the result to
* `limits=(-8.0, 0.0)`: specify the limits in the `log_range`
* `log_range=range(limits[1], limits[2]; length=N)`: specify the range of points
  (in log scale) to sample the differential line
* `N=101`: number of points to verify within the `log_range` default range ``[10^{-8},10^{0}]``
* `name="differential"`: name to display in the plot
* `plot=false`: whether to plot the result (if `Plots.jl` is loaded).
  The plot is in log-log-scale. This is returned and can then also be saved.
$(_kwargs(:retraction_method))
* `slope_tol=0.1`: tolerance for the slope (global) of the approximation
* `throw_error=false`: throw an error message if the differential is wrong
* `window=nothing`: specify window sizes within the `log_range` that are used for
  the slope estimation. The default is, to use all window sizes `2:N`.
"""
function check_differential(
        M::AbstractManifold,
        F,
        dF,
        p = rand(M),
        X = rand(M; vector_at = p);
        exactness_tol = 1.0e-12,
        io::Union{IO, Nothing} = nothing,
        limits = (-8.0, 0.0),
        N = 101,
        name = "differential",
        log_range = range(limits[1], limits[2]; length = N),
        plot = false,
        retraction_method = default_retraction_method(M, typeof(p)),
        slope_tol = 0.1,
        error::Symbol = :none,
        window = nothing,
    )
    Xn = X ./ norm(M, p, X) # normalize tangent direction
    # function for the directional derivative
    #
    T = exp10.(log_range)
    # points `p_i` to evaluate the error function at
    points = map(t -> ManifoldsBase.retract_fused(M, p, Xn, t, retraction_method), T)
    costs = [F(M, pi) for pi in points]
    # linearized
    linearized = map(t -> F(M, p) + t * dF(M, p, Xn), T)
    return prepare_check_result(
        log_range,
        abs.(costs .- linearized),
        2.0;
        exactness_tol = exactness_tol,
        io = io,
        name = name,
        plot = plot,
        slope_tol = slope_tol,
        error = error,
        window = window,
    )
end

_doc_check_gradient_formula = """
```math
f($(_tex(:retr))_p(tX)) = f(p) + t⟨$(_tex(:grad)) f(p), X⟩ + $(_tex(:Cal, "O"))(t^2)
```
"""
@doc """
    check_gradient(M, f, grad_f, p=rand(M), X=rand(M; vector_at=p); kwargs...)

Verify numerically whether the gradient `grad_f(M,p)` of `f(M,p)` is correct, that is whether

$_doc_check_gradient_formula

or in other words, that the error between the function ``f`` and its first order Taylor
behaves in error ``$(_tex(:Cal, "O"))(t^2)``, which indicates that the gradient is correct,
cf. also [Boumal:2023; Section 4.8](@cite).

Note that if the errors are below the given tolerance and the method is exact,
no plot is generated.

# Keyword arguments

* `check_vector=true`:
  verify that ``$(_tex(:grad))f(p) ∈ $(_math(:TangentSpace))`` using `is_vector`.
* `exactness_tol=1e-12`:
  if all errors are below this tolerance, the gradient is considered to be exact
* `io=nothing`:
  provide an `IO` to print the result to
* `gradient=grad_f(M, p)`:
  instead of the gradient function you can also provide the gradient at `p` directly
* `limits=(-8.0, 0.0)`: specify the limits in the `log_range`
* `log_range=range(limits[1], limits[2]; length=N)`:
  - specify the range of points (in log scale) to sample the gradient line
* `N=101`:
  number of points to verify within the `log_range` default range ``[10^{-8},10^{0}]``
* `plot=false`:
  whether to plot the result (if `Plots.jl` is loaded).
  The plot is in log-log-scale. This is returned and can then also be saved.
$(_kwargs(:retraction_method))
* `slope_tol=0.1`:
  tolerance for the slope (global) of the approximation
* `atol`, `rtol`:
  (same defaults as `isapprox`) tolerances that are passed down to `is_vector` if `check_vector` is set to `true`
* `error=:none`:
  how to handle errors, possible values: `:error`, `:info`, `:warn`
* `window=nothing`:
  specify window sizes within the `log_range` that are used for the slope estimation.
  the default is, to use all window sizes `2:N`.

The remaining keyword arguments are also passed down to the `check_vector` call, such that tolerances can
easily be set.

"""
function check_gradient(
        M::AbstractManifold,
        f,
        grad_f,
        p = rand(M),
        X = rand(M; vector_at = p);
        gradient = grad_f(M, p),
        check_vector::Bool = false,
        error::Symbol = :none,
        atol::Real = 0,
        rtol::Real = atol > 0 ? 0 : sqrt(eps(real(eltype(p)))),
        kwargs...,
    )
    check_vector &&
        (!is_vector(M, p, gradient, error === :error; atol = atol, rtol = rtol) && return false)
    # function for the directional derivative - real so it also works on complex manifolds
    df(M, p, Y) = real(inner(M, p, gradient, Y))
    return check_differential(M, f, df, p, X; name = "gradient", error = error, kwargs...)
end

_doc_check_Hess_formula = """
```math
f($(_tex(:retr))_p(tX)) = f(p) + t⟨$(_tex(:grad)) f(p), X⟩ + $(_tex(:frac, "t^2", "2"))⟨$(_tex(:Hess))f(p)[X], X⟩ + $(_tex(:Cal, "O"))(t^3)
```
"""

@doc """
    check_Hessian(M, f, grad_f, Hess_f, p=rand(M), X=rand(M; vector_at=p), Y=rand(M, vector_at=p); kwargs...)

Verify numerically whether the Hessian `Hess_f(M,p, X)` of `f(M,p)` is correct.

For this either a second-order retraction or a critical point ``p`` of `f` is required.
The approximation is then

$_doc_check_Hess_formula

or in other words, that the error between the function ``f`` and its second order Taylor
behaves in error ``$(_tex(:Cal, "O"))(t^3)``, which indicates that the Hessian is correct,
cf. also [Boumal:2023; Section 6.8](@cite).

Note that if the errors are below the given tolerance and the method is exact,
no plot is generated.

# Keyword arguments

* `check_grad=true`:
  verify that ``$(_tex(:grad))f(p) ∈ $(_math(:TangentSpace)))``.
* `check_linearity=true`:
  verify that the Hessian is linear, see [`is_Hessian_linear`](@ref) using `a`, `b`, `X`, and `Y`
* `check_symmetry=true`:
  verify that the Hessian is symmetric, see [`is_Hessian_symmetric`](@ref)
* `check_vector=false`:
  verify that `$(_tex(:Hess)) f(p)[X] ∈ $(_math(:TangentSpace)))`` using `is_vector`.
* `mode=:Default`:
  specify the mode for the verification; the default assumption is,
  that the retraction provided is of second order. Otherwise one can also verify the Hessian
  if the point `p` is a critical point.
  THen set the mode to `:CritalPoint` to use [`gradient_descent`](@ref) to find a critical point.
  Note: this requires (and evaluates) new tangent vectors `X` and `Y`
* `atol`, `rtol`:      (same defaults as `isapprox`) tolerances that are passed down to all checks
* `a`, `b`            two real values to verify linearity of the Hessian (if `check_linearity=true`)
* `N=101`:
  number of points to verify within the `log_range` default range ``[10^{-8},10^{0}]``
* `exactness_tol=1e-12`:
  if all errors are below this tolerance, the verification is considered to be exact
* `io=nothing`:
  provide an `IO` to print the result to
* `gradient=grad_f(M, p)`:
  instead of the gradient function you can also provide the gradient at `p` directly
* `Hessian=Hess_f(M, p, X)`:
  instead of the Hessian function you can provide the result of ``$(_tex(:Hess)) f(p)[X]`` directly.
  Note that evaluations of the Hessian might still be necessary for checking linearity and symmetry and/or when using `:CriticalPoint` mode.
* `limits=(-8.0, 0.0)`: specify the limits in the `log_range`
* `log_range=range(limits[1], limits[2]; length=N)`:
  specify the range of points (in log scale) to sample the Hessian line
* `N=101`:
  number of points to use within the `log_range` default range ``[10^{-8},10^{0}]``
* `plot=false`:
  whether to plot the resulting verification (requires `Plots.jl` to be loaded). The plot is in log-log-scale. This is returned and can then also be saved.
$(_kwargs(:retraction_method))
* `slope_tol=0.1`:
  tolerance for the slope (global) of the approximation
* `error=:none`:
  how to handle errors, possible values: `:error`, `:info`, `:warn`
* `window=nothing`:
  specify window sizes within the `log_range` that are used for the slope estimation.
  the default is, to use all window sizes `2:N`.

The `kwargs...` are also passed down to the `check_vector` and the `check_gradient` call, such that tolerances can
easily be set.

While `check_vector` is also passed to the inner call to `check_gradient` as well as the `retraction_method`,
this inner `check_gradient` is meant to be just for inner verification, so it does not throw an error nor produce a plot itself.
"""
function check_Hessian(
        M::AbstractManifold,
        f,
        grad_f,
        Hess_f,
        p = rand(M),
        X = rand(M; vector_at = p),
        Y = rand(M; vector_at = p);
        a = randn(),
        atol::Real = 0,
        b = randn(),
        check_grad = true,
        check_vector = false,
        check_symmetry = true,
        check_linearity = true,
        exactness_tol = 1.0e-12,
        io::Union{IO, Nothing} = nothing,
        gradient = grad_f(M, p),
        Hessian = Hess_f(M, p, X),
        limits = (-8.0, 0.0),
        mode::Symbol = :Default,
        N = 101,
        log_range = range(limits[1], limits[2]; length = N),
        plot = false,
        retraction_method = default_retraction_method(M, typeof(p)),
        rtol::Real = atol > 0 ? 0 : sqrt(eps(real(eltype(p)))),
        slope_tol = 0.1,
        error = :none,
        window = nothing,
        kwargs...,
    )
    if check_grad
        if !check_gradient(
                M,
                f,
                grad_f,
                p,
                X;
                gradient = gradient,
                error = error,
                io = io,
                check_vector = check_vector,
                atol = atol,
                rtol = rtol,
                retraction_method = retraction_method,
                kwargs...,
            )
            return false
        end
    end
    check_vector &&
        (!is_vector(M, p, Hessian, error === :error; atol = atol, rtol = rtol) && return false)
    if check_linearity
        if !is_Hessian_linear(
                M, Hess_f, p, X, Y, a, b; error = error, io = io, atol = atol, rtol = rtol
            )
            return false
        end
    end
    if check_symmetry
        if !is_Hessian_symmetric(
                M, Hess_f, p, X, Y; error = error, io = io, atol = atol, rtol = rtol
            )
            return false
        end
    end
    if mode === :CriticalPoint # find a critical point and update gradient, Hessian and tangent vector
        p = gradient_descent(M, f, grad_f, p)
        gradient = grad_f(M, p)
        X = rand(M; vector_at = p)
        Hessian = Hess_f(M, p, X)
    end
    #
    # slope verification
    X_n = X ./ norm(M, p, X) # normalize tangent direction
    Hessian_n = Hessian ./ norm(M, p, X)
    # function for the directional derivative
    #
    T = exp10.(log_range)
    # points `p_i` to evaluate error function at
    points = map(t -> ManifoldsBase.retract_fused(M, p, X_n, t, retraction_method), T)
    # corresponding costs
    costs = [f(M, pi) for pi in points]
    # linearized
    linearized = map(
        t ->
        f(M, p) +
            t * real(inner(M, p, gradient, X_n)) +
            t^2 / 2 * real(inner(M, p, Hessian_n, X_n)),
        T,
    )
    return prepare_check_result(
        log_range,
        abs.(costs .- linearized),
        3.0;
        exactness_tol = exactness_tol,
        io = io,
        name = "Hessian",
        plot = plot,
        slope_tol = slope_tol,
        error = error,
        window = window,
    )
end

@doc """
    is_Hessian_linear(M, Hess_f, p,
        X=rand(M; vector_at=p), Y=rand(M; vector_at=p), a=randn(), b=randn();
        error=:none, io=nothing, kwargs...
    )

Verify whether the Hessian function `Hess_f` fulfills linearity,

```math
$(_tex(:Hess)) f(p)[aX + bY] = b$(_tex(:Hess)) f(p)[X]
 + b$(_tex(:Hess)) f(p)[Y]
```

which is checked using `isapprox` and the keyword arguments are passed to this function.

# Optional arguments

* `error=:none`:
  how to handle errors, possible values: `:error`, `:info`, `:warn`

"""
function is_Hessian_linear(
        M,
        Hess_f,
        p,
        X = rand(M; vector_at = p),
        Y = rand(M; vector_at = p),
        a = randn(),
        b = randn();
        error = :none,
        io = nothing,
        kwargs...,
    )
    Z1 = Hess_f(M, p, a * X + b * Y)
    Z2 = a * Hess_f(M, p, X) + b * Hess_f(M, p, Y)
    isapprox(M, p, Z1, Z2; kwargs...) && return true
    n = norm(M, p, Z1 - Z2)
    m = "Hess f seems to not be linear since Hess_f(p)[aX+bY] differs from aHess f(p)[X] + b*Hess f(p)[Y] by $(n).\n"
    (io !== nothing) && print(io, m)
    (error === :info) && @info m
    (error === :warn) && @warn m
    (error === :error) && throw(ErrorException(m))
    return false
end

@doc """
    is_Hessian_symmetric(M, Hess_f, p=rand(M), X=rand(M; vector_at=p), Y=rand(M; vector_at=p);
    error=:none, io=nothing, atol::Real=0, rtol::Real=atol>0 ? 0 : √eps
)

Verify whether the Hessian function `Hess_f` fulfills symmetry, which means that

```math
⟨$(_tex(:Hess)) f(p)[X], Y⟩ = ⟨X, $(_tex(:Hess)) f(p)[Y]⟩
```

which is checked using `isapprox` and the `kwargs...` are passed to this function.

# Optional arguments

* `atol`, `rtol`   with the same defaults as the usual `isapprox`
* `error=:none`:
  how to handle errors, possible values: `:error`, `:info`, `:warn`
"""
function is_Hessian_symmetric(
        M,
        Hess_f,
        p = rand(M),
        X = rand(M; vector_at = p),
        Y = rand(M; vector_at = p);
        error = :none,
        io = nothing,
        atol::Real = 0,
        rtol::Real = atol > 0 ? 0 : sqrt(eps(real(eltype(p)))),
        kwargs...,
    )
    a = inner(M, p, Hess_f(M, p, X), Y)
    b = inner(M, p, X, Hess_f(M, p, Y))
    isapprox(a, b; atol = atol, rtol = rtol) && (return true)
    m = "Hess f seems to not be symmetric: ⟨Hess f(p)[X], Y⟩ = $a != $b = ⟨Hess f(p)[Y], X⟩"
    (io !== nothing) && print(io, m)
    (error === :info) && @info m
    (error === :warn) && @warn m
    (error === :error) && throw(ErrorException(m))
    return false
end
