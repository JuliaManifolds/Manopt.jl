
function plot_slope end

"""
    plot_slope(x, y; slope=2, line_base=0, a=0, b=2.0, i=1,j=length(x))

Plot the result from the verification functions [`check_gradient`](@ref), [`check_differential`](@ref), [`check_Hessian`](@ref)
on data `x,y` with two comparison lines

1) `line_base` + t`slope`  as the global slope the plot should have
2) `a` + `b*t` on the interval [`x[i]`, `x[j]`] for some (best fitting) comparison slope
"""
plot_slope(x, y)

"""
    prepare_check_result(log_range, errors, slope)

Given a range of values `log_range`, with computed `errors`,
verify whether this yields a slope of `slope` in log-scale

Note that if the errors are below the given tolerance and the method is exact,
no plot is be generated,

# Keyword arguments

* `exactness_tol`: (`1e3*eps(eltype(errors))`) is all errors are below this tolerance, the verification is considered to be exact
* `io`:            (`nothing`) provide an `IO` to print the result to
* `name`:          (`"differential"`) name to display in the plot title
* `plot`:          (`false`) whether to plot the result (if `Plots.jl` is loaded).
  The plot is in log-log-scale. This is returned and can then also be saved.
* `slope_tol`:     (`0.1`) tolerance for the slope (global) of the approximation
* `throw_error`:   (`false`) throw an error message if the gradient or Hessian is wrong
"""
function prepare_check_result(
    log_range,
    errors,
    slope;
    io::Union{IO,Nothing}=nothing,
    name="estimated slope",
    slope_tol=1e-1,
    plot=false,
    throw_error=false,
    window=nothing,
    exactness_tol=1e3 * eps(eltype(errors)),
)
    if max(errors...) < exactness_tol
        (io !== nothing) && print(
            io,
            "All errors are below the exactness tolerance $(exactness_tol). Your check can be considered exact, hence there is no use to check for a slope.\n",
        )
        return true
    end
    x = log_range[errors .> 0]
    T = exp10.(x)
    y = log10.(errors[errors .> 0])
    (a, b) = find_best_slope_window(x, y, length(x))[1:2]
    if isapprox(b, slope; atol=slope_tol)
        plot && return plot_slope(
            T,
            errors[errors .> 0];
            slope=slope,
            line_base=errors[1],
            a=a,
            b=b,
            i=1,
            j=length(y),
        )
        (io !== nothing) && print(
            io,
            "Your $name's slope is globally $(@sprintf("%.4f", b)), so within $slope ± $(slope_tol).\n",
        )
        return true
    end
    # otherwise
    # find best contiguous window of length w
    (ab, bb, ib, jb) = find_best_slope_window(x, y, window; slope_tol=slope_tol)
    msg = "The $(name) fits best on [$(T[ib]),$(T[jb])] with slope  $(@sprintf("%.4f", bb)), but globally your slope $(@sprintf("%.4f", b)) is outside of the tolerance $slope ± $(slope_tol).\n"
    (io !== nothing) && print(io, msg)
    plot && return plot_slope(
        T, errors[errors .> 0]; slope=slope, line_base=errors[1], a=ab, b=bb, i=ib, j=jb
    )
    throw_error && throw(ErrorException(msg))
    return false
end

@doc raw"""
    check_differential(M, F, dF, p=rand(M), X=rand(M; vector_at=p); kwargs...)

Check numerically whether the differential `dF(M,p,X)` of `F(M,p)` is correct.

This implements the method described in [Boumal:2023; Section 4.8](@cite).

Note that if the errors are below the given tolerance and the method is exact,
no plot is generated,

# Keyword arguments

* `exactness_tol`:     (`1e-12`) if all errors are below this tolerance, the differential is considered to be exact
* `io`:                (`nothing`) provide an `IO` to print the result to
* `limits`:            (`(1e-8,1)`) specify the limits in the `log_range`
* `log_range`:         (`range(limits[1], limits[2]; length=N)`) specify the range of points (in log scale) to sample the differential line
* `N`:                 (`101`) number of points to verify within the `log_range` default range ``[10^{-8},10^{0}]``
* `name`:              (`"differential"`) name to display in the plot
* `plot`:              (`false`) whether to plot the result (if `Plots.jl` is loaded).
  The plot is in log-log-scale. This is returned and can then also be saved.
* `retraction_method`: (`default_retraction_method(M, typeof(p))`) retraction method to use
* `slope_tol`:         (`0.1`) tolerance for the slope (global) of the approximation
* `throw_error`:       (`false`) throw an error message if the differential is wrong
* `window`:            (`nothing`) specify window sizes within the `log_range` that are used for the slope estimation.
  the default is, to use all window sizes `2:N`.
"""
function check_differential(
    M::AbstractManifold,
    F,
    dF,
    p=rand(M),
    X=rand(M; vector_at=p);
    exactness_tol=1e-12,
    io::Union{IO,Nothing}=nothing,
    limits=(-8.0, 0.0),
    N=101,
    name="differential",
    log_range=range(limits[1], limits[2]; length=N),
    plot=false,
    retraction_method=default_retraction_method(M, typeof(p)),
    slope_tol=0.1,
    throw_error=false,
    window=nothing,
)
    Xn = X ./ norm(M, p, X) # normalize tangent direction
    # function for the directional derivative
    #
    T = exp10.(log_range)
    # points `p_i` to evaluate the error function at
    points = map(t -> retract(M, p, Xn, t, retraction_method), T)
    costs = [F(M, pi) for pi in points]
    # linearized
    linearized = map(t -> F(M, p) + t * dF(M, p, Xn), T)
    return prepare_check_result(
        log_range,
        abs.(costs .- linearized),
        2.0;
        exactness_tol=exactness_tol,
        io=io,
        name=name,
        plot=plot,
        slope_tol=slope_tol,
        throw_error=throw_error,
        window=window,
    )
end

@doc raw"""
    check_gradient(M, F, gradF, p=rand(M), X=rand(M; vector_at=p); kwargs...)

Verify numerically whether the gradient `gradF(M,p)` of `F(M,p)` is correct, that is whether


```math
f(\operatorname{retr}_p(tX)) = f(p) + t⟨\operatorname{grad} f(p), X⟩ + \mathcal O(t^2)
```

or in other words, that the error between the function ``f`` and its first order Taylor
behaves in error ``\mathcal O(t^2)``, which indicates that the gradient is correct,
cf. also [Boumal:2023; Section 4.8](@cite).

Note that if the errors are below the given tolerance and the method is exact,
no plot is generated.

# Keyword arguments

* `check_vector`:      (`true`) verify that ``\operatorname{grad} f(p) ∈ T_p\mathcal M`` using `is_vector`.
* `exactness_tol`:     (`1e-12`) if all errors are below this tolerance, the gradient is considered to be exact
* `io`:                (`nothing`) provide an `IO` to print the result to
* `gradient`:          (`grad_f(M, p)`) instead of the gradient function you can also provide the gradient at `p` directly
* `limits`:            (`(1e-8,1)`) specify the limits in the `log_range`
* `log_range`:         (`range(limits[1], limits[2]; length=N)`) - specify the range of points (in log scale) to sample the gradient line
* `N`:                 (`101`) number of points to verify within the `log_range` default range ``[10^{-8},10^{0}]``
* `plot`:              (`false`) whether to plot the result (if `Plots.jl` is loaded).
  The plot is in log-log-scale. This is returned and can then also be saved.
* `retraction_method`: (`default_retraction_method(M, typeof(p))`) retraction method to use
* `slope_tol`:         (`0.1`) tolerance for the slope (global) of the approximation
* `atol`, `rtol`:      (same defaults as `isapprox`) tolerances that are passed down to `is_vector` if `check_vector` is set to `true`
* `throw_error`:       (`false`) throw an error message if the gradient is wrong
* `window`:            (`nothing`) specify window sizes within the `log_range` that are used for the slope estimation.
  the default is, to use all window sizes `2:N`.

The remaining keyword arguments are also passed down to the `check_vector` call, such that tolerances can
easily be set.

"""
function check_gradient(
    M::AbstractManifold,
    f,
    grad_f,
    p=rand(M),
    X=rand(M; vector_at=p);
    gradient=grad_f(M, p),
    check_vector=false,
    throw_error=false,
    atol::Real=0,
    rtol::Real=atol > 0 ? 0 : sqrt(eps(eltype(p))),
    kwargs...,
)
    check_vector &&
        (!is_vector(M, p, gradient, throw_error; atol=atol, rtol=rtol) && return false)
    # function for the directional derivative - real so it also works on complex manifolds
    df(M, p, Y) = real(inner(M, p, gradient, Y))
    return check_differential(
        M, f, df, p, X; name="gradient", throw_error=throw_error, kwargs...
    )
end

@doc raw"""
    check_Hessian(M, f, grad_f, Hess_f, p=rand(M), X=rand(M; vector_at=p), Y=rand(M, vector_at=p); kwargs...)

Verify numerically whether the Hessian ``\operatorname{Hess} f(M,p, X)`` of `f(M,p)` is correct.

For this either a second-order retraction or a critical point ``p`` of `f` is required.
The approximation is then

```math
f(\operatorname{retr}_p(tX)) = f(p) + t⟨\operatorname{grad} f(p), X⟩ + \frac{t^2}{2}⟨\operatorname{Hess}f(p)[X], X⟩ + \mathcal O(t^3)
```

or in other words, that the error between the function ``f`` and its second order Taylor
behaves in error ``\mathcal O(t^3)``, which indicates that the Hessian is correct,
cf. also [Boumal:2023; Section 6.8](@cite).

Note that if the errors are below the given tolerance and the method is exact,
no plot is generated.

# Keyword arguments

* `check_grad`:       (`true`) verify that ``\operatorname{grad} f(p) ∈ T_p\mathcal M``.
* `check_linearity`:  (`true`) verify that the Hessian is linear, see [`is_Hessian_linear`](@ref) using `a`, `b`, `X`, and `Y`
* `check_symmetry`:   (`true`) verify that the Hessian is symmetric, see [`is_Hessian_symmetric`](@ref)
* `check_vector`:     (`false`) verify that ``\operatorname{Hess} f(p)[X] ∈ T_p\mathcal M`` using `is_vector`.
* `mode`:             (`:Default`) specify the mode for the verification; the default assumption is,
  that the retraction provided is of second order. Otherwise one can also verify the Hessian
  if the point `p` is a critical point.
  THen set the mode to `:CritalPoint` to use [`gradient_descent`](@ref) to find a critical point.
  Note: this requires (and evaluates) new tangent vectors `X` and `Y`

* `atol`, `rtol`:      (same defaults as `isapprox`) tolerances that are passed down to all checks
* `a`, `b`            two real values to verify linearity of the Hessian (if `check_linearity=true`)
* `N`:                 (`101`) number of points to verify within the `log_range` default range ``[10^{-8},10^{0}]``
* `exactness_tol`:     (`1e-12`) if all errors are below this tolerance, the verification is considered to be exact
* `io`:                (`nothing`) provide an `IO` to print the result to
* `gradient`:          (`grad_f(M, p)`) instead of the gradient function you can also provide the gradient at `p` directly
* `Hessian`:           (`Hess_f(M, p, X)`) instead of the Hessian function you can provide the result of ``\operatorname{Hess} f(p)[X]`` directly.
  Note that evaluations of the Hessian might still be necessary for checking linearity and symmetry and/or when using `:CriticalPoint` mode.
* `limits`:            (`(1e-8,1)`) specify the limits in the `log_range`
* `log_range`:         (`range(limits[1], limits[2]; length=N)`) specify the range of points (in log scale) to sample the Hessian line
* `N`:                 (`101`) number of points to use within the `log_range` default range ``[10^{-8},10^{0}]``
* `plot`:              (`false`) whether to plot the resulting verification (requires `Plots.jl` to be loaded). The plot is in log-log-scale. This is returned and can then also be saved.
* `retraction_method`: (`default_retraction_method(M, typeof(p))`) retraction method to use for
* `slope_tol`:         (`0.1`) tolerance for the slope (global) of the approximation
* `throw_error`:       (`false`) throw an error message if the Hessian is wrong
* `window`:            (`nothing`) specify window sizes within the `log_range` that are used for the slope estimation.
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
    p=rand(M),
    X=rand(M; vector_at=p),
    Y=rand(M; vector_at=p);
    a=randn(),
    atol::Real=0,
    b=randn(),
    check_grad=true,
    check_vector=false,
    check_symmetry=true,
    check_linearity=true,
    exactness_tol=1e-12,
    io::Union{IO,Nothing}=nothing,
    gradient=grad_f(M, p),
    Hessian=Hess_f(M, p, X),
    limits=(-8.0, 0.0),
    mode::Symbol=:Default,
    N=101,
    log_range=range(limits[1], limits[2]; length=N),
    plot=false,
    retraction_method=default_retraction_method(M, typeof(p)),
    rtol::Real=atol > 0 ? 0 : sqrt(eps(eltype(p))),
    slope_tol=0.1,
    throw_error=false,
    window=nothing,
    kwargs...,
)
    if check_grad
        if !check_gradient(
            M,
            f,
            grad_f,
            p,
            X;
            gradient=gradient,
            throw_error=throw_error,
            io=io,
            check_vector=check_vector,
            atol=atol,
            rtol=rtol,
            retraction_method=retraction_method,
            kwargs...,
        )
            return false
        end
    end
    check_vector &&
        (!is_vector(M, p, Hessian, throw_error; atol=atol, rtol=rtol) && return false)
    if check_linearity
        if !is_Hessian_linear(
            M, Hess_f, p, X, Y, a, b; throw_error=throw_error, io=io, atol=atol, rtol=rtol
        )
            return false
        end
    end
    if check_symmetry
        if !is_Hessian_symmetric(
            M, Hess_f, p, X, Y; throw_error=throw_error, io=io, atol=atol, rtol=rtol
        )
            return false
        end
    end
    if mode === :CriticalPoint # find a critical point and update gradient, Hessian and tangent vector
        p = gradient_descent(M, f, grad_f, p)
        gradient = grad_f(M, p)
        X = rand(M; vector_at=p)
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
    points = map(t -> retract(M, p, X_n, t, retraction_method), T)
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
        exactness_tol=exactness_tol,
        io=io,
        name="Hessian",
        plot=plot,
        slope_tol=slope_tol,
        throw_error=throw_error,
        window=window,
    )
end

@doc raw"""
    is_Hessian_linear(M, Hess_f, p,
        X=rand(M; vector_at=p), Y=rand(M; vector_at=p), a=randn(), b=randn();
        throw_error=false, io=nothing, kwargs...
    )

Verify whether the Hessian function `Hess_f` fulfills linearity,

```math
\operatorname{Hess} f(p)[aX + bY] = b\operatorname{Hess} f(p)[X]
 + b\operatorname{Hess} f(p)[Y]
```

which is checked using `isapprox` and the keyword arguments are passed to this function.

# Optional arguments

* `throw_error`: (`false`) throw an error message if the Hessian is wrong

"""
function is_Hessian_linear(
    M,
    Hess_f,
    p,
    X=rand(M; vector_at=p),
    Y=rand(M; vector_at=p),
    a=randn(),
    b=randn();
    throw_error=false,
    io=nothing,
    kwargs...,
)
    Z1 = Hess_f(M, p, a * X + b * Y)
    Z2 = a * Hess_f(M, p, X) + b * Hess_f(M, p, Y)
    isapprox(M, p, Z1, Z2; kwargs...) && return true
    n = norm(M, p, Z1 - Z2)
    m = "Hess f seems to not be linear since Hess_f(p)[aX+bY] differs from aHess f(p)[X] + b*Hess f(p)[Y] by $(n).\n"
    (io !== nothing) && print(io, m)
    throw_error && throw(ErrorException(m))
    return false
end

@doc raw"""
    is_Hessian_symmetric(M, Hess_f, p=rand(M), X=rand(M; vector_at=p), Y=rand(M; vector_at=p);
    throw_error=false, io=nothing, atol::Real=0, rtol::Real=atol>0 ? 0 : √eps
)

Verify whether the Hessian function `Hess_f` fulfills symmetry, which means that

```math
⟨\operatorname{Hess} f(p)[X], Y⟩ = ⟨X, \operatorname{Hess} f(p)[Y]⟩
```

which is checked using `isapprox` and the `kwargs...` are passed to this function.

# Optional arguments

* `atol`, `rtol`   with the same defaults as the usual `isapprox`
* `throw_error`:    (`false`) throw an error message if the Hessian is wrong
"""
function is_Hessian_symmetric(
    M,
    Hess_f,
    p=rand(M),
    X=rand(M; vector_at=p),
    Y=rand(M; vector_at=p);
    throw_error=false,
    io=nothing,
    atol::Real=0,
    rtol::Real=atol > 0 ? 0 : sqrt(eps(number_eltype(p))),
    kwargs...,
)
    a = inner(M, p, Hess_f(M, p, X), Y)
    b = inner(M, p, X, Hess_f(M, p, Y))
    isapprox(a, b; atol=atol, rtol=rtol) && (return true)
    m = "Hess f seems to not be symmetric: ⟨Hess f(p)[X], Y⟩ = $a != $b = ⟨Hess f(p)[Y], X⟩"
    (io !== nothing) && print(io, m)
    throw_error && throw(ErrorException(m))
    return false
end

"""
    (a,b,i,j) = find_best_slope_window(X,Y,window=nothing; slope=2.0, slope_tol=0.1)

Check data X,Y for the largest contiguous interval (window) with a regression line fitting “best”.
Among all intervals with a slope within `slope_tol` to `slope` the longest one is taken.
If no such interval exists, the one with the slope closest to `slope` is taken.

If the window is set to `nothing` (default), all window sizes `2,...,length(X)` are checked.
You can also specify a window size or an array of window sizes.

For each window size, all its translates in the data is checked.
For all these (shifted) windows the regression line is computed (with `a,b` in `a + t*b`)
and the best line is computed.

From the best line the following data is returned

* `a`, `b` specifying the regression line `a + t*b`
* `i`, `j` determining the window, i.e the regression line stems from data `X[i], ..., X[j]`
"""
function find_best_slope_window(X, Y, window=nothing; slope=2.0, slope_tol=0.1)
    n = length(X)
    if window !== nothing && (any(window .> n))
        error(
            "One of the window sizes ($(window)) is larger than the length of the signal (n=$n).",
        )
    end
    a_best = 0
    b_best = -Inf
    i_best = 0
    j_best = 0
    r_best = 0 # longest interval
    for w in (window === nothing ? (2:n) : [window...])
        for j in 1:(n - w + 1)
            x = X[j:(j + w - 1)]
            y = Y[j:(j + w - 1)]
            # fit a line a + bx
            c = cor(x, y)
            b = std(y) / std(x) * c
            a = mean(y) - b * mean(x)
            # look for the largest interval where b is within slope tolerance
            r = (maximum(x) - minimum(x))
            if (r > r_best) && abs(b - slope) < slope_tol #longer interval found.
                r_best = r
                a_best = a
                b_best = b
                i_best = j
                j_best = j + w - 1 #last index (see x and y from before)
            end
            # not best interval - maybe it is still the (first) best slope?
            if r_best == 0 && abs(b - slope) < abs(b_best - slope)
                # but do not update `r` since this indicates only a best r
                a_best = a
                b_best = b
                i_best = j
                j_best = j + w - 1 #last index (see x and y from before)
            end
        end
    end
    return (a_best, b_best, i_best, j_best)
end
