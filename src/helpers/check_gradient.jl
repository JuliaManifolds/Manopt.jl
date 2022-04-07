@doc raw"""
    check_gradient(M, F, gradF, p=random_point(M), X=random_tangent(M,p))

Check numerivcally whether the gradient `gradF` of `F` is correct.

This implements the method described in Section 4.8 [^Boumal2022].

# Keyword arguments

* `N` (`101`) – number of points to check within the `log_range` default range ``[10^{-8},10^{0}]``
* `error` - (`false`) throw an error message if the gradient is wrong
* `io` – (`nothing`) provide an `IO` to print the check result to
* `limits` (`(10e-8,1)`) specify the limits in the `log_range`
* `log_range` (`range(limits[1], limits[2]; length=N)`) - specify the range of points (in log scale) to sample the gradient line
* `plot`- (`false`) whether to plot the resulting check (if `Plots.jl` is loaded)
* `retraction_method` - (`default_retraction_method(M)`) retraction method to use for the check
* `slope_tol` – (`0.1`) tolerance for the slope (global) of the approximation

[^Boumal2022]:
    > Boumal, N.: _An Introduction to Optimization on Smooth Manifolds_, book in preparation,
    > 2022. url: [http://www.nicolasboumal.net/book](http://www.nicolasboumal.net/book).
"""
function check_gradient(
    M::AbstractManifold,
    F,
    gradF,
    p=random_point(M),
    X=random_tangent(M, p);
    plot=false,
    error=false,
    io::Union{IO,Nothing}=nothing,
    limits=(-8.0, 1.0),
    N=101,
    log_range=range(limits[1], limits[2]; length=N),
    retraction_method=default_retraction_method(M),
    slope_tol=0.1,
    window=nothing,
)
    gradient = gradF(M, p)
    Xn = X ./ norm(M, p, X) # normalize tangent direction
    is_vector(M, p, gradient, error)
    # function for the directional derivative
    df(M, p, Y) = inner(M, p, gradient, Y)
    #
    T = exp10.(log_range)
    n = length(T)
    # points p_i to evaluate our error function at
    points = map(t -> retract(M, p, X, t, retraction_method), T)
    # F(p_i)
    costs = [F(M, pi) for pi in points]
    # linearized
    linearized = map(t -> F(M, p) - t * df(M, p, Xn), T)
    lin_error = abs.(costs .- linearized)
    max_error = maximum(lin_error)
    if io !== nothing
        print(io, "The maximal error in the gradient check is $(max_error).\n")
    end
    # global fit a + bx
    x = log_range
    y = log10.(lin_error)
    b = std(y) / std(x) * cor(x, y)
    a = mean(y) - b * mean(x)
    if isapprox(b, 2.0; atol=slope_tol)
        plot && plot_slope(T, lin_error; line_base=costs[1], a=a, b=b, i=1, j=n)
        return true
    end
    # otherwise
    # find best contiguous window of length w
    a_best = 0
    b_best = 0
    min_err = Inf
    i_best = 0
    j_best = 0
    for w in (window === nothing ? (2:(n - 1)) : [window...])
        for j in 1:(n - w)
            x = log_range[j:(j + w)]
            y = log10.(lin_error[j:(j + w)])
            # fit a line a + bx
            c = cor(x, y)
            b = std(y) / std(x) * c
            a = mean(y) - b * mean(x)
            # look for the best error relative to log scale interval
            r = (maximum(x) - minimum(x))
            if c^2 / r < min_err
                a_best = a
                b_best = b
                min_err = c^2 / r
                i_best = j
                j_best = j + w
            end
        end
    end
    plot &&
        plot_slope(T, lin_error; line_base=costs[1], a=a_best, b=b_best, i=i_best, j=j_best)
    if io !== nothing
        print(
            io,
            "You gradient fits best on [$(T[i_best]),$(T[j_best])] with slope $(b_best), but globally your slope $b is outside of the tolerance 2 ± $(slope_tol).\n",
        )
    end
    return false
end
