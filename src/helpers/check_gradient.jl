"""
    check_gradient(M, F, gradF, p=random_point(M), X=random_tangent(M,p))

Check numerivcally whether the gradient `gradF` of `F` is correct.

This implements the method described in Section 4.8 [^Boumal2022].

# Keyword arguments

* `N` (`101`) – number of points to check within the `log_range` default range ``[10^{-8},10^{0}]``
* `error` - (`false`) throw an error message if the gradient is wrong
* `io` – (`nothing`) provide an `IO` to print the check result to
* `limits` (`(10e-8,1)`) specify the limits in the `log_range`
* `plot`- (`false`) whether to plot the resulting check (if `Plots.jl` is loaded)
* `retraction_method` - (`default_retraction_method(M)`) retraction method to use for the check
* `slope_tolerance` – (`0.1`)
* `log_range` (`10e `)
[^Boumal2022}:
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
)
    gradient = gradF(M, p)
    Xn = X ./ norm(M, p, X) # normalize tangent direction
    is_vector(M, p, gradient, error)
    # function for the directional derivative
    df(M, p, Y) = inner(M, p, gradient, Y)
    #
    T = 10 .^ (log_range)
    # points p_i to evaluate our error function at
    points = map(t -> retract(M, p, X, t, retraction_method), T)
    # F(p_i)
    costs = [F(M, pi) for pi in points]
    # linearized
    linearized = map(t -> F(M, p) - t * df(M, p, Xn), T)
    lin_error = abs.(costs .- linearized)
    max_error = maximum(lin_error)
    plot && plot_slope(T, lin_error, 2.0, costs[1])
    if io !== nothing
        println(io, "The maximal error in the gradient check is $max_error.")
    end
    # estmate (global) slope

    # estimate (local) slope?
end
