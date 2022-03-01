
# Fits \hat{y} = a + b x.
# Returns a, b, and the R^2 of the fit.
function solve_simple_lm(x, y)
    # See https://en.wikipedia.org/wiki/Simple_linear_regression
    correlation = cor(x, y)
    beta = std(y) / std(x) * correlation
    alpha = mean(y) - beta * mean(x)
    r2 = size(x, 1) <= 2 ? 1 : correlation^2
    return (alpha, beta, r2)
end

function linearize_function(x, y; r2_cutoff=0.99)
    @assert size(x) == size(y)
    y = y[sortperm(x)]
    x = sort(x)
    seq_len = size(y, 1)
    n_elem = (seq_len - 1) * (seq_len) ÷ 2
    df = DataFrame(;
        a=zeros(n_elem),
        b=zeros(n_elem),
        r2=zeros(n_elem),
        left=zeros(n_elem),
        right=zeros(n_elem),
        length=zeros(n_elem),
        left_idx=zeros(Int32, n_elem),
        right_idx=zeros(Int32, n_elem),
    )
    counter = 1
    for i in 1:seq_len
        for j in (i + 1):seq_len
            a, b, r2 = solve_simple_lm(x[i:j], y[i:j])
            df[counter, :a] = a
            df[counter, :b] = b
            df[counter, :r2] = r2
            df[counter, :left] = x[i]
            df[counter, :right] = x[j]
            df[counter, :length] = x[j] - x[i]
            df[counter, :left_idx] = i
            df[counter, :right_idx] = j
            counter += 1
        end
    end

    df = df[df[:, :r2] .> r2_cutoff, :]
    df = sort(df, :length; rev=true)

    res = DataFrame(;
        a=zeros(seq_len),
        b=zeros(seq_len),
        r2=zeros(seq_len),
        left=zeros(seq_len),
        right=zeros(seq_len),
        length=zeros(seq_len),
        left_idx=zeros(Int32, seq_len),
        right_idx=zeros(Int32, seq_len),
    )
    counter = 1
    covered_set = collect(1:seq_len)
    while size(covered_set, 1) > 0
        line = df[1, :]
        res[counter, :] = line
        counter += 1
        remove_rows =
            (line.left .<= df[:, :left] .< line.right) .|
            (line.left .< df[:, :right] .<= line.right)
        remove_rows[1] = true
        df = df[.!remove_rows, :]
        setdiff!(covered_set, collect((line.left_idx):(line.right_idx)))
    end
    res = res[res[:, :left_idx] .!= 0, :]
    res = sort(res, :left)

    return res
end

function check_diff(
    M::AbstractManifold,
    cost_fnc,
    diff_fnc;
    p=random_point(M),
    X=random_tangent(M, p),
    log_space_range=range(-8, 0; length=101),
    plot=false,
)
    check_point(M, p)
    check_vector(M, p, X)
    f0 = cost_fnc(M, p)
    df0 = diff_fnc(M, p, X)

    log_space = 10 .^ log_space_range

    # Based on the Taylor approximation of development of the cost function around point x:
    # f(R_x(t * X)) = f(x) + t * <grad f(x), X)> + O(t^2)
    lhs = [cost_fnc(M, geodesic(M, p, X, t)) for t in log_space]
    rhs = [f0 + t * df0 for t in log_space]
    err = abs.(rhs - lhs)

    df = linearize_function(log10.(log_space), log10.(err))
    total_quadratic_length = sum(df[1.8 .< df[:, :b], :length])
    mostly_quadratic =
        total_quadratic_length >
        0.75 * (maximum(log_space_range) - minimum(log_space_range))
    if plot
        plt = Plots.scatter(
            log10.(log_space),
            log10.(err);
            legend=false,
            title=if mostly_quadratic
                "The differential seems correct"
            else
                "The differential seems incorrect"
            end,
            xlabel="log10 distance to p",
            ylabel="log10 approximation error of differential",
        )
        Plots.abline!(plt, 1, 0; color="red")
        Plots.abline!(plt, 2, 0; color="green")
        for i in 1:size(df, 1)
            xs = [df[i, :left], df[i, :right]]
            Plots.plot!(
                plt,
                xs,
                df[i, :a] .+ xs .* df[i, :b];
                color=df[i, :b] > 1.8 ? "green" : "red",
            )
        end
        return plt
    else
        if !mostly_quadratic
            return ErrorException("Less than 75% of the range has a slope of roughly two.")
        else
            return nothing
        end
    end
end
