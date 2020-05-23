@doc raw"""
    quasi_Newton(M, F, ∇F, x, H, )
"""
function quasi_Newton(
    M::MT,
    F::Function,
    ∇F::Function,
    x,
    H::Union{Function,Missing};
    )


    if return_options
        return resultO
    else
        return get_solver_result(resultO)
    end
end

function initialize_solver!(p::P,o::O) where {P <: GradientProblem, O <: quasi_Newton_Options}
end

function step_solver!(p::P,o::O,iter) where {P <: GradientProblem, O <: quasi_Newton_Options}
    # Compute BFGS direction
    η = get_quasi_Newton_Direction()

    # Execute line-search
    α = line_search()

    # Compute Step
    o.x = o.Retraction(p.M, o.x)



    # Query cost and gradient at the candidate new point.
    [xNextCost, xNextGrad] = getCostGrad(problem, xNext, storedb, newkey);

    # Compute sk and yk
    sk = M.transp(xCur, xNext, step);
    yk = M.lincomb(xNext, 1, xNextGrad, ...
            -1, M.transp(xCur, xNext, xCurGradient));

    Computation of the BFGS step is invariant under scaling of sk and
    # yk by a common factor. For numerical reasons, we scale sk and yk
    # so that sk is a unit norm vector.
    norm_sk = M.norm(xNext, sk);
    sk = M.lincomb(xNext, 1/norm_sk, sk);
    yk = M.lincomb(xNext, 1/norm_sk, yk);

    inner_sk_yk = M.inner(xNext, sk, yk);
    inner_sk_sk = M.norm(xNext, sk)^2;    # ensures nonnegativity


    # If the cautious step is accepted (which is the intended
    # behavior), we record sk, yk and rhok and need to do some
    # housekeeping. If the cautious step is rejected, these are not
    # recorded. In all cases, xNext is the next iterate: the notion of
    # accept/reject here is limited to whether or not we keep track of
    # sk, yk, rhok to update the BFGS operator.
    cap = options.strict_inc_func(xCurGradNorm);
    if inner_sk_sk ~= 0 && (inner_sk_yk / inner_sk_sk) >= cap

    accepted = true;

    rhok = 1/inner_sk_yk;

    scaleFactor = inner_sk_yk / M.norm(xNext, yk)^2;

    # Time to store the vectors sk, yk and the scalar rhok.
    # Remember: we need to transport all vectors to the most
    # current tangent space.

    # If we are out of memory
    if k >= options.memory

    # sk and yk are saved from 1 to the end with the most
    # current recorded to the rightmost hand side of the cells
    # that are occupied. When memory is full, do a shift so
    # that the rightmost is earliest and replace it with the
    # most recent sk, yk.
    for  i = 2 : options.memory
    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
    end
    if options.memory > 1
    sHistory = sHistory([2:end, 1]);
    yHistory = yHistory([2:end, 1]);
    rhoHistory = rhoHistory([2:end 1]);
    end
    if options.memory > 0
    sHistory{options.memory} = sk;
    yHistory{options.memory} = yk;
    rhoHistory{options.memory} = rhok;
    end

    # If we are not out of memory
    else

    for i = 1:k
    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
    end
    sHistory{k+1} = sk;
    yHistory{k+1} = yk;
    rhoHistory{k+1} = rhok;

    end

    k = k + 1;

    # The cautious step is rejected: we do not store sk, yk, rhok but
    # we still need to transport stored vectors to the new tangent
    # space.
    else

    accepted = false;

    for  i = 1 : min(k, options.memory)
    sHistory{i} = M.transp(xCur, xNext, sHistory{i});
    yHistory{i} = M.transp(xCur, xNext, yHistory{i});
    end

    end

    # Update variables to new iterate.
    storedb.removefirstifdifferent(key, newkey);
    xCur = xNext;
    key = newkey;
    xCurGradient = xNextGrad;
    xCurGradNorm = M.norm(xNext, xNextGrad);
    xCurCost = xNextCost;


end

function get_quasi_Newton_Direction(p::GradientProblem, o::Standard_quasi_Newton_Options)
        return -o.Hessian_Inverse_Aproximation(p.M, o.x, get_gradient(p,o.x))
end

function get_quasi_Newton_Direction(p::GradientProblem, o::Limited_Memory_quasi_Newton_Options)

        q = get_gradient(p, o.x)

    inner_s_q = zeros(1, k);

    for i = k : -1 : 1
        inner_s_q(1, i) = rhoHistory{i} * M.inner(xCur, sHistory{i}, q);
        q = M.lincomb(xCur, 1, q, -inner_s_q(1, i), yHistory{i});
    end

    r = M.lincomb(xCur, scaleFactor, q);

    for i = 1 : k
         omega = rhoHistory{i} * M.inner(xCur, yHistory{i}, r);
         r = M.lincomb(xCur, 1, r, inner_s_q(1, i)-omega, sHistory{i});
    end

    dir = M.lincomb(xCur, -1, r);
end


get_solver_result(o::O) where {O <: quasi_Newton_Options} = o.x
