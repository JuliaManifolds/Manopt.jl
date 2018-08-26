export S1Signal

@doc doc"""
    S1Signal(x)
evaluate the example signal $f(x), x\in [0,1]$
of phase-valued data introduces in Sec. 5.1 of

> Bergmann, Laus, Steidl, Weinmann, Second Order Differences of Cyclic Data and
> Applications in Variational Denoising, SIAM J. Imaging Sci., 7(4), 2916–2953, 2014.
> doi: [10.1137/140969993](https://dx.doi.org/10.1137/140969993)

# Optional
- `wrap` : (`true`) to obtain the real valued (unwrapped) signal set `wrap` to false
"""
function S1Signal(x, wrap=true)
    if x < 0
        y = missing
    elseif x <= 1/4
        y = - 24 * π * (x-1/4)^2  +  3/4 * π
    elseif x <= 3/8
        y = 4 * π * x  -  π / 4
    elseif x <= 1/2
        y = - π * x - 3*π/8;
    elseif x <= (3*0 + 19)/32
        y = - (0+7)/8*π
    elseif x <= (3*1 + 19)/32
        y = - (1+7)/8*π
    elseif x <= (3*2 + 19)/32
        y = - (2+7)/8*π
    elseif x <= (3*3 + 19)/32
        y = - (3+7)/8*π
    elseif x <= 1
        y = 3 / 2 * π * exp(8-1/(1-x))  -  3/4*π
    else
        y = missing
    end
    if wrap && !ismissing(y)
        y = symRem(y)
    end
    return y
end
