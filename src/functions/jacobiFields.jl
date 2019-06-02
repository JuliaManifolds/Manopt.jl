#
#
# JacobiFields
#
#
export βDgx, βDexpx, βDexpξ, βDlogx, βDlogy

@doc doc"""
    βDgx(κ,t,d)
weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_x g(t;x,y)[\eta]$.
They are

```math
\beta(\kappa) = \begin{cases}
\frac{\sinh(d(1-t)\sqrt{-\kappa})}{\sinh(d\sqrt{-\kappa})}
&\text{ if }\kappa < 0,\\
1-t & \text{ if } \kappa = 0,\\
\frac{\sin((1-t)d\sqrt{\kappa})}{\sinh(d\sqrt{\kappa})}
&\text{ if }\kappa > 0.
\end{cases}
```

Due to a symmetry agrument, these are also used to compute $D_y g(t; x,y)[\eta]$

# See also
 [`DyGeo`](@ref), [`DxGeo`](@ref), [`jacobiField`](@ref)
"""
function βDgx(κ::Number,t::Number,d::Number)
    if (d==0) || (κ==0)
        return (1-t)
    else
        if κ < 0
            return sinh(sqrt(-κ)*(1-t)*d)/sinh(sqrt(-κ)*d)
        elseif κ > 0
            return sin(sqrt(κ)*(1-t)*d)/sin(sqrt(κ)*d)
        end
    end
end
@doc doc"""
    βDexpx(κ,t,d)

weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_x \exp_x(\xi)[\eta]$. They are

```math
\beta(\kappa) = \begin{cases}
\cosh(\sqrt{-\kappa})&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\cos(\sqrt{\kappa}) &\text{ if }\kappa > 0.
\end{cases}
```

# See also
 [`DxExp`](@ref), [`jacobiField`](@ref)
"""
function βDexpx(κ::Number,t::Number,d::Number)
    if κ < 0
        return cosh(sqrt(-κ)*d)
    elseif κ > 0
        return cos(sqrt(κ)*d)
    else
        return 1.0;
    end
end
@doc doc"""
    βDexpξ(κ,t,d)

weights for the [`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_\xi \exp_x(\xi)[\eta]$. They are

```math
$\beta(\kappa) = \begin{cases}
\frac{\sinh(d\sqrt{-\kappa})}{d\sqrt{-\kappa}}&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\frac{\sin(d\sqrt{\kappa})}{\sqrt{d\kappa}}&\text{ if }\kappa > 0.
\end{cases}
```

# See also
 [`DξExp`](@ref), [`jacobiField`](@ref)
"""
function βDexpξ(κ::Number,t::Number,d::Number)
    if (d==0) || (κ==0)
        return 1.0
    else
        if κ < 0
            return sinh(sqrt(-κ)*d)/( d*sqrt((-κ)) )
        elseif κ > 0
            return sin( sqrt(κ)*d )/( d*sqrt(κ) )
        end
    end
end
@doc doc"""
    βDlogx(κ,t,d)

weights for the[`jacobiField`](@ref) corresponding to the differential of the geodesic
with respect to its start point $D_x \log_xy[\eta]$. They are

```math
\beta(\kappa) = \begin{cases}
-\sqrt{-\kappa}d\frac{\cosh(d\sqrt{-\kappa})}{\sinh(d\sqrt{-\kappa})}&\text{ if }\kappa < 0,\\
-1 & \text{ if } \kappa = 0,\\
-\sqrt{\kappa}d\frac{\cos(d\sqrt{\kappa})}{\sin(d\sqrt{\kappa})}&\text{ if }\kappa > 0.
\end{cases}
```

# See also
[`DxLog`](@ref), [`DyLog`](@ref), [`jacobiField`](@ref)
"""
function βDlogx(κ::Number,t::Number,d::Number)
    if (d==0) || (κ==0)
        return -1.0
    else
        if κ < 0
            return - sqrt(-κ)*d*cosh(sqrt(-κ)*d)/sinh(sqrt(-κ)*d)
        else #if κ > 0
            return - sqrt(κ)*d*cos(sqrt(κ)*d)/sin(sqrt(κ)*d)
        end
    end
end
@doc doc"""
    βDlogy(κ,t,d)

weights for the JacobiField corresponding to the differential of the logarithmic
map with respect to its argument $D_y \log_xy[\eta]$.
They are
```math
\beta(\kappa) = \begin{cases}
\frac{ d\sqrt{-\kappa} }{\sinh(d\sqrt{-\kappa})}&\text{ if }\kappa < 0,\\
1 & \text{ if } \kappa = 0,\\
\frac{ d\sqrt{\kappa} }{\sin(d\sqrt{\kappa})}&\text{ if }\kappa > 0.
\end{cases}
```

# See also
 [`DyLog`](@ref), [`jacobiField`](@ref)
"""
function βDlogy(κ::Number,t::Number,d::Number)
    if (d==0) || (κ==0)
        return 1.0
    else
        if κ < 0
            return sqrt(-κ)*d/sinh(sqrt(-κ)*d)
        else #if κ > 0
            return sqrt(κ)*d/sin(sqrt(κ)*d)
        end
    end
end
