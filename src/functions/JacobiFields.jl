#
#
# Specific JacobiFields
#
#
export βgStartPoint

doc"""
    βgStartPoint(κ,t,d)
weights for the JacobiField corresponding to the differential of the geodesic
with respect to its start point $D_x g_{x,y}$.
"""
function βgStartPoint(κ::Number,t::Number,d::Number)
    if κ > 0
        return sin(sqrt(κ)*t*d)/sin(sqrt(κ)*d);
    elseif κ < 0
        return sinh(sqrt(-κ)*t*d)/sinh(sqrt(-κ)*d);
    else
        return t;
    end
end
