#
# debugFunctions.jl – provides several debug functions for algorithms, problems
# and sovlers.
#
# Manopt.jl, R. Bergmann, 2018-06-27
export gradientDebug,cyclicProximalPointDebug, subGradientDebug
"""
    gradientDebug(v)
print all fields of a gradient descent method, if they are present in the dictionary
of values `v`.
"""
function gradientDebug(data::Dict{String,Any})
    # if step is given only output every step-th iterate
    s::String=""
    r = getStopReasonString(data)
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            s = string(getIterationString(data), getCostString(data),
                getNormGradientString(data), getLastChangeString(data),
                length(r)>0 ? "\n$(r)" : "")
            print(s,"\n")
        end
    else
        s = string(getIterationString(data), getCostString(data),
            getNormGradientString(data), getLastChangeString(data),
            length(r)>0 ? "\n$(r)" : "")
        print(s,"\n")
    end
    return s;
end
"""
    subGradientDebug(v)
print all fields of a subgradient descent method, if they are present in the dictionary
of values `v`.
"""
function subGradientDebug(data::Dict{String,Any})
    # if step is given only output every step-th iterate
    s::String=""
    r = getStopReasonString(data)
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            s = string(
                getIterationString(data), getCostString(data,"xopt"),
                getNormSubGradientString(data), getLastChangeString(data),
                length(r)>0 ? "\n$(r)" : "")
            print(s,"\n")
        end
    else
        s = string(getIterationString(data), getCostString(data),
            getNormGradientString(data), getLastChangeString(data),
            length(r)>0 ? "\n$(r)" : "")
        print(s,"\n")
    end
    return s;
end
"""
     cyclicProcimalPointDebug(v)
create the string conatining all CPPA values present in the dictionary `v`.
"""
function cyclicProximalPointDebug(data::Dict{String,Any})
    # if step is given only output every step-th iterate
    s::String=""
    r = getStopReasonString(data)
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            s = string(getIterationString(data), getCostString(data),
                getKeyValueString(data,"λ"), getLastChangeString(data),
                length(r)>0 ? "\n$(r)" : "")
            print(s,"\n")
        end
    else
        s = string(getIterationString(data), getCostString(data),
            getKeyValueString(data,"λ"), getLastChangeString(data),
            length(r)>0 ? "\n$(r)" : "")
        print(s,"\n")
    end
    return s;
end
#
# Local Building blocks
#
"""
    getIterationString(v)
returns the String `#I`, where `I` is the value in the field "Iteration" of v,
otherwise and empty string.
"""
function getIterationString(data::Dict{String,<:Any})
    s::String="";
    if haskey(data,"Iteration")
        s = string("#",data["Iteration"]);
    end
    return s;
end
"""
    getCostString(data[,Point="x"])
returns (if both items presen) the cost function evaluated at the point stored
in the field addressed by `Point`, which is by default `"x"`.
"""
function getCostString(data::Dict{String,<:Any},Point::String="x")
    s::String="";
    if haskey(data,"costFunction") && haskey(data,Point)
        x = data[Point];
        F = data["costFunction"]
        if get(data,"Format","short") == "short"
            s = string(" | ",string(F(x)) );
        else
            s = string(" | F: ",string(F(x)))
        end
    end
    return s;
end
"""
    getLastChangeString(data)
returns (if all 3 items provided) last change between `x` and `xnew` measured
using the distance on their manifold M.
"""
function getLastChangeString(data::Dict{String,<:Any})
    s::String="";
    if haskey(data,"x") && haskey(data,"xnew") && haskey(data,"manifold")
        M = data["manifold"]
        x = data["x"]
        xnew = data["xnew"]
        if get(data,"Format","short") == "short"
            s = string(" | ",string(distance(M,x,xnew)));
        else
            s = string(" | Last change: ",string(distance(M,x,xnew)));
        end
    end
    return s
end
"""
    getNormSubGradientString(data)
returns (if provided) the subgradient norm as a string from the debug values.
"""
function getNormSubGradientString(data::Dict{String,<:Any})
    s::String="";
    if haskey(data,"subgradient") && haskey(data,"x") && haskey(data,"manifold")
        M = data["manifold"];
        x = data["x"];
        g = data["subgradient"];
        nG = sqrt(dot(M,x,g,g));
        if get(data,"Format","short") == "short"
            s = string(" | ",string(nG) );
        else
            s = string(" | Norm of subgradient: ",string(nG));
        end
    end
    return s;
end
"""
    getNormGradientString(data)
returns (if provided) the gradient norm as a string from the debug values.
"""
function getNormGradientString(data::Dict{String,<:Any})
    s::String=""
    if haskey(data,"gradient") && haskey(data,"x") && haskey(data,"manifold")
        M = data["manifold"]
        x = data["x"]
        g = data["gradient"]
        nG = sqrt(dot(M,x,g,g))
        if get(data,"Format","short") == "short"
            s = string(" | ",string(nG) )
        else
            s = string(" | Norm of gradient: ",string(nG))
        end
    end
    return s
end
"""
   getStopReasonString(data)
returns the stop reason if it is within the debug values and a nonempty string.
"""
function getStopReasonString(data::Dict{String,<:Any})
    s::String=""
    if length(get(data,"StopReason","")) > 0
        s = string(" | ",data["StopReason"])
    end
    return s
end
"""
   getKeyValueString(data,key)
extracts a key-value pair addresses by its `key` from the debug dictionary.
"""
function getKeyValueString(data::Dict{String,<:Any},key::String)
    s::String="";
    if haskey(data,key)
        k = data[key];
        if get(data,"Format","short") == "short"
            s = string(" | ",string(k) );
        else
            s = string(" | ",key,": ",string(k));
        end
    end
    return s;
end
