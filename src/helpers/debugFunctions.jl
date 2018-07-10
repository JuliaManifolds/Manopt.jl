#
# debugFunctions.jl – provides several debug functions for algorithms, problems
# and sovlers.
#
# Manopt.jl, R. Bergmann, 2018-06-27
export gradientDebug
function gradientDebug(data::Dict{String,Any})
    # if step is given only output every step-th iterate
    s::String="";
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            s = string(getIterationString(data), getCostString(data),
                getNormGradientString(data), getLastChangeString(data));
            print(s,"\n")
        end
    else
        s = string(getIterationString(data), getCostString(data),
            getNormGradientString(data), getLastChangeString(data));
        print(s,"\n")
    end
    return s;
end
function cyclicProcimalPointDebug(data::Dict{String,Any})
    # if step is given only output every step-th iterate
    s::String="";
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            s = string(getIterationString(data), getCostString(data),
                getgetKeyValueString(data,"λ"), getLastChangeString(data));
            print(s,"\n")
        end
    else
        s = string(getIterationString(data), getCostString(data),
            getgetKeyValueString(data,"λ"), getLastChangeString(data));
        print(s,"\n")
    end
    return s;
end
#
# Local Building blocks
#
function getIterationString(data::Dict{String,Any})
    s::String="";
    if haskey(data,"Iteration")
        s = string("#",data["Iteration"]);
    end
    return s;
end
function getCostString(data::Dict{String,Any})
    s::String="";
    if haskey(data,"costFunction") && haskey(data,"x")
        x = data["x"];
        F = data["costFunction"]
        if get(data,"Format","short") == "short"
            s = string(" | ",string(F(x)) );
        else
            s = string(" | F: ",string(F(x)))
        end
    end
    return s;
end
function getLastChangeString(data::Dict{String,Any})
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
function getNormGradientString(data::Dict{String,Any})
    s::String="";
    if haskey(data,"gradient") && haskey(data,"x") && haskey(data,"manifold")
        M = data["manifold"];
        x = data["x"];
        g = data["gradient"];
        nG = sqrt(dot(M,x,g,g));
        if get(data,"Format","short") == "short"
            s = string(" | ",string(nG) );
        else
            s = string(" | Norm of gradient: ",string(nG));
        end
    end
    return s;
end
function getKeyValueString(data::Dict{String,Any},key::String)
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
