#
# debugFunctions.jl – provides several debug functions for algorithms, problems
# and sovlers.
#
# Manopt.jl, R. Bergmann, 2018-06-27
export gradientDebug
function gradientDebug(data::Dict{String,Any})
    # if step is given only output every step-th iterate
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            s = string(s,getIterationString(data), getCostString(data),
                getNormGradientString(data), getLastChangeString(data));
            print(s,"\n")
        end
    else
        s = string(s,getIterationString(data), getCostString(data),
            getNormGradientString(data), getLastChangeString(data));
        print(s,"\n")
    end
end
#
# Local Building blocks
#
function getIterationString(data::Dict{String,Any})
    if haskey(data,"Iteration")
        s = string("#",data["Iteration"]);
    else
        s="";
    end
    return s;
end
function getCost(data::Dict{String,Any})
    if haskey(data,"costFunction") && haskey(data,"x")
        x = data["x"];
        F = data["costFunction"]
        if get(data,"Format","short") == "short"
            s = string(" | ",string(F(x)) );
        else
            s = string(" | F: ",string(F(x)))
        end
    else
        s = "";
    end
    return s;
end
function getLastChangeString(data::Dict{String,Any})
    if haskey(data,"x") && haskey(data,"xold") && haskey(data,"manifold")
        M = data["manifold"]
        x = data["x"]
        xold = data["xold"]
        if get(data,"Format","short") == "short"
            s = string(" | ",string(distance(M,x,xold)));
        else
            s = string(" | Last change: ",string(distance(M,x,xold)));
        end
    else
        s = "";
    end
    return s
end
function getNormGradientString(data::Dict{String,Any})
    if haskey(data,"gradient") && haskey(data,"x") && haskey(data,"manifold")
        M = data["manifold"];
        x = data["x"];
        g = data["gradient"];
        nG = sqrt(dot(M,x,g(x),g(x)));
        if get(data,"Format","short") == "short"
            s = string(" | ",string(nG) );
        else
            s = string(" | Norm of gradient: ",string(nG));
        end
    else
        s = "";
    end
    return s;
end
