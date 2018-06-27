#
# debugFunctions.jl – provides several debug functions for algorithms, problems
# and sovlers.
#
# Manopt.jl, R. Bergmann, 2018-06-27
export gradientDebug
# TODO do single string generators on the dictionary returning strings
# to concatenate them for own debugs
function gradientDebug(data::Dict{String,Any})
    s = "";
    Format = get(data,"Format","short");
    if haskey(data,"Iteration")
        sLocal = string("#",data["Iteration"]);
        s = string(s,sLocal);
    end
    if haskey(data,"costFunction") && haskey(data,"x")
        x = data["x"];
        F = data["costFunction"]
        if Format == "short"
            sLocal = string(" | ",string(F(x)) );
        else
            sLocal = string(" | F: ",string(F(x)))
        end
        s = string(s,sLocal);
    end
    if haskey(data,"gradient") && haskey(data,"x") && haskey(data,"manifold")
        M = data["manifold"];
        x = data["x"];
        g = data["gradient"];
        nG = sqrt(dot(M,x,g(x),g(x)));
        if Format == "short"
            sLocal = string(" | ",string(nG) );
        else
            sLocal = string(" | Norm of gradient: ",string(nG));
        end
        s = string(s,sLocal);
    end
    if haskey(data,"x") && haskey(data,"xold") && haskey(data,"manifold")
        M = data["manifold"]
        x = data["x"]
        xold = data["xold"]
        if Format == "short"
            sLocal = string(" | ",string(distance(M,x,xold)));
        else
            sLocal = string(" | Last change: ",string(distance(M,x,xold)));
        end
        s = string(s,sLocal);
    end
    if haskey(data,"step") && haskey(data,"Iteration")
        if mod(data["Iteration"],data["step"]) == 0
            print(s,"\n")
        end
    else
        print(s,"\n")
    end
end
