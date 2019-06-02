using Manopt

M = SymmetricPositiveDefinite(2)

for i= 1:1000
    a = randomMPoint(M)
    b = randomMPoint(M)
    c = randomMPoint(M)
    d = randomMPoint(M)
    m = b

    ξb = log(M,b,c) + parallelTransport(M,c,b,log(M,c,a)) - log(M,b,d)
        + parallelTransport(M,m,b,log(M,m,d)) - parallelTransport(M,m,b,log(M,m,a))
    ηb = log(M,b,c)
    
    v = dot(M,b,ξb,ηb)

    print("Test: ",v,"\n")
end