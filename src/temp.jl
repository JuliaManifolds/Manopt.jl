using Manopt

M = SymmetricPositiveDefinite(2)

for i= 1:1000
    a = rand(M)
    b = rand(M)
    c = rand(M)
    d = rand(M)
    m = b

    ξb = log(M,b,c) + parallelTransport(M,c,b,log(M,c,a)) - log(M,b,d)
        + parallelTransport(M,m,b,log(M,m,d)) - parallelTransport(M,m,b,log(M,m,a))
    ηb = log(M,b,c)

    v = inner(M,b,ξb,ηb)

    print("Test: ",v,"\n")
end