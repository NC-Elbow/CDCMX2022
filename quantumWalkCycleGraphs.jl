using LinearAlgebra, Plots
#=
This will 
=#
function buildLaplacian(nPoints)
    L = 2*diagm(ones(nPoints))
    L += -1*diagm(1=> ones(nPoints - 1))
    L += -1*diagm(-1=> ones(nPoints - 1))
    L[1,end] = -1
    L[end,1] = -1
    return L
end #function


function getUnitary(t, Laplacian)
    L = Laplacian
    U = exp(-1im*t*L)
    return U
end

function basis(k,nPoints)
    v = zeros(nPoints)
    v[k] = 1
    return v
end

function getProbability(initialPosition, finalPosition, Laplacian, tempo)
    L = Laplacian
    nPoints = size(L)[1]
    amplitude = basis(finalPosition,nPoints)'*getUnitary(tempo,Laplacian)*basis(initialPosition,nPoints)
    prob = norm(amplitude)^2
    return prob
end

#=
Let's try a 100 vertex cycle graph

L = buildLaplacian(100)
getProbability(1,50,L,10)

=#