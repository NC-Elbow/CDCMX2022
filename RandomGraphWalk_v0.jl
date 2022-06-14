using LinearAlgebra

#=
This is a short script which provides a function for performing a random walk on a graph given an adjacency Matrix
weighted or unweighted.
=#

function basis(n,k)
    v = zeros(k)
    v[n]=1
    return v
end

function RandomWalk(adjacencyMatrix, initialLocation, numSteps)
    rwalk = []
    A = adjacencyMatrix
    B = A./sum(A,dims=1) #this will make column vectors appropriate for randomwalks
    r,c = size(A) #r and c should be the same, we'll use c as the dimension of basis vector
    push!(rwalk, initialLocation)
    v = basis(initialLocation, c)
    for k = 1:numSteps
        newV = B*v #probabilities of landing at each vertex. These become weights of the sampling
        newLoc = wsample(collect(1:c), newV)
        push!(rwalk,newLoc)
        v = basis(newLoc, c)
    end
    return rwalk
end        

