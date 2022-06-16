using LinearAlgebra, StatsBase

#=
This is a script working through hitting time on the Petersen graph
=#

block = zeros(5,5)
A11 = block + diagm(1=>ones(4),-1=>ones(4))
A11[1,5] = 1
A11[5,1] = 1 #A11 is the outer pentagon
A12 = diagm(ones(5))
A21 = diagm(ones(5)) #these are the connecting vertices 1<->6 ... k <-> 5+k
A22 = block + diagm(2=>ones(3),3=>ones(2),-2=>ones(3),-3=>ones(2))

A = [A11 A12; A21 A22]

W = A/3;

function basis(n)
    v = zeros(10) 
    v[n] = 1
    return v
end 

function performOneStep(walkingMatrix, initialLocation)
    v0 = basis(initialLocation);
    W = walkingMatrix;
    wts = W*v0; #this is done as a column because W is symmetric so row and column eigenspaces are identical
    v1Location = wsample(collect(1:size(walkingMatrix)[1]), wts)
    #v1 = basis(v1Location)
    return v1Location
end    

function MCRandomWalk(walkingMatrix, initialLocation, finalLocation)
    counter = 0;
    loc = initialLocation
    while loc != finalLocation
        loc = performOneStep(W,loc)
        counter += 1;
    end #while
    return counter
end # function

function EstimateHittingTime(walkingMatrix, intialLocation, fincalLocation, numberTrials)
    steps = []
    for k=1:numberTrials
        push!(steps, MCRandomWalk(walkingMatrix,intialLocation, fincalLocation))
    end #for
    return mean(steps) 
end#function

function calculateHittingTime(walkingMatrix, initialLocation, finalLocation)
    W = walkingMatrix;
    v0 = initialLocation;
    vf = finalLocation;
    probs = [W[v0,vf]]
    cummulativeProbs = cumsum(probs)
    stepCounter = 1
    while cummulativeProbs[end] < 1
        newprob = (1 - (W^(stepCounter))[v0,vf])*(W^(stepCounter + 1))[v0,vf]
        #=
        This step needs a little explaining.
        We want to count how long it takes to reach a vertex for the first time.
        In this case we're looking for the probability P(X_k = vf | X_(k-1) != vf)
        In other words, how likely is it that we reach the final vertex at step k, 
        given the step k-1 landed elsewhere.  
        So the probability of step k-1 landing elsewhere is the entry in the walking 
        matrix raised to the (number of) steps at entry [v0,vf]
        The probability we've missed the final vertex is 1 - computed probability.
        Since we're given conditional probabilities, the probabilities multiply.
        We now push the new probability to the vector of probabilities
        and we sum all the probabilities.  In this way, we can see the point 
        at which the cummulative dsitribution reaches probability 1.
        The indexing may to wrong by one.
        =#
        push!(probs,newprob)
        cummulativeProbs = cumsum(probs)
        stepCounter += 1
    end #while 
    return stepCounter 
end #function     

