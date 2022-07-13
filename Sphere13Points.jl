using LinearAlgebra, StatsBase
#= 
We're going to try to find the 13 points on a unit sphere
which are best equidistributed
=#

function getDistanceBetweenTwoPoints(pt1::Vector, pt2::Vector)
    D = norm(pt1-pt2)
end #function    

function getMinimalSpacing(pts::Array)# 13x3 Array of unit vectors
    npts = size(pts)[1]
    spacings = []
    for p1 = 1:npts-1
        for p2 = p1+1:npts
            push!(spacings,getDistanceBetweenTwoPoints(pts[p1,:],pts[p2,:]))
        end #second for 
    end #first for
    spacing = minimum(spacings)
    #println(spacings)
    return spacing        
end #function    

function normalizePoints(pts::Array) #13x3 array of points in 3 space without the origin
    npts = size(pts)[1]
    for n = 1:npts
        p = pts[n,:]
        if norm(p) == 0
            p += rand(3)
            p = p/norm(p)
        else    
            p = p/norm(p)
        end #if    
        pts[n,:] = p
    end#for
    return pts
end #function    

function jiggerPts(pts::Array)
    pts += randn(size(pts))/size(pts)[1]
    return pts
end #function    

function acceptanceProbability(newScore, oldScore, temperature)
    ap = exp((newScore-oldScore)/temperature) #we're trying to maximize minimal spacing so we take new-old 
    return ap
end # function

function annealSphere(pts)
    T = 1
    oldScore = getMinimalSpacing(pts)
    currentScore = oldScore
    bestScore = oldScore
    bestpts = copy(pts)
    currentpts = copy(pts)
    while T > 0.001
        for sweeps = 1:200
            newpts = jiggerPts(pts)
            newScore = getMinimalSpacing(newpts)
            if newScore > bestScore
                bestScore = newScore
                bestpts = newpts
                currentScore = newScore
                currentpts = newpts
            elseif acceptanceProbability(newScore,currentScore,T) > rand()
                currentpts = newpts
                currentScore = newScore
            end
        end
        T *= 0.99 #cool by some%      
    end #while
    return bestScore, bestpts
end#function        