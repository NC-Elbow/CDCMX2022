using LinearAlgebra
#points = rand(10,2) #make some random points in the unit square
function reverseSection(points::Matrix)
    nPoints = size(points)[1]
    v = collect(1:nPoints)
    r1, r2 = sort(sample(v,2,replace = false))
    v[r1:r2] = v[collect(r2:-1:r1)]
    newPoints = points[v,:]
    return newPoints    
end

function calculateDistanceBetweenTwoPoints(pt1::Vector, pt2::Vector)
    D = norm(pt1 - pt2)
    return D
end

function calculateTripDistance(points::Matrix)
    nPoints = size(points)[1]
    totalDistance = 0
    for p = 1:nPoints-1
        totalDistance += calculateDistanceBetweenTwoPoints(points[p,:], points[p+1,:])
    end
    totalDistance += calculateDistanceBetweenTwoPoints(points[nPoints,:], points[1,:])    
    return totalDistance
end        

function acceptanceProbability(newCost, oldCost, temperature)
    ap = exp((oldCost - newCost)/temperature)
    return ap
end    

function plotTrip(points)
    trip = [points; points[1,:]']
    plot(trip[:,1], trip[:,2])
end    

function anneal(points::Matrix)
    T = .09
    shortestDistance = calculateTripDistance(points)
    shortestTrip = points
    currentDistance = shortestDistance #only a copy of this
    currentTrip = points
    while T > 0.0025 #freezing temperature
        for sweep = 1:30000
            newTrip = reverseSection(currentTrip)
            newDistance = calculateTripDistance(newTrip)
            if newDistance < shortestDistance
                shortestDistance = newDistance
                shortestTrip = newTrip
                currentDistance = newDistance
                currentTrip = newTrip
                println(shortestDistance) #can comment this away if desired
                println(T)#can comment this away if desired
            elseif acceptanceProbability(newDistance,currentDistance,T) > rand()
                currentTrip = newTrip
                currentDistance = newDistance
            end
        end
        T *= 0.97 #cool by some%      
    end
    println(shortestDistance)
    #plotTrip(shortestTrip)
    return shortestTrip    
end

#=
To run this code for 30 points...

points = rand(30,2) #gives 30 points in the unit square
plotTrip(points) #see how inefficient the starting position is
println("Initial trip distance is: ", calculateTripDistance(points))

newPoints = anneal(points)
plotTrip(newPoints)
println("A near optimal solution has distance: ", calculateTripDistance(newPoints))

=#

#= 
Two things to play around with....

Try new neighbor solutions.
Switch two points, change two complete sections, reverse two section, try cycles,
Try changing the "annealing schedule"  That is, how fast the temperature reduces.
=#