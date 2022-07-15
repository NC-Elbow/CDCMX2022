using LinearAlgebra, Plots, Statistics, StatsBase
using CSV, DataFrames

function setStatics(latitudeStart, longitudeStart, latitudeEnd, longitudeEnd)
    startPt = (latitudeStart,longitudeStart)
    endPt = (latitudeEnd, longitudeEnd)
    latitudeMovement = latitudeEnd - latitudeStart
    longitudeMovement = longitudeEnd - longitudeStart
    return startPt, endPt, latitudeMovement, longitudeMovement
end

global earthRadiusKm = 6378.137
global earthRadiusMile = 3963.19
global earthPolarRadiusKm = 6356.7523
global earthPolarRadiusMi = earthPolarRadiusKm*0.621371
# https://www.vcalc.com/wiki/vCalc/WGS-84+Earth+equatorial+radius+%28meters%29
global earthDeltaRadius = earthRadiusKm - earthPolarRadiusKm
global inverseFlattening = 298.25642
# https://en.wikipedia.org/wiki/Earth_ellipsoid
global flattening = 1/inverseFlattening

function getRadians(degreeCoord)
    radianCoord = degreeCoord*pi / 180
    return radianCoord
end

function Haversine(lat1, long1, lat2, long2)
    startPt, endPt, latMovement, longMovement = setStatics(lat1,long1,lat2,long2)
    dlat = getRadians(abs(latMovement))
    dlong = getRadians(abs(longMovement))
    lat1Rad = getRadians(lat1)
    lat2Rad = getRadians(lat2)
    a = sin(dlat/2)^2 + cos(lat1Rad)*cos(lat2Rad)*(sin(dlong/2)^2)
    c = 2*atan(sqrt(a), sqrt(1 - a))
    haversineDistanceKm = c*earthRadiusKm
    haversineDistanceMi = haversineDistanceKm*0.621371
    return haversineDistanceKm, haversineDistanceMi
end #function        

function Lambert(lat1,long1,lat2,long2)
    beta1 = atan((1-flattening)*tan(getRadians(lat1)))
    beta2 = atan((1-flattening)*tan(getRadians(lat2)))
    haversineDistanceKm, haversineDistanceMi = Haversine(lat1,long1,lat2,long2)
    centralAngle = haversineDistanceKm/earthRadiusKm
    # https://en.wikipedia.org/wiki/Geographical_distance#Ellipsoidal-surface_formulae
    P = (beta1 + beta2)/2
    Q = (beta2 - beta1)/2
    X = (centralAngle - sin(centralAngle))*(sin(P)^2)*cos(Q)^2/cos(centralAngle/2)^2
    Y = (centralAngle + sin(centralAngle))*sin(Q)^2 *cos(P)^2/sin(centralAngle/2)^2
    equatorialRadiusKM = earthPolarRadiusKm + earthDeltaRadius*cos(centralAngle)
    lambertdistanceKm = equatorialRadiusKM*(centralAngle - flattening/2*(X + Y))
    lambertdistanceMi = lambertdistanceKm*0.621371
    return lambertdistanceKm, lambertdistanceMi
end
# first neighbor type solution
function reverseSection(points::Matrix)
    nPoints = size(points)[1]
    v = collect(1:nPoints)
    r1, r2 = sort(sample(v,2,replace = false))
    v[r1:r2] = v[collect(r2:-1:r1)]
    newPoints = points[v,:]
    return newPoints    
end

function switchTwoPoints(points::Matrix)
    nPoints = size(points)[1]
    v = collect(1:nPoints)
    r1, r2 = sort(sample(v,2,replace = false))
    v[r1],v[r2] = v[r2],v[r1]
    newPoints = points[v,:]
    return newPoints    
end    

function reverseTwoSections(points::Matrix)
    nPoints = size(points)[1]
    v = collect(1:nPoints)
    r1, r2, r3, r4  = sort(sample(v,4,replace = false))
    v[r1:r2] = v[collect(r2:-1:r1)]
    v[r3:r4] = v[collect(r4:-1:r3)]
    newPoints = points[v,:]
    return newPoints    
end

function shiftSectionByOneRight(points::Matrix)
   nPoints = size(points)[1] 
   v = collect(1:nPoints)
   r1, r2 = sort(sample(v,2,replace = false))
   v[r1:r2] = [v[r1+1:r2];v[r1]]
   newPoints = points[v,:]
   return newPoints
end     

function switchTwoBlocks(points::Matrix)
    nPoints = size(points)[1]
    v = collect(1:nPoints)
    r1, r2, r3, r4  = sort(sample(v,4,replace = false))
    newV = [v[1:r1-1];v[r3:r4];v[r2+1:r3-1];v[r1:r2];v[r4+1:end]]
    newPoints = points[newV,:]
    return newPoints    
end #function    


function calculateDistanceBetweenTwoPoints(pt1::Vector, pt2::Vector)
    D = Lambert(pt1[1],pt1[2],pt2[1],pt2[2])[1] #get actual distance on this earth between two lat/long pairs
    # This can be changed to Haversine, Euclidean, etc
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
    #=
    The coordinates are listed as lat/long, but in our plotting
    longitude is equivalent to our "x" coordinate, so we switch the columns
    =#
    points = [points[:,end] points[:,end-1]] #Latitude and Longitude are the final two columns
    trip = [points; points[1,:]']
    plot(trip[:,1], trip[:,2], label = "viaje", markershape =:circle)
end    

function anneal(points::Matrix, getNeighbor::Function)
    T = .09
    shortestDistance = calculateTripDistance(points[:,end-1:end]) #use lat/long as final two columns
    shortestTrip = points
    currentDistance = shortestDistance #only a copy of this
    currentTrip = points
    while T > 0.0025 #freezing temperature
        for sweep = 1:300 #num trials at each temperature
            newTrip = getNeighbor(currentTrip)
            newDistance = calculateTripDistance(newTrip[:, end-1:end])
            if newDistance < shortestDistance
                shortestDistance = newDistance
                shortestTrip = newTrip
                currentDistance = newDistance
                currentTrip = newTrip
                #println(shortestDistance) #can comment this away if desired
                #println(T)#can comment this away if desired
            elseif acceptanceProbability(newDistance,currentDistance,T) > rand()
                currentTrip = newTrip
                currentDistance = newDistance
            end
        end
        T *= 0.975 #cool by some%      
    end
    println(shortestDistance)
    #plotTrip(shortestTrip)
    return shortestTrip    
end

function mainProcess(filepath, numTrials, counterStart)
    df = CSV.File(filepath) |> DataFrame
    nadf = Array(df[!,[:State,:Latitude,:Longitude]])
    newpts = anneal(nadf, switchTwoPoints)
    counter = counterStart
    for k = 1:numTrials
        newpts = anneal(newpts, shiftSectionByOneRight)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
        newpts = anneal(newpts, reverseTwoSections)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
        newpts = anneal(newpts, reverseSection)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
        newpts = anneal(newpts, switchTwoPoints)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
        newpts = anneal(newpts, switchTwoBlocks)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
        newpts = anneal(newpts, switchTwoPoints)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
        newpts = anneal(newpts, reverseSection)
        p = plotTrip(newpts)
        labelnumber = string(1000+counter ,".jpg")
        name = string("/home/clark/Computing/julia_projects/TSPpics/", labelnumber)
        savefig(name)
        counter += 1
    end #for loop
end #function       
