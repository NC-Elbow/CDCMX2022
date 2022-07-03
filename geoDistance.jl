using LinearAlgebra

#=
This is a script that has some distance calculations
on the earth given two coordinates in latitude/longitude
=#

function setStatics(latitudeStart, longitudeStart, latitudeEnd, longitudeEnd)
    startPt = (latitudeStart,longitudeStart)
    endPt = (latitudeEnd, longitudeEnd)
    latitudeMovement = latitudeEnd - latitudeStart
    longitudeMovement = longitudeEnd - longitudeStart
    return startPt, endPt, latitudeMovement, longitudeMovement
end

earthRadiusKm = 6378.137
earthRadiusMile = 3963.19
earthPolarRadiusKm = 6356.7523
earthPolarRadiusMi = earthPolarRadiusKm*0.621371
# https://www.vcalc.com/wiki/vCalc/WGS-84+Earth+equatorial+radius+%28meters%29
earthDeltaRadius = earthRadiusKm - earthPolarRadiusKm
inverseFlattening = 298.25642
# https://en.wikipedia.org/wiki/Earth_ellipsoid
flattening = 1/inverseFlattening



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
    c = 2*atan2(sqrt(a), sqrt(1 - a))
    haversineDistanceKm = c*earthRadiusKm
    haversineDistanceMi = haversineDistanceKm*0.621371
    return haversineDistanceKm, haversineDistanceMi
end #function        


function Lambert(lat1,long1,lat2,long2)
    beta1 = atan((1-flattening)*tan(getRadians(lat1)))
    beta2 = np.arctan((1-flattening)*tan(getRadians(lat2)))
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