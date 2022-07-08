using LinearAlgebra

#=
This is calculating pi using a Monte Carlo method
=#

numTrials = 10000 #10^5
pts = rand(numTrials, 2)

count = 0
for 