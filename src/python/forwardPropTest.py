# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 09:29:01 2016

@author: Justin Fletcher
"""

import numpy as np
from matplotlib import pyplot as plt
import time


def createSynapseMatrix(layerSpecVector):

    # Get the number of nodes in the largest layer.
    maxNumNodes = np.max(layerSpecVector)

    # Get the number of layers.
    numLayers = len(layerSpecVector)
	# add a +1 in the 1st dimension here to make all bias..
    # Initialize an empty array to contain the weights.
    weightsArray = np.zeros((maxNumNodes,maxNumNodes,numLayers-1))

    for layer in range(0,numLayers-1):
        
        thisLayerNumNeurons = layerSpecVector[layer]

        nextLayerNumNeurons = layerSpecVector[layer+1]
        weightsArray[0:thisLayerNumNeurons, 0:nextLayerNumNeurons, layer] = 0.2*(np.random.rand(thisLayerNumNeurons, nextLayerNumNeurons)-0.5)

    return(weightsArray)

#Any neural computer can be carved from the neural substrate... The question is how...
def viewWeightMatrix(weightMatrix):
    for weightLayer in range(0,np.shape(weightMatrix)[2]):
    
          
        plt.subplot(1, np.shape(weightMatrix)[2], weightLayer+1)
        plt.imshow(weightMatrix[:,:,weightLayer], interpolation="nearest", cmap="RdBu")
    #    inputVector = np.tanh((np.dot(inputVector, weightMatrix[:,:,weightLayer]))

def dotProp(weightMatrix, inputVector):
    
    propPattern = inputVector
    
    for weightLayer in range(0,np.shape(weightMatrix)[2]):

        propPattern = np.tanh((np.dot(propPattern, weightMatrix[:,:,weightLayer])))
        
    return(propPattern)
        
        

#viewWeightMatrix(weightMatrix)

networkWidth = 128
maxDepth = 10000
resolution = 1000


timeVector = []
weightCountVector = []

for networkDepth in range(3, maxDepth, resolution):
    
    layerSpecVector = np.repeat(networkWidth, networkDepth)
    
    weightMatrix = createSynapseMatrix(layerSpecVector)
    
    inputVector = np.random.rand(networkWidth)
    
    start = time.time()
    
    dotProp(weightMatrix, inputVector)
    
    timeVector.append(time.time() - start)
    
    weightCountVector.append(np.size(weightMatrix))
    
plt.plot(timeVector)
#plt.plot(weightCountVector)

