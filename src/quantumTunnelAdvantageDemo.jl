height = 0:0.01:2
width = transpose(0.01:0.01:2)
propTraversal_thermal = exp(-(height))
propTraversal_quantum = exp(-width.*(sqrt(height)))
# plot(propTraversal_thermal)
# plot(propTraversal_quantum)

quantumAdvantage = propTraversal_quantum./propTraversal_thermal
mesh(quantumAdvantage)