function calculateGammaDistShapeScale(mu, variance)

  scale = variance/mu
  shape = mu/scale
  return(shape,scale)
end


function calcGammaCumDistFitToVec(v, shape, scale)

end

