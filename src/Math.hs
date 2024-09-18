module Math where

-- vector addition
addV :: [Double] -> [Double] -> [Double]
addV = zipWith (+)

-- matrix−vector multiplication
mulMV :: [[Double]] -> [Double] -> [Double]
mulMV xss ys = map (sum . zipWith (*) ys) xss

-- sigmoid function (over a vector)
sigmoid :: [Double] -> [Double]
sigmoid = map (\x -> 1.0 / (1.0 + exp (negate x)))

-- vector multiplication
mulV :: [Double] -> [Double] -> [Double]
mulV   = zipWith (*)

-- vector subtraction
subV :: [Double] -> [Double] -> [Double]
subV  = zipWith (-)

-- matrix subtraction
subM :: [[Double]] -> [[Double]] -> [[Double]]
subM  = zipWith subV

-- outer product
outerProd :: [Double] -> [Double] -> [[Double]]
outerProd xs ys = map (\x -> map (x *) ys) xs

-- inverse then differential sigmoid function  (over a vector)
invSigmoid :: [Double] -> [Double]
invSigmoid = map (\x -> let y = log(x/(1 - x)) in y * (1 - y))