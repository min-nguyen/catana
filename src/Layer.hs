{-# LANGUAGE PatternSynonyms, ViewPatterns #-}

module Layer where

import Free

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |                        Fully Connected NN                      | | ---
  --- ‾------------------------------------------------------------------‾---

-- | auxiliary types

-- layer parameters
type Biases  = [Double]
type Weights = [[Double]]
-- layer inputs
type Values  = [Double]
-- data for back propagation
type Deltas  = [Double]
data BackProp = BackProp { inputStack :: [Values], nextWeights :: Weights, nextDeltas :: Deltas, desiredOutput :: Values}

-- | layers of a fully connected neural network
data DenseLayer k = DenseLayer Weights Biases k  deriving Functor
data InputLayer k = InputLayer            deriving Functor

denselayer :: (DenseLayer ⊂ nn) => Weights -> Biases -> Free nn ()
denselayer w b = inject (DenseLayer w b (Pure ()))

inputlayer :: (InputLayer ⊂ nn) => Free nn a
inputlayer = inject InputLayer

pattern DenseLayer' :: (DenseLayer ⊂ nn) => Weights -> Biases ->  a -> nn a
pattern DenseLayer' w b k <- (prj -> Just (DenseLayer w b k))

pattern InputLayer' :: (InputLayer ⊂ nn) => nn a
pattern InputLayer' <- (prj -> Just InputLayer)

-- | other util
class Functor f => AlgShow f  where
  algShow :: f String -> String

instance  {-# OVERLAPPING #-}  AlgShow InputLayer  where
  algShow InputLayer =  "InputLayer "

instance  {-# OVERLAPPING #-}  AlgShow DenseLayer  where
  algShow (DenseLayer w b k) = k ++ ("(DenseLayer " ++ show w ++ " " ++ show b ++ ") ")

instance {-# OVERLAPPABLE #-}  (AlgShow f , AlgShow g ) => AlgShow (f :+: g)  where
    algShow (L r) =  algShow r
    algShow (R r) =  algShow r

showNN :: (AlgShow f, Show a) => Free f a -> String
showNN = eval algShow (const "")
