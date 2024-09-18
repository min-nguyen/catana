module Forward where

import Free
import Layer
import Math

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |         Forward Prop as a Fold with an Algebra                 | | ---
  --- ‾------------------------------------------------------------------‾---
class Functor f => AlgFwd f  where
  algFwd :: f (Values -> [Values]) -> (Values -> [Values])

-- forward propagation over a single layer as an algebra
instance  {-# OVERLAPPING #-}  AlgFwd InputLayer  where
  algFwd InputLayer = (:[])
instance  {-# OVERLAPPING #-}  AlgFwd DenseLayer  where
  algFwd :: DenseLayer (Values -> [Values]) -> Values -> [Values]
  algFwd (DenseLayer w b k) =
    (\(xs:xss) -> forward w b xs : xs : xss)  . k
   where
    -- forward propagation logic over a single layer
    forward :: Weights -> Biases -> Values ->  Values
    forward weights biases input = sigmoid (weights `mulMV` (input `addV` biases))

instance {-# OVERLAPPABLE #-}  (AlgFwd f , AlgFwd g ) => AlgFwd (f :+: g)  where
  algFwd :: (AlgFwd f, AlgFwd g) => (:+:) f g (Values -> [Values]) -> Values -> [Values]
  algFwd (L r) =  algFwd r
  algFwd (R r) =  algFwd r

-- forward propagation over a neural network as a fold (eval)
runForwardAlg :: AlgFwd nn => Free nn b -> (Values -> [Values])
runForwardAlg = eval algFwd genFwd

-- generator for initialising a forward pass function
genFwd :: a -> (Values -> [Values])
genFwd = const (: [])

-- example of running forward propagation on a fixed neural net and input
exampleForward :: [Values]
exampleForward = runForwardAlg exampleNN exampleInput
  where
  exampleNN :: Free (InputLayer :+: DenseLayer) ()
  exampleNN = do
    denselayer [[1,2],[0.2,1.7]] [0, 0]
    denselayer [[1,2],[0.2,1.5]] [0, 0]
    inputlayer
  exampleInput :: [Double]
  exampleInput = [1,2]