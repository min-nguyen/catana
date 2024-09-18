{-# LANGUAGE UndecidableInstances #-}

module Backward where

import Free
import Layer
import Math
import Data.List (transpose)

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |         Backward Prop as an Unfold with a Coalgebra            | | ---
  --- ‾------------------------------------------------------------------‾---
class Functor f => CoalgBwd f  where
  coalgBwd :: (Free f a, BackProp) -> f (Free f a, BackProp)

-- | backward propagation over a single layer as a coalgebra
instance CoalgBwd (InputLayer :+: DenseLayer)  where
  coalgBwd (Op InputLayer', _) = inj InputLayer
  coalgBwd (Op (DenseLayer' w b k), back_prop) =
    let (w', b', deltas) = backward w b back_prop
        back_prop' = back_prop {  inputStack = tail (inputStack back_prop),
                                  nextWeights = w,
                                  nextDeltas = deltas }
    in  inj (DenseLayer w' b' (k, back_prop'))

backward :: Weights -> Biases -> BackProp -> (Weights, Biases, Deltas)
backward weights biases (BackProp (output : input : _) next_weights next_deltas des_output) =
  let deltas = case next_deltas of
        [] -> (output `subV` des_output) `mulV` (invSigmoid input)
        _  -> (transpose next_weights `mulMV` next_deltas) `mulV` (invSigmoid input)
      weights' = weights `subM` (input `outerProd` deltas)
      biases' = biases `subV` deltas
  in  (weights', biases', deltas)

-- | backward propagation over a neural network as an unfold (build)
runBackwardCoalg :: CoalgBwd nn => (Free nn a, BackProp) -> Free nn a
runBackwardCoalg = build coalgBwd

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |         Backward Prop as an Fold with an Algebra               | | ---
  --- ‾------------------------------------------------------------------‾---
class AlgBwd f g where
  algBwd :: f (BackProp -> Free g a) -> (BackProp -> Free g a)

-- | backward propagation over a single layer as an algebra
instance {-# OVERLAPPING #-} (InputLayer ⊂ g) => AlgBwd InputLayer g where
  algBwd InputLayer = const (inputlayer)

instance {-# OVERLAPPING #-} (DenseLayer ⊂ g) => AlgBwd DenseLayer g where
  algBwd (DenseLayer w b k) back_prop =
    let (w', b', deltas) = backward w b back_prop
        back_prop' = back_prop {  inputStack = tail (inputStack back_prop),
                                  nextWeights = w,
                                  nextDeltas = deltas  }
    in  inject (DenseLayer w' b' (k back_prop'))

instance {-# OVERLAPPING #-} (AlgBwd f (f :+: g), AlgBwd g (f :+: g)) => AlgBwd (f :+: g) (f :+: g)  where
  algBwd (L r) =  algBwd r
  algBwd (R r) =  algBwd r

-- | backward propagation over a neural network as a fold (eval)
runBackwardAlg :: (AlgBwd nn nn, InputLayer ⊂ nn)  => Free nn a -> BackProp -> Free nn a
runBackwardAlg = eval algBwd genBwd

-- generator for initialising a backward pass function
genBwd :: InputLayer ⊂ nn => a -> (BackProp -> Free nn a)
genBwd _ = const (inject InputLayer)
