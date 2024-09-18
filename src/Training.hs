module Training where

import Free
import Layer
import Forward
import Backward

---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |         Training as a Fold then Unfold                         | | ---
  --- ‾------------------------------------------------------------------‾---
-- training a fully connected neural network for a single input and desired output
trainAsFoldUnfold :: forall nn a. (InputLayer ⊂ nn, AlgFwd nn, CoalgBwd nn)
  => Free nn a -> (Values, Values) ->  Free nn a
trainAsFoldUnfold nn (input, des_output)  = (runBackwardCoalg . h . runForwardAlg) nn
  where
  -- intermediary function that connects output of forward prop to input of back prop
  h :: (Values -> [Values]) -> (Free nn a, BackProp)
  h fwd_pass = let input_stack = fwd_pass input
               in (nn, BackProp { inputStack = input_stack, nextWeights = [], nextDeltas = [], desiredOutput = des_output })

-- training a fully connected neural network for a batch of inputs and desired outputs
trainAsFoldUnfoldMany :: (InputLayer ⊂ nn, AlgFwd nn, CoalgBwd nn)
  => Free nn a -> [(Values, Values)] -> Free nn a
trainAsFoldUnfoldMany = foldr (flip trainAsFoldUnfold)


---- |‾| -------------------------------------------------------------- |‾| ----
 --- | |         Training as a single Fold                              | | ---
  --- ‾------------------------------------------------------------------‾---
-- training a fully connected  neural network for a single input and desired output
trainAsFold :: forall nn a. (InputLayer ⊂ nn, AlgFwd nn, AlgBwd nn nn) => Free nn a -> (Values, Values) -> Free nn a
trainAsFold nn (input, des_output)  = (backwardPass . h . forwardPass) input
  where
      algTrain :: nn (Values -> [Values], BackProp -> Free nn a) -> (Values -> [Values], BackProp -> Free nn a)
      algTrain = pairAlg (algFwd, algBwd)
      genTrain :: a -> (Values -> [Values], BackProp -> Free nn a)
      genTrain = pairGen (genFwd, genBwd)

      (  forwardPass  :: Values -> [Values]
       , backwardPass :: BackProp -> Free nn a)
        = eval algTrain genTrain nn

      -- intermediary function that connects output of forward prop to input of back prop
      h :: [Values] -> BackProp
      h input_stack = (BackProp { inputStack = input_stack, nextDeltas = [], nextWeights = [], desiredOutput = des_output })

-- training a fully connected  neural network for a batch of inputs and desired outputs
trainAsFoldMany :: (InputLayer ⊂ nn, AlgFwd nn, AlgBwd nn nn)
  => Free nn a -> [(Values, Values)] -> Free nn a
trainAsFoldMany = foldr (flip trainAsFold)
