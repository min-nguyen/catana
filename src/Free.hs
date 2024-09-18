module Free where

---- |‾| ------------------------- |‾| ----
 --- | |        Free monads        | | ---
  --- ‾-----------------------  ----‾---
data Free f a = Pure a | Op (f (Free f a)) deriving Functor

instance (Functor f) => Applicative (Free f) where
  pure = Pure
  Pure f  <*> as  = fmap f as
  Op faf  <*> as  = Op (fmap (<*> as) faf)

instance (Functor sig) => Monad (Free sig) where
  return          = Pure
  Pure v >>= prog = prog v
  Op op >>= prog  = Op (fmap (>>= prog ) op)

inject :: (sub ⊂ sup) => sub (Free sup a) -> Free sup a
inject = Op . inj

project :: (sub ⊂ sup) => Free sup a -> Maybe (sub (Free sup a))
project (Op s) = prj s
project _      = Nothing

---- |‾| -------------------------------------- |‾| ----
 --- | |     Folds and Unfolds over Free Monads | | ---
  --- ‾------------------------------------------‾---
-- generalised fold (catamorphism)
eval :: Functor f => (f b -> b) -> (a -> b) -> Free f a -> b
eval _ gen (Pure x) = gen x
eval alg gen (Op f) = (alg . fmap (eval alg gen)) f

-- generalised unfold (anamorphism)
build :: Functor f => (b -> f b) -> b -> Free f a
build f = Op . fmap (build f) . f

-- pairing two algebras
pairAlg :: Functor f => (f a -> a, f b -> b) -> f (a, b) -> (a, b)
pairAlg (h, g) f = (h (fst <$> f), g (snd <$> f))

-- pairing two generators
pairGen :: (a -> b, a -> c) -> a -> (b, c)
pairGen (gen_h, gen_g) a = (gen_h a, gen_g a)

---- |‾| ------------------------- |‾| ----
 --- | |        Coproducts         | | ---
  --- ‾-----------------------  ----‾---
data (f :+: g) k = L (f k) | R (g k)
  deriving Functor

class (Functor sub, Functor sup) => sub ⊂ sup where
  inj :: sub a -> sup a
  prj :: sup a -> Maybe (sub a)

instance {-# OVERLAPPING #-} Functor sig => sig ⊂ sig where
  inj = id
  prj = Just

instance {-# OVERLAPPING #-} (Functor f, Functor g) => f ⊂ (f :+: g) where
  inj         = L
  prj (L fa)  = Just fa
  prj _       = Nothing

instance {-# OVERLAPPING #-} (Functor g, f ⊂ h) => f ⊂ (g :+: h) where
  inj         = R . inj
  prj (R ga)  = prj ga
  prj _       = Nothing