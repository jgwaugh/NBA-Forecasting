# Confidence Intervals

I've found it easiest to approach confidence intervals from a single feature perspective. I might change this over time, but for now, this suffices as a first pass. 

Suppose we are trying to predict $X_t | X_{t-1}$. The simplest formulation would be something like this

$$ x_t = f(x_{t-1}, t) + \sigma(x_{t - 1}, t)\epsilon $$

Where $\epsilon \sim N(0, 1)$

We can fit $ \hat{f}(x_{t-1}, t) $ on the training set, and on the validation set, write the following

$$  Z = log(|x_t - \hat{f}(x_{t-1}, t)|) = log(\sigma(x_{t-1}, t)) + log(\epsilon) $$

If we can train a model, $ g(x_{t-1}, t) $ to predict $ Z$, then we should have

$$ exp(\hat{Z}) = \hat{\sigma}(x_{t-1}, t) \approx \sigma(x_{t - 1}, t) $$

From here, confidence intervals can be build. 