# Parameter Tracking with Machine Learning
Codes for ''Machine-learning parameter tracking with partial state observation'', a manuscript submitted to Physical Review Research.

# Example

Suppose we have a chaotic food-chain system of three species: resource, consumer, and predator, descirbed by the following set of nonlinear differential equations:
$$
\begin{align}
sin⁡(α)={opposite \over hypotenuse}={h0 \over c}={h2 \over b}
\end{align}
$$
Run ''params_extraction.m'' to get the ground truth and tracked paramter variations of the food chain system:

<img src='results/foodchain.png' width='300'>

Change ''system'' to others to track parameters of different systems, e.g., system = 'mg';
