# Parameter Tracking with Machine Learning
Codes for ''Machine-learning parameter tracking with partial state observation'', a manuscript submitted to Physical Review Research.

# Example

Suppose we have a chaotic food-chain system of three species: resource, consumer, and predator, descirbed by the following set of nonlinear differential equations:
$
\begin{aligned}
\frac{dR}{dt} &= {R(1-\frac{R}{\rm K}) - \frac{ {\rm x_c y_c} C R}{R+ {\rm R_0}}}, \nonumber \\
\frac{d C}{dt} &= {\rm x_c} C (\frac{{\rm y_c} R}{R+{\rm R_0}}-1) - \frac{{\rm x_p y_p } P C}{C+{\rm C_0}} , \\
\frac{d P}{dt} &= {\rm x_p} P(\frac{ {\rm y_p} C}{C + {\rm C_0}}-1), \nonumber
\end{aligned}
$
Run ''params_extraction.m'' to get the ground truth and tracked paramter variations of the food chain system:

<img src='results/foodchain.png' width='300'>

Change ''system'' to others to track parameters of different systems, e.g., system = 'mg';
