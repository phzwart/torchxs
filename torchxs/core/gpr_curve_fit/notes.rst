Gaussian Process Regression Curve Fitting
-----------------------------------------

Assume we have a signal log[I(q)] that we want to approximate with a Gaussian Process.

Because we want to be able to extrapolate estimates beyond observed data ranges and
include some sort of awareness of reality, we will inject a physics-based prior mean
function into the GP. One way is to build a mean function that is a weighted mixture
of known shapes, like Sphere with different radii::

    mu(q) = log[ sum_{j=1}^{N} w_j sphere_j(q | radius_j) ]

where sphere_j(q | radius_j) is the theoretical scattering curve for a sphere with known
diameter. Because the intensity is always positive, and we use Gaussian Process, we will
fit the log-intensity instead of the intensity.

This defines the mean function, but we
still need to take care of the covariance kernel. One thing that is obvious from the
scattering factors of certain shapes, is that using a stationary kernel is not likely to
be the best choice.








