######################
# ConvolvedModel API #
######################

from astropy.modeling import ConvolvedModel
from astropy.modeling.models import Box2D, Lorentz1D, Gaussian2D
from astropy.convolution import Gaussian2DKernel, Gaussian1DKernel

# for simplicity we only support pixel coordinates and assume a fixed psf model.
# the convolved model should be initialized with a model and a kernel instance
# like:
source = Box2D(1, 0, 0, 1, 1)
psf = Gaussian2DKernel(1)
convolved_model = ConvolvedModel(source, psf)

# support of 1D models
source = Lorentz1D(1, 0, 0, 1, 1)
psf = Gaussian1DKernel(1)
convolved_model = ConvolvedModel(source, psf)

# should there be a Convolved1DModel and Convolved2DModel?

# later we can think about using any model as convolution kernels
# and even implement the analytical solution for the common Gaussian case
# or other cases where analytical solutions are known
source = Gaussian2D(1, 0, 0, 1, 1)
psf = Gaussian2D(1, 0, 0, 1, 1)
convolved_model = ConvolvedModel(source, psf)

# ConvolvedModel should inherit from FittableModel but overrides the evaluate method
# to perfom an additional numerical convolution
model_data = model(x, y)

# fitting works the same way as for FittableModel
# the parameters of the psf model are fixed during the fit.


# the convolution method can be set via string, to choose between the astropy.convolution
# methods
convolved_model = ConvolvedModel(source, psf, convolution='fft')
convolved_model = ConvolvedModel(source, psf, convolution='standard')

# or by passing a function that should be used. This is mainly nececessary, because
# the performance of the astropy convolution algorithms is rather weak...
convolved_model = ConvolvedModel(source, psf, convolution=scipy.signal.fftconvolve)

# recommendation would be to use scipy.signal.fftconvolve which is very fast

# additional keyword arguments can be passed to the convolution function
convolved_model = ConvolvedModel(source, psf, convolution='standard', boundary='none')

# Further notes:
# 1. Implementation of this class shpould be straight forward and not much effort
# 2. Testing could be set up using Gaussian models, where the analytical solution is
# known
# 3. Later one could think about assigning the corresponding fourier transforms to analytical
# models and use this for the convolution, which will probably have a better performance.
# Source model and PSF are evaluated analytically in Fourier space, mutliplied and transformed
# back to normal space.
# 4. As convolution is a linear operation the derivative of a convolution behaves like 
# http://en.wikipedia.org/wiki/Convolution#Differentiation this relation can be used
# to implement `fit_deriv` methods for convolved models.



#########################
# Model integration API #
#########################

import numpy as np

# to avoid bias, when the scale of the model is similar to the size of the pixels,
# models have to be integrated over pixels, for this purpose models could define
# an integrate function, that takes upper and lower bounds of the integration
# (see also https://github.com/astropy/astropy/issues/1586)
model = Gaussian1D(1, 0, 1)
model.integrate(0, np.inf)

# it should also work with an arrays of bounds of course
x_lo = np.arange(-10, 11) - 0.5
x_high = np.arange(-10, 11) + 0.5
model.integrate(x_lo, x_hi)

# whenever possible (e.g. polynomials) the integrate function would be implemented
# analytically, for all others a standard numerical integration routine would be used
# e.g. scipy.

# for convenience one could incorporate the functionality in the standard model API
model = Gaussian1D(1, 0, 1, eval_method='integrate')
model = Gaussian1D(1, 0, 1, eval_method='oversample')
model = Gaussian1D(1, 0, 1, eval_method='center')

# where the given evaluation method would be used when the model is called
# for evaluation
x = np.arange(-10, 11)
model(x)

# models that are spatially confined to a certain region such as Gaussian2D, Disk2D,
# Delta2D, Box2D, ... should have the possibility to apply a different normalization,
# where the amplitude parameter corresponds to the integral of the model and not the
# peak value at maximum (see e.g.: https://gammapy.readthedocs.org/en/latest/api/gammapy.morphology.Shell2D.html#gammapy.morphology.Shell2D)
# This is very useful for modeling sources, because the amplitude parameter then
# corresponds to the total flux of the source...
model = Gaussian2D(1, 0, 0, 1, 1, normalization='peak')
model = Gaussian2D(1, 0, 0, 1, 1, normalization='integral')

# alternatively one could introduce new NormGaussian2D, NormBox2D, ... models, but that
# would probably be a lot of code duplication?

# all models that are 'normalisable' should have an `extent` attribute that specifies
# a rectangular region where the model is nonzero or has 99.99% containment (or a similar
# criterion). A reasonable value would choosen as default and would be given as multiples of the
# models size parameters. Model evaluation can later be limited to the region defined
# by `extent` for better performance 
Gaussian2D.extent= dict(x=8 * width, y=8 * width)
Disk2D.extent = dict(x=radius, y=radius) 

# models should have an attribute that defines a reasonable sample size for the model,
# warnings could be raised, when models are undersampled
Gaussian2D.sample_size = width / 5.


###############################
# Model Image Computation API #
###############################

# functionality is added to NDData via NDDataMixin classes
class SkyImage(NDData, NDSlicingMixin, NDWCSCutoutMixin) : pass

# a sky image object is created as following
sky_image = SkyImage(np.zeros(501, 10001), wcs=WCS())

# it has a coordinates attribute, that returns a SkyCoord object
# with every pixel containing the corresponding position
sky_image.coordinates

# one can make rectangular cutouts of the sky image using
cutout = sky_image.extract(position, source.extent)
cutout = sky_image.extract(position, source.extent, copy=True)

# which returns again a SkyImage object, but with modified WCS transform 

# to render model on the sky a new function should be defined
source_image = render_model_to_sky(source, wcs=WCS())

# which would return again a (smaller) SkyImage object centered on the
# nearest pixel of the source position which can be added to sky_image
# with

sky_image.add(source_image)

# later `.add` could use any kind of resampling/reprojection
# for now we just assume that `sky_image` and `source_image`
# pixels are aligned (which should be checked...)
