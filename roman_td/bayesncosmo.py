import os
import numpy as np
import matplotlib.pyplot as plt
import bayesn
import sncosmo
import extinction as ext_lib
from scipy.stats import norm
from scipy.linalg import solve_banded, solve_triangular
from scipy.interpolate import interp1d, RegularGridInterpolator
from astropy.config import get_cache_dir
from sncosmo import Source
from sncosmo.utils import download_file
from ruamel.yaml import YAML

class BAYESNSource(Source):
    """
    The BayeSN Type Ia supernova spectral time series model.

    Standalone sncosmo implementation.

    The free parameters of the model are ``amplitude`` and ``theta``.
    The ``amplitude`` is a multiplicative scaling of the SED. Setting
    ``amplitude = 10**(-0.4*mu)`` where ``mu`` is a distance modulus
    will reproduce the BayeSN model.

    Parameters
    ----------
    model_name : str
        Name/path of a YAML file defining a model.
        For the default options, it will attempt to download
        a YAML file from github.com/bayesn/bayesn-model-files.
        Options: ``'M20'``, ``'T21'``, ``'W22'``, ``'W22x'``.
        Default: ``'T21'``.
    use_epsilon : bool, optional
        If ``True``, includes a random realization from BayeSN's
        residual scatter model upon initialization. Default: ``False``.
    fit_epsilon : bool, optional
        If ``True``, includes epsilon as fittable named
        parameters (``'e_1_1', 'e_2_1', ...``). Default: ``False``.
    """
    def __init__(self, model_name='T21',
                 use_epsilon=False,
                 fit_epsilon=False,
                 name=None, version=None):

        self.name = name
        self.version = version

        self._param_names = ['amplitude', 'theta']
        self.param_names_latex = ['10^{-0.4(\\mu + \\delta M}', '\\theta_1']

        # use/cache version of public models
        if model_name in ['M20', 'T21', 'W22', 'W22x']:
            model_path = os.path.join(get_cache_dir(),
                 'sncosmo', 'models', 'bayesn')
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            model_path = os.path.join(model_path, model_name + '.YAML')
            # attempt to download the model file
            if not os.path.exists(model_path):
                url = ("https://github.com/bayesn/bayesn-model-files/" +
                     "raw/refs/heads/main/" +
                     "BAYESN.{}/BAYESN.YAML".format(model_name))
                download_file(url, model_path)
        # otherwise try to find the model locally
        else:
            if os.path.exists(model_name):
                model_path = model_name
            else:
                raise ValueError(model_name + 
                    " not recognised as a valid BayeSN model name or path!")
        
        # load the YAML
        with open(model_path, "r") as f:
            with YAML(typ='safe') as yaml:
                _model_dict = yaml.load(f)
        self._phase = np.array(_model_dict["TAU_KNOTS"])
        self._wave = np.array(_model_dict["L_KNOTS"])
        self._L_Sigma = np.array(_model_dict["L_SIGMA_EPSILON"])
        self._W0 = np.array(_model_dict["W0"])
        self._W1 = np.array(_model_dict["W1"])
        self._M0 = _model_dict["M0"]
        self._S0 = sncosmo.Model(source='hsiao')

        if fit_epsilon is True:
            self._param_names += ["e_{}_{}".format(i+1,j)
                for i in range(len(self._wave)-2)
                for j in range(len(self._phase))]
            self.param_names_latex += ["\\epsilon_{{{},{}}}".format(i+1,j)
                for i in range(len(self._wave)-2)
                for j in range(len(self._phase))]
            # default parameters
            self._parameters = np.array([1.0, 0.0] + [0.0]*len(self._L_Sigma))
            self._epsilon_free = True
        else:
            # default parameters
            self._parameters = np.array([1.0, 0.0])
            self._epsilon_free = False

        # set epsilon
        if use_epsilon:
            self._epsilon_nz = self._sample_epsilon(flat=True)
        else:
            self._epsilon_nz = np.zeros(len(self._L_Sigma))

        # construct K^{-1}D matrices for getting spline 2nd derivatives
        self._invKD_phase = self._construct_invKD(self._phase)
        self._invKD_wave = self._construct_invKD(self._wave)

    # resdiual scatter vector (non-zero elements of epsilon)
    @property
    def _epsilon_nz(self):
        if self._epsilon_free:
            return self._parameters[2:]
        else:
            return self._e
    @_epsilon_nz.setter
    def _epsilon_nz(self, e):
        if self._epsilon_free:
            self._parameters[2:] = e
        else:
            self._e = e

    # resdiual scatter matrix (including zero-elements)
    @property
    def _epsilon(self):
        if np.all(self._epsilon_nz == 0.0):
            return 0.0
        else:
            e = np.zeros(self._W0.shape)
            e[1:-1] = self._epsilon_nz.reshape(e[1:-1].shape, order="F")
            return e   
    @_epsilon.setter
    def _epsilon(self, e):
        self._epsilon_nz = e[1:-1].flatten(order='F')

    def maxwave(self):
        # Add 0.5% buffer so sncosmo accepts bandpasses (e.g. F444W) whose
        # red edge falls just beyond the last wavelength knot after redshifting.
        return self._wave[-1] * 1.005

    def _construct_invKD(self, x):
        """
        Constructs a matrix from spline knot locations to enable
        computation of second derivatives.

        Parameters
        ----------
        x : numpy.array
            Spline knot locations.

        Returns
        -------
        invKD : numpy.array
            Matrix for computing 2nd derivatives.
        """
        N = len(x)
        K = np.zeros((3, N-2))
        D = np.zeros((N-2, N))
        for i in range(N-2):
            K[1,i] = (x[i+2] - x[i])/3.0
            if i < N-3:
                K[0,i+1] = K[2,i] = (x[i+2] - x[i+1])/6.0
            D[i,i] = 1.0/(x[i+1] - x[i])
            D[i,i+1] = - 1.0/(x[i+1] - x[i]) - 1.0/(x[i+2] - x[i+1])
            D[i,i+2] = 1.0/(x[i+2] - x[i+1])
        invKD = np.zeros((N,N))
        invKD[1:-1] = solve_banded((1,1), K, D, 
            overwrite_ab=True, overwrite_b=True)
        return invKD

    def _construct_J(self, x, x_knots, invKD):
        """
        Constructs a matrix that interpolates a set of spline knots.

        Parameters
        ----------
        x : numpy.array
            Desired interpolation points.
        x_knots : numpy.array
            Spline knot locations.
        invKD : numpy.array
            K^{-1}D matrix.

        Returns
        -------
        J : numpy.array
            Spline coefficient matrix.
        """
        x = np.atleast_1d(x)
        p = np.arange(len(x)) # indices
        q = np.zeros(len(x), dtype=int) # knot indices
        d = np.zeros(len(x))
        l = x < x_knots[0] # mask for left-side extrapolation
        r = x >= x_knots[-1] # mask for right-side extrapolation
        m = (~l)*(~r) # mask for interpolation
        q[r] = -2
        q[m] = np.searchsorted(x_knots, x[m], side="right")-1
        h = x_knots[q+1] - x_knots[q]
        a = (x_knots[q+1] - x)/h
        b = 1.0 - a
        c = h*h/6.0
        d[m] = (b[m]**3 - b[m])*c[m]
        c[m] = (a[m]**3 - a[m])*c[m]
        c[l] = -b[l]*c[l]
        c[r] = -a[r]*c[r]
        J = c[:,None]*invKD[q] + d[:,None]*invKD[q+1]
        J[p,q] = J[p,q] + a
        J[p,q+1] = J[p,q+1] + b
        return J

    def _flux(self, phase, wave):
        """
        Model flux.

        Parameters
        ----------
        phase : float or numpy.array
            Rest-frame phase in days.
        wave : float or numpy.array
            Rest-frame wavelength in Å.

        Returns
        -------
        flux : numpy.array
            SED evaluated at ``phase`` and ``wave``.
        """
        J_wave = self._construct_J(wave, self._wave, self._invKD_wave)
        J_phase = self._construct_J(phase, self._phase, self._invKD_phase)
        W = self._W0 + self._parameters[1]*self._W1 + self._epsilon
        W = (J_wave @ W @ J_phase.T).T

        S = self._S0._flux(phase, wave)*10**(-0.4*W)

        scale = self._parameters[0]*10**(-0.4*self._M0)
        result = S * scale
        if not np.all(np.isfinite(result)):
            result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result

    def _ln_prior_theta(self):
        """
        Returns the prior probability of the current ``'theta'``.
        
        Returns
        -------
        lnp : float
            Natural logarithm of prior probability.
        """
        return - 0.5*np.log(2.0*np.pi) - 0.5*(self._parameters[1])**2

    def _ln_prior_epsilon(self):
        """
        Returns the prior probability of the current epsilon.
        
        Returns
        -------
        lnp : float
            Natural logarithm of prior probability.
        """
        twopin = len(self._L_Sigma)*np.log(2.0*np.pi)
        logdet = 2.0*np.sum(np.log(np.diagonal(self._L_Sigma)))
        invLe  = solve_triangular(self._L_Sigma, self._epsilon_nz, lower=True)
        return -0.5*(twopin + logdet + np.dot(invLe, invLe))

    def _ptform_theta(self, u):
        """
        Transforms a uniform random variable to theta.
        For use with nested samplers.

        Parameters
        ----------
        u : float or numpy.array
            Uniform random variable.

        Returns
        -------
        theta : float or numpy.array
            Transform of ``u``.
        """
        return norm.ppf(u)

    def _ptform_epsilon(self, u, flat=False):
        """
        Transform uniform random variables to epsilon.
        For use with nested samplers.

        Parameters
        ----------
        u : numpy.array
            Uniform random vector of length ``len(self._L_Sigma)``.
        flat : bool, optional
            If ``True``, returns a flattened vector containing only the
            non-zero elements of epsilon. If ``False``, returns the full
            epsilon matrix. Default: ``False``.

        Returns
        -------
        epsilon : numpy.array
            Transform of ``u``.
        """
        nu = norm.ppf(u)
        if flat is True:
            return self._L_Sigma @ nu
        else:
            epsilon = np.zeros(self._W0.shape)
            epsilon[1:-1] = (self._L_Sigma @ nu).reshape(
                epsilon[1:-1].shape, order="F")
            return epsilon

    def _sample_epsilon(self, flat=False):
        """
        Samples a residual scatter realisation.

        Parameters
        ----------
        flat : bool, optional
            If ``True``, returns a flattened vector containing only the
            non-zero elements of epsilon. If ``False``, returns the full
            epsilon matrix. Default: ``False``.

        Returns
        -------
        epsilon : numpy.array
            Residual scatter realisation.
        """
        nu = np.random.normal(0, 1, len(self._L_Sigma))
        if flat is True:
            return self._L_Sigma @ nu
        else:
            epsilon = np.zeros(self._W0.shape)
            epsilon[1:-1] = (self._L_Sigma @ nu).reshape(
                epsilon[1:-1].shape, order="F")
            return epsilon

    def _update_epsilon(self):
        """
        Updates the stored residual scatter realisation.
        """
        self._epsilon_nz = self._sample_epsilon(flat=True)

class FixedRVDust(sncosmo.PropagationEffect):
    """Fitzpatrick (1999) dust with fixed R_V, using SNTD's 3-arg propagate convention.

    SNTD's _mlFlux calls effect.propagate(phase, wave, flux).  sncosmo's
    built-in F99Dust has propagate(wave, flux, phase=None), so SNTD passes
    phase values as 'wave', causing fitzpatrick99(wave=0) → NaN when any
    observation coincides with t0.  This class accepts the 3-arg SNTD
    signature and maps the arguments correctly.
    """

    _param_names = ['ebv']
    param_names_latex = ['E(B-V)']

    def __init__(self, r_v=3.1):
        self._r_v = r_v
        self._parameters = np.array([0.0])
        self._minwave = 910.0
        self._maxwave = 60000.0

    def propagate(self, phase, wave, flux):
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        ebv = self._parameters[0]
        if ebv <= 0.0:
            return flux
        a_v = self._r_v * ebv
        mag_ext = ext_lib.fitzpatrick99(
            np.asarray(wave, dtype=np.float64), a_v, self._r_v
        )
        return flux * 10.0 ** (-0.4 * mag_ext)


class FreeRVDust(sncosmo.PropagationEffect):
    """Fitzpatrick (1999) dust with a free R_V parameter.

    When added to a sncosmo Model as effect name 'host', the parameters
    appear as 'hostebv' and 'hostr_v' in model.param_names so SNTD's
    fitting machinery can sample them normally.
    """

    _param_names = ['ebv', 'r_v']
    param_names_latex = ['E(B-V)', 'R_V']

    def __init__(self):
        self._parameters = np.array([0.0, 3.1])
        self._minwave = 910.0    # F99 valid range lower bound (Å)
        self._maxwave = 60000.0  # F99 valid range upper bound (Å)

    def propagate(self, phase, wave, flux):
        flux = np.nan_to_num(flux, nan=0.0, posinf=0.0, neginf=0.0)
        ebv = self._parameters[0]
        r_v = self._parameters[1]
        if ebv <= 0.0:
            return flux
        a_v = r_v * ebv
        mag_ext = ext_lib.fitzpatrick99(
            np.asarray(wave, dtype=np.float64), a_v, r_v
        )
        return flux * 10.0 ** (-0.4 * mag_ext)


class NumPyroBAYESNSource(Source):
    """
    The BayeSN Type Ia supernova spectral time series model.

    Wrapper for the bayesn NumPyro model.

    The free parameters of the model are ``amplitude`` and ``theta``.
    The ``amplitude`` is a multiplicative scaling of the SED. Setting
    ``amplitude = 10**(-0.4*mu)`` where ``mu`` is a distance modulus
    will reproduce the BayeSN model.

    Parameters
    ----------
    model_name : str, optional
        BayeSN model name or path to a YAML file defining a model.
        Choices: ``'M20_model'``, ``'T21_model'``, ``'W22_model'``.
        Default: ``'T21_model'``.
    use_epsilon : bool, optional
        If `True`, includes a residual realization from BayeSN's
        residual scatter model. Default: `False`.
    interp_kind : str, optional
        Interpolation method used by ``scipy.interpolate.interp1d``
        to interpolate model spectra onto the required wavelengths.
        Default: ``'linear'``.

    Attributes
    ----------
    _model : bayesn.SEDmodel
        BayeSN model object.
    _parameters : numpy.array
        Source parameters: ``[amplitude, theta]``.
    _epsilon : numpy.array
        Residual scatter realization.
    _interp_kind : str
        Spectrum interpolation method.
    _phase : numpy.array
        Rest-frame phase knots of the BayeSN model.
    _wave : numpy.array
        Rest-frame wavelength knots of the BayeSN model.
    """
    _param_names = ['amplitude', 'theta']
    param_names_latex = ['10^{-0.4(\\mu + \\delta M}', '\\theta_1']

    def __init__(self, model_name='T21_model',
                 use_epsilon=False,
                 interp_kind='linear',
                 name=None, version=None):

        self.name = name
        self.version = version

        from bayesn import SEDmodel
        
        self._model = SEDmodel(load_model=model_name)
        self._parameters = np.array([1.0, 0.0])
        if use_epsilon:
            self._epsilon = self._model.sample_epsilon(1)
        else:
            self._epsilon = 0
        self._interp_kind = interp_kind
        self._phase = self._model.tau_knots
        self._wave = self._model.l_knots

    def _flux(self, phase, wave):
        """
        Model flux at some phase and rest wavelength.

        Calls ``bayesn.SEDmodel.simulate_spectrum``.
        """
        phase = np.atleast_1d(phase)
        wave = np.atleast_1d(wave)

        l, f, _ = self._model.simulate_spectrum(phase, 1, z=0, AV=0, del_M=0,
            eps=self._epsilon, theta = self._parameters[1])
        
        spec = interp1d(l[0], f[0], kind=self._interp_kind, copy=False, axis=0)
        scale = self._parameters[0]*10**(-0.4*self._model.M0)

        return scale*spec(wave).T

    def _update_epsilon(self, epsilon=None):
        """
        Updates the stored residual scatter realisation.

        Parameters
        ----------
        epsilon : numpy.array, optional
            If provided, sets ``self._epsilon=epsilon``.
            If ``None``, samples a new ``self._epsilon`` from the prior.
        """
        if epsilon is None:
            self._epsilon = self._model.sample_epsilon(1)
        else:
            self._epsilon = epsilon
