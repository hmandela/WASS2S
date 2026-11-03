import xarray as xr
import numpy as np
import inspect
from wass2s.was_eof import WAS_EOF
from wass2s.was_scientific_utils import align_predictor_target
from wass2s.was_probability_calibration import normalize_probability_calibration


class WAS_PCR:
    """Spatial principal-component regression wrapping a WAS regression model.

    This class implements a two-stage pipeline:

    1. **EOF dimensionality reduction** — the spatial predictor field(s) are
       decomposed into principal components (PCs) by :class:`~wass2s.was_eof.WAS_EOF`,
       which handles NaN filling, optional linear detrending, and optional
       per-field variance normalization entirely within each cross-validation
       fold (no data leakage).
    2. **Regression** — the resulting PCs are passed to any compatible WAS
       regression model (e.g. :class:`~wass2s.was_linear_models.WAS_MLR`,
       :class:`~wass2s.was_machine_learning.WAS_RandomForest`, …).

    Single-field and multivariate predictors
    -----------------------------------------
    ``Predictor`` may be a single :class:`xarray.DataArray` *or* a **list** of
    DataArrays representing different variables and/or grids (e.g. SST + SLP +
    rainfall indices).  A list triggers a **combined multivariate EOF**: each
    field is preprocessed and normalized independently on the training fold so
    that no single field dominates the joint covariance, then
    :class:`~wass2s.was_eof.WAS_EOF` builds one shared set of PCs that feed
    the regression.

    .. note::
        When using this class inside :class:`~wass2s.was_cross_validate.WAS_Cross_Validator`,
        pass the **raw** predictor (do *not* pre-detrend outside the fold loop).
        For a list predictor, slice every field per fold::

            X_train = [p.isel(T=train_idx) for p in Predictor]
            X_test  = [p.isel(T=test_idx)  for p in Predictor]

    Parameters
    ----------
    regression_model : object
        An instantiated WAS regression model exposing at least
        ``compute_model(X_train, y_train, X_test, ...)`` and, optionally,
        ``compute_prob(...)`` and ``forecast(...)``.
    n_modes : int, optional
        Number of EOF modes to retain.  Ignored when
        ``opti_explained_variance`` is set.
    use_coslat : bool, default True
        Apply cosine-latitude area weighting in the EOF decomposition.
    standardize : bool, default False
        Standardize each grid point before the EOF decomposition (useful
        when the predictor is not expressed as anomalies).
    detrend : bool, default True
        Remove a linear trend from each grid point before the EOF
        decomposition.  The trend is estimated on the training fold and
        applied to the test fold to avoid leakage.
    opti_explained_variance : float, optional
        Target cumulative explained variance (%) used to select the optimal
        number of modes automatically (e.g. ``90.0`` for 90 %).
    L2norm : bool, default False
        Return L2-normalised EOF scores and components from
        :class:`~wass2s.was_eof.WAS_EOF`.
    normalize : bool or None, default None
        Per-field scalar rescaling so that no field dominates the joint
        covariance in a multivariate EOF.  ``None`` means *auto*: enabled
        for a list of fields, disabled for a single field.

    probability_calibration : {"outer", "nested", "inner"}, default "outer"
        Residual calibration used by the cross-validator. ``inner`` aliases
        ``nested``; model selection remains controlled separately.

    Attributes
    ----------
    eof_model : :class:`~wass2s.was_eof.WAS_EOF` or None
        The fitted EOF object, available after the first call to
        :meth:`compute_model` or :meth:`forecast`.

    Examples
    --------
    Single-field PCR with a Ridge regression back-end:

    >>> from wass2s import WAS_PCR, WAS_Ridge_Model
    >>> pcr = WAS_PCR(WAS_Ridge_Model(alpha=1.0), n_modes=10, detrend=True)
    >>> result = pcr.compute_model(X_train, y_train, X_test)

    Multivariate PCR (SST + SLP):

    >>> pcr_mv = WAS_PCR(WAS_MLR(), opti_explained_variance=85.0)
    >>> result  = pcr_mv.compute_model([sst_train, slp_train], y_train,
    ...                                 [sst_test,  slp_test])
    """

    def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
                 detrend=True, opti_explained_variance=None, L2norm=False, normalize=None,
                 probability_calibration="outer"):
        self.reg_model = regression_model
        self.probability_calibration = normalize_probability_calibration(probability_calibration)
        self.n_modes = n_modes
        self.use_coslat = use_coslat
        self.standardize = standardize
        self.detrend = detrend
        self.opti_explained_variance = opti_explained_variance
        self.L2norm = L2norm
        self.normalize = normalize
        self.eof_model = None

    @staticmethod
    def _ensure_T(x):
        """Guarantee the presence of a ``T`` dimension.

        Parameters
        ----------
        x : xarray.DataArray or list of xarray.DataArray
            Input field(s).

        Returns
        -------
        xarray.DataArray or list of xarray.DataArray
            Same as input, with a ``T`` dimension added where missing.
        """
        if isinstance(x, (list, tuple)):
            return [d if "T" in d.dims else d.expand_dims("T") for d in x]
        return x if "T" in x.dims else x.expand_dims("T")

    def _prepare_pcs(self, X_train, X_test):
        """Fit EOF on the training fold and project both folds onto the PC space.

        The EOF model is stored in :attr:`eof_model` so that spatial patterns
        can be inspected after fitting.

        Parameters
        ----------
        X_train : xarray.DataArray or list of xarray.DataArray
            Training predictor field(s) with a ``T`` dimension.
        X_test : xarray.DataArray or list of xarray.DataArray
            Test predictor field(s).  May lack a ``T`` dimension for a
            single-year forecast step (it will be added automatically).

        Returns
        -------
        X_train_pcs : xarray.DataArray
            Training PCs with dimensions ``(T, features)``.
        X_test_pcs : xarray.DataArray
            Test PCs with dimensions ``(T, features)``.
        """
        X_test = self._ensure_T(X_test)
        test_fields = X_test if isinstance(X_test, (list, tuple)) else [X_test]
        available = None
        for field in test_fields:
            observed = np.isfinite(field).any([d for d in field.dims if d != "T"])
            available = observed if available is None else available | observed
        eof = WAS_EOF(n_modes=self.n_modes, use_coslat=self.use_coslat,
                      standardize=self.standardize, detrend=self.detrend,
                      opti_explained_variance=self.opti_explained_variance,
                      L2norm=self.L2norm, normalize=self.normalize)
        self.eof_model = eof
        from wass2s.was_eof import EOFRankError
        try:
            _, s_train, _ = eof.fit(X_train, dim="T")
        except EOFRankError:
            # A zero-rank field contains no usable predictor variation. A
            # constant feature lets supported regression backends estimate
            # their intercept without amplifying null singular vectors.
            train = X_train[0] if isinstance(X_train, (list, tuple)) else X_train
            test = self._ensure_T(X_test)
            test = test[0] if isinstance(test, (list, tuple)) else test
            self.rank_zero_fallback_ = True
            def constant_feature(time):
                return xr.DataArray(np.zeros((len(time), 1)), dims=("T", "features"),
                                    coords={"T": time, "features": [0]})
            return constant_feature(train["T"]), constant_feature(test["T"]).where(available)
        self.rank_zero_fallback_ = False

        X_train_pcs = s_train.rename({"mode": "features"}).transpose("T", "features")
        s_test = eof.transform(self._ensure_T(X_test), dim="T")
        X_test_pcs = s_test.rename({"mode": "features"}).where(available).transpose("T", "features")
        return X_train_pcs, X_test_pcs

    def compute_model(self, X_train, y_train, X_test, y_test=None, **kwargs):
        """Run the full PCR pipeline on a single train/test split.

        Performs EOF dimensionality reduction on the predictor, then delegates
        to the wrapped regression model.

        Parameters
        ----------
        X_train : xarray.DataArray or list of xarray.DataArray
            Training predictor field(s).
        y_train : xarray.DataArray
            Training target (predictand).
        X_test : xarray.DataArray or list of xarray.DataArray
            Test predictor field(s).
        y_test : xarray.DataArray, optional
            Test target.  Forwarded to the regression model only when its
            ``compute_model`` signature accepts ``y_test``.
        **kwargs
            Additional keyword arguments forwarded to the regression model.

        Returns
        -------
        object
            Whatever the wrapped regression model's ``compute_model`` returns
            (typically a deterministic hindcast DataArray).
        """
        X_train = align_predictor_target(X_train, y_train, mode="season_year")
        X_train_pcs, X_test_pcs = self._prepare_pcs(X_train, X_test)
        parameters = inspect.signature(self.reg_model.compute_model).parameters
        if y_test is not None and "y_test" in parameters:
            return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test=y_test, **kwargs)
        return self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, **kwargs)

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det, **kwargs):
        """Compute tercile probability forecasts from a deterministic hindcast.

        Delegates directly to the wrapped regression model's ``compute_prob``
        method.  Returns ``None`` if the model does not implement it.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand over the full hindcast period.
        clim_year_start : int
            First year of the climatological reference period.
        clim_year_end : int
            Last year of the climatological reference period.
        hindcast_det : xarray.DataArray
            Deterministic hindcast produced by :meth:`compute_model`.
        **kwargs
            Additional keyword arguments forwarded to the regression model.

        Returns
        -------
        xarray.DataArray or None
            Tercile probabilities (below-normal, near-normal, above-normal),
            or ``None`` if the regression model does not support probabilistic
            output.
        """
        if not hasattr(self.reg_model, "compute_prob"):
            return None
        signature = inspect.signature(self.reg_model.compute_prob)
        accepts_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        params_prob = {k: v for k, v in kwargs.items()
                       if accepts_kwargs or k in signature.parameters}
        return self.reg_model.compute_prob(Predictant, clim_year_start, clim_year_end,
                                           hindcast_det, **params_prob)

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor,
                 hindcast_det, Predictor_for_year, target_times=None, **kwargs):
        """Generate an operational forecast for a target year.

        EOF patterns are fitted on the full hindcast predictor, and the
        target-year predictor is projected onto those patterns before being
        passed to the regression model.

        Parameters
        ----------
        Predictant : xarray.DataArray
            Observed predictand over the full hindcast period.
        clim_year_start : int
            First year of the climatological reference period.
        clim_year_end : int
            Last year of the climatological reference period.
        Predictor : xarray.DataArray or list of xarray.DataArray
            Full hindcast predictor field(s) used to fit the EOF model.
        hindcast_det : xarray.DataArray
            Deterministic hindcast produced during cross-validation.
        Predictor_for_year : xarray.DataArray or list of xarray.DataArray
            Predictor field(s) labelled with target valid times. Lag availability
            dates must be retained separately from this verification calendar.
        target_times : array-like, optional
            Explicit verification dates to assign BEFORE EOF projection and
            detrending, useful when the source predictor is labelled by its lag
            month. No cross-year shift is inferred from months alone.
        **kwargs
            Additional keyword arguments forwarded to the regression model.

        Returns
        -------
        object
            Operational forecast returned by the wrapped regression model's
            ``forecast`` method.
        """
        Predictor = align_predictor_target(Predictor, Predictant, mode="season_year")
        if target_times is not None:
            times = np.atleast_1d(np.asarray(target_times))
            def label_target(field):
                field = self._ensure_T(field)
                if field.sizes["T"] != len(times):
                    raise ValueError("target_times must match the forecast predictor's T length")
                return field.assign_coords(T=times)
            Predictor_for_year = ([label_target(field) for field in Predictor_for_year]
                if isinstance(Predictor_for_year, (list, tuple)) else label_target(Predictor_for_year))
        # A seasonal lag month cannot serve as the target valid time: it would
        # also change the extrapolated training trend before regression. Ask
        # for explicit target dates rather than guessing a cross-year offset.
        if np.issubdtype(Predictant["T"].dtype, np.datetime64):
            target_months = np.unique(Predictant["T"].dt.month.values)
            future_fields = (Predictor_for_year if isinstance(Predictor_for_year, (list, tuple))
                             else [Predictor_for_year])
            for field in future_fields:
                if np.issubdtype(field["T"].dtype, np.datetime64):
                    if len(target_months) == 1 and np.any(field["T"].dt.month.values != target_months[0]):
                        raise ValueError("PCR forecast T must label the target season; supply target_times explicitly for lagged predictors")
        Predictor_pcs, Predictor_year_pcs = self._prepare_pcs(Predictor, Predictor_for_year)
        result = self.reg_model.forecast(
            Predictant, clim_year_start, clim_year_end,
            Predictor_pcs, hindcast_det, Predictor_year_pcs, **kwargs)
        # The backends share the same conditional probability kernel as CV.
        # Preserve their declared reference and metadata without recalibrating
        # a second time in the wrapper.
        return result


# Keep constructor inputs for fresh model instances in nested cross-validation.
from .was_crossfit_adapters import capture_constructors as _capture_constructors
_capture_constructors(globals())
