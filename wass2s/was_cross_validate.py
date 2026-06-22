"""Leakage-free cross-validation for the WASS2S pipeline.

Provides a leave-one-out splitter with a configurable exclusion window and
a high-level orchestrator that dispatches each fold to the correct
preprocessing path for every supported model family.

Classes
-------
CustomTimeSeriesSplit
    Symmetric-window leave-one-out splitter; prevents near-neighbour
    temporal leakage.
WAS_Cross_Validator
    Orchestrates cross-validation for CCA, Kernel CCA, PCR, linear models,
    machine-learning models, MME methods, and analog ensembles.

Helper functions
----------------
_pcr_isel_T
    ``isel(T=idx)`` for a DataArray or list/tuple of DataArrays.
_pcr_T_index
    Return the ``T`` coordinate used to drive the splitter.
"""
from tqdm.auto import tqdm
from wass2s.was_linear_models import *
from wass2s.was_eof import *
from wass2s.was_pcr import *
from wass2s.was_cca import *
from wass2s.was_cca_2 import *
from wass2s.was_machine_learning import *
from wass2s.was_analog import *
from wass2s.utils import *
from wass2s.was_transformdata import *
from wass2s.was_mme import *


def _pcr_isel_T(P, idx):
    """isel(T=idx) for a DataArray or a list/tuple of DataArrays."""
    if isinstance(P, (list, tuple)):
        return [p.isel(T=idx) for p in P]
    return P.isel(T=idx)


def _pcr_T_index(P):
    """The T coordinate used to drive the splitter (first field if a list)."""
    return (P[0] if isinstance(P, (list, tuple)) else P)["T"]


class CustomTimeSeriesSplit:
    """
    Custom time series cross-validator for splitting data into training and test sets.

    Ensures temporal ordering is maintained by generating training and test indices
    suitable for time series data, with an option to omit samples after the test index.

    Parameters
    ----------
    n_splits : int
        Number of splits for the cross-validation.
    """

    def __init__(self, n_splits):
        self.n_splits = n_splits

    def split(self, X, nb_omit, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.

        Yields training indices before the test index (excluding a specified number of samples
        after the test index) and test indices for each split.

        Parameters
        ----------
        X : array-like
            The data to be split, typically time series data.
        nb_omit : int
            Number of samples to omit from training after the test index to avoid data leakage.
        y : array-like, optional
            The target variable (ignored in this implementation).
        groups : array-like, optional
            Group labels for the samples (ignored in this implementation).

        Yields
        ------
        train_indices : ndarray
            The training set indices for the current split.
        test_indices : list
            The test set indices for the current split.
        """

        n_samples = len(X)
        indices = np.arange(n_samples)
        n_folds = min(self.n_splits, n_samples)

        window_len = nb_omit + 1  # omitted neighbors + test year
        if window_len > n_samples:
            raise ValueError("nb_omit too large for number of samples")

        for i in range(n_folds):
            # 1) Start with window centered on i
            start = i - window_len // 2
            if start < 0:
                start = 0
            end = start + window_len
            if end > n_samples:
                end = n_samples
                start = end - window_len

            # This window has ALWAYS length = window_len
            window_indices = np.arange(start, end)

            # 2) Training indices = all indices outside the window
            train_indices = np.setdiff1d(indices, window_indices)

            # 3) Test index = i
            test_indices = np.array([i], dtype=int)

            yield train_indices, test_indices
    def get_n_splits(self, X=None, y=None, groups=None):
        """
        Return the number of splits for the cross-validation.

        Parameters
        ----------
        X : array-like, optional
            The data to be split (ignored in this implementation).
        y : array-like, optional
            The target variable (ignored in this implementation).
        groups : array-like, optional
            Group labels for the samples (ignored in this implementation).

        Returns
        -------
        int
            The number of splits configured for the cross-validator.
        """
        return self.n_splits


class WAS_Cross_Validator:
    """
    Performs cross-validation for time series forecasting models using a custom time series split.

    This class wraps a custom time series cross-validator to evaluate forecasting models,
    handling both deterministic hindcasts and probabilistic (tercile) predictions.

    Parameters
    ----------
    n_splits : int
        Number of splits for the cross-validation.
    nb_omit : int
        Number of samples to omit from training after the test index to prevent data leakage.
    """

    def __init__(self, n_splits, nb_omit):
        self.custom_cv = CustomTimeSeriesSplit(n_splits=n_splits)
        self.nb_omit = nb_omit

    def cross_validate(self, model, Predictant, Predictor=None, clim_year_start=None, clim_year_end=None, **model_params):
        """
        Perform cross-validation to compute deterministic hindcasts and tercile probabilities.

        Iterates over time series splits, trains the model on training data, and generates
        predictions for test data. Supports special handling for specific model types
        (e.g., CCA, Analog, ELM, ELR, and various machine learning models).

        Parameters
        ----------
        model : object
            The forecasting model instance to evaluate.
        Predictant : xarray.DataArray
            Target dataset with dimensions ('T', 'Y', 'X') or ('T', 'M', 'Y', 'X').
        Predictor : xarray.DataArray, optional
            Predictor dataset with dimensions ('T', 'M', 'Y', 'X') or ('T', 'features').
            Required for most models except specific cases like WAS_Analog.
        clim_year_start : int or str, optional
            Start year of the climatology period for standardization and probability calculations.
        clim_year_end : int or str, optional
            End year of the climatology period for standardization and probability calculations.
        **model_params : dict
            Additional keyword arguments to pass to the model's compute_model method.

        Returns
        -------
        tuple or xarray.DataArray
            If the model supports probability calculations (has compute_prob method):
                - hindcast_det : xarray.DataArray
                    Deterministic hindcast results with dimensions ('T', 'Y', 'X') or
                    ('probability', 'T', 'Y', 'X') for specific models.
                - hindcast_prob : xarray.DataArray
                    Tercile probabilities with dimensions ('probability', 'T', 'Y', 'X'),
                    where 'probability' includes ['PB', 'PN', 'PA'] (below-normal, normal, above-normal).
            If the model does not support probability calculations (e.g., WAS_mme_ELR):
                - hindcast_det : xarray.DataArray
                    Deterministic hindcast results with dimensions ('T', 'Y', 'X').
        """
        hindcast_det = []
        hindcast_prob = []
        n_splits = len(Predictant.get_index("T"))

        same_kind_model1 = [WAS_LinearRegression_Model, WAS_MARS_Model,
                          WAS_PolynomialRegression,
                           WAS_Ridge_Model, WAS_Lasso_Model,
                           WAS_LassoLars_Model, WAS_ElasticNet_Model,
                          WAS_RandomForest_Model, WAS_ExtraTrees_Model,
                         WAS_GradientBoosting_Model, WAS_AdaBoost_Model,
                         WAS_XGBoost_Model, WAS_LightGBM_Model,
                         WAS_SVR_Model, WAS_KNN_Model]
                         
        same_kind_count = [WAS_PoissonRegression, WAS_NegativeBinomial_Model, WAS_ZINB_Model, WAS_HurdleNB_Model]

        same_kind_model2 = [WAS_KNN_Classifier_Model, WAS_SVC_Classifier_Model, WAS_RandomForest_Classifier_Model, WAS_ExtraTrees_Classifier_Model, WAS_GradientBoosting_Classifier_Model, WAS_AdaBoost_Classifier_Model, WAS_LightGBM_Classifier_Model, WAS_XGBoost_Classifier_Model, WAS_NaiveBayes_Classifier_Model, WAS_LinearDiscriminant_Classifier_Model, WAS_QuadraticDiscriminant_Classifier_Model, WAS_LogisticRegression_Model]


        same_kind_model3 = [WAS_mme_MLP, WAS_mme_hpELM,
                            WAS_mme_XGBoosting, WAS_mme_RF,
                            WAS_mme_PCR]

        if isinstance(model, WAS_CCA_op):
            all_params = {**model_params}
            params_prob = {
                k: v for k, v in all_params.items()
                if k not in model.compute_model.__code__.co_varnames
            }
            # print(all_params)
            # print(params_prob)
            params_models = {
                k: v for k, v in all_params.items()
                if k not in params_prob
            }

            Predictant_safe = Predictant.copy(deep=True)
            mask = xr.where(~np.isnan(Predictant_safe.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()

            Predictor_detrend, coeffss, metas = detrended_data(Predictor, dim="T")
            Predictor_ready = Predictor_detrend.fillna(0.0)

            Predictant_st = standardize_timeseries(Predictant_safe, clim_year_start, clim_year_end)
            Predictant_st_detrend, coeffs, meta = detrended_data(Predictant_st, dim="T")
            Predictant_ready = Predictant_st_detrend.fillna(0.0)

            print("Cross-validation ongoing")

            hindcast_list = []

            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train = Predictor_ready.isel(T=train_index)
                X_test = Predictor_ready.isel(T=test_index)
                y_train = Predictant_ready.isel(T=train_index)
                y_test = Predictant_ready.isel(T=test_index)

                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_list.append(pred_det)

            hindcast_det = xr.concat(hindcast_list, dim="T")
            hindcast_det = hindcast_det.sortby("T")

            hindcast_det = hindcast_det + apply_detrend_data(hindcast_det, coeffs, meta)
            hindcast_det = hindcast_det.transpose('T', 'Y', 'X') * mask
            hindcast_det = reverse_standardize(hindcast_det, Predictant_safe, clim_year_start, clim_year_end)
            hindcast_det = hindcast_det.clip(min=0)

            hindcast_det['T'] = Predictant_safe['T']

            hindcast_prob = model.compute_prob(
                Predictant_safe,
                clim_year_start,
                clim_year_end,
                hindcast_det,
                **params_prob
            )
            hindcast_prob = (hindcast_prob * mask).clip(min=0, max=1)
            # print(np.unique(hindcast_prob))
            return hindcast_det, hindcast_prob

        elif isinstance(model, WAS_CCA_base):

            all_params = {**model_params}

            compute_model_args = model.compute_model.__code__.co_varnames

            params_models = {
                key: value
                for key, value in all_params.items()
                if key in compute_model_args
            }

            params_prob = {
                key: value
                for key, value in all_params.items()
                if key not in compute_model_args
            }

            # Remove member dimension if needed.
            if "M" in Predictant.dims:
                Predictant_obs = Predictant.isel(M=0, drop=True)
            else:
                Predictant_obs = Predictant

            if "M" in Predictor.dims:
                Predictor_cv = Predictor.isel(M=0, drop=True)
            else:
                Predictor_cv = Predictor

            Predictant_obs = Predictant_obs.transpose("T", "Y", "X")
            Predictor_cv = Predictor_cv.transpose("T", "Y", "X")

            # Spatial mask based on observed predictand.
            mask = xr.where(
                np.isfinite(Predictant_obs.isel(T=0)),
                1.0,
                np.nan,
            )

            # Standardize predictand only.
            # No trend_data.
            Predictant_st = standardize_timeseries(
                Predictant_obs,
                clim_year_start,
                clim_year_end,
            )

            hindcast_det = []

            print("Cross-validation ongoing")

            for i, (train_index, test_index) in enumerate(
                tqdm(
                    self.custom_cv.split(Predictor_cv["T"], self.nb_omit),
                    total=n_splits,
                ),
                start=1,
            ):

                X_train = Predictor_cv.isel(T=train_index)
                X_test = Predictor_cv.isel(T=test_index)

                y_train = Predictant_st.isel(T=train_index)
                y_test = Predictant_st.isel(T=test_index)

                pred_det = model.compute_model(
                    X_train,
                    y_train,
                    X_test,
                    y_test,
                    **params_models,
                )

                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det = hindcast_det.sortby("T")

            # Reindex to original predictand time axis.
            hindcast_det = hindcast_det.reindex(T=Predictant_obs["T"])

            # Back-transform from standardized predictand space to original rainfall scale.
            hindcast_det = reverse_standardize(
                hindcast_det,
                Predictant_obs,
                clim_year_start,
                clim_year_end,
            )

            hindcast_det = xr.where(hindcast_det < 0, 0, hindcast_det)
            hindcast_det = hindcast_det * mask

            hindcast_prob = model.compute_prob(
                Predictant=Predictant_obs,
                clim_year_start=clim_year_start,
                clim_year_end=clim_year_end,
                hindcast_det=hindcast_det,
                **params_prob,
            )

            hindcast_prob = xr.where(hindcast_prob < 0, 0, hindcast_prob)
            hindcast_prob = hindcast_prob.clip(min=0, max=1)

            return hindcast_det, hindcast_prob

        elif isinstance(model, WAS_CCA):

            all_params = {**model_params}
            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            Predictor_ = (Predictor - extract_leading_eeof_component(Predictor).fillna(extract_leading_eeof_component(Predictor)[-3])).fillna(0)
            Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
            Predictant_ = (Predictant_st - extract_leading_eeof_component(Predictant_st).fillna(extract_leading_eeof_component(Predictant_st)[-3])).fillna(0)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor_['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor_.isel(T=train_index), Predictor_.isel(T=test_index)
                X_train_, X_test_ = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant_.isel(T=train_index), Predictant_.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test_, y_test, **params_models)
                # pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant_['T']
            hindcast_det = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return xr.where(hindcast_det<0, 0, hindcast_det)*mask, xr.where(hindcast_prob<0, 0, hindcast_prob)*mask

        elif isinstance(model, WAS_CCA_strict):
            import inspect

            all_params = {**model_params}

            # Split parameters safely between compute_model() and compute_prob()
            compute_model_args = set(inspect.signature(model.compute_model).parameters.keys())

            params_models = {
                k: v for k, v in all_params.items()
                if k in compute_model_args
            }

            params_prob = {
                k: v for k, v in all_params.items()
                if k not in compute_model_args
            }

            # ------------------------------------------------------------------
            # 1. Prepare predictand and mask
            # ------------------------------------------------------------------
            Predictant_safe = Predictant.copy(deep=True)

            if "M" in Predictant_safe.dims:
                Predictant_safe = Predictant_safe.isel(M=0).squeeze()
                if "M" in Predictant_safe.coords:
                    Predictant_safe = Predictant_safe.drop_vars("M")

            Predictant_safe = Predictant_safe.transpose("T", "Y", "X")

            mask = model._spatial_mask(Predictant_safe)

            n_splits = len(Predictant_safe.get_index("T"))

            print("Cross-validation ongoing")

            hindcast_list = []

            # ------------------------------------------------------------------
            # 2. Strict fold-safe cross-validation
            #    All preprocessing is fitted only on the training fold.
            # ------------------------------------------------------------------
            for train_index, test_index in tqdm(
                self.custom_cv.split(Predictor["T"], self.nb_omit),
                total=n_splits,
            ):
                X_train_raw = Predictor.isel(T=train_index)
                X_test_raw = Predictor.isel(T=test_index)

                y_train_raw = Predictant_safe.isel(T=train_index)
                y_test_raw = Predictant_safe.isel(T=test_index)

                # --------------------------------------------------------------
                # 2.1 Fill predictor using training information only
                # --------------------------------------------------------------
                X_train_filled = model._fill_spatial_gaps(X_train_raw)
                X_test_filled = model._fill_spatial_gaps(
                    X_test_raw,
                    ref=X_train_filled,
                )

                # --------------------------------------------------------------
                # 2.2 Detrend predictor using training fold only
                # --------------------------------------------------------------
                X_train_detr, coeffs_X, meta_X = detrended_data(
                    X_train_filled,
                    dim="T",
                )

                X_test_detr = X_test_filled - apply_detrend_data(
                    X_test_filled,
                    coeffs_X,
                    meta_X,
                )

                X_train_ready = X_train_detr.fillna(0.0)
                X_test_ready = X_test_detr.fillna(0.0)

                # --------------------------------------------------------------
                # 2.3 Standardize predictand using training fold only
                # --------------------------------------------------------------
                y_train_st = standardize_timeseries(
                    y_train_raw,
                    clim_year_start,
                    clim_year_end,
                )

                y_train_st = y_train_st.where(np.isfinite(y_train_st))

                # --------------------------------------------------------------
                # 2.4 Detrend standardized predictand using training fold only
                # --------------------------------------------------------------
                y_train_st_detr, coeffs_Y, meta_Y = detrended_data(
                    y_train_st,
                    dim="T",
                )

                y_train_ready = y_train_st_detr.fillna(0.0)

                # y_test is only used to preserve the validation T coordinate
                y_test_placeholder = xr.zeros_like(y_test_raw)

                # --------------------------------------------------------------
                # 2.5 CCA prediction in transformed space
                # --------------------------------------------------------------
                pred_st_detr = model.compute_model(
                    X_train_ready,
                    y_train_ready,
                    X_test_ready,
                    y_test_placeholder,
                    **params_models,
                )

                pred_st_detr = pred_st_detr.transpose("T", "Y", "X")

                # --------------------------------------------------------------
                # 2.6 Add predictand trend back at validation-year T
                # --------------------------------------------------------------
                pred_st = pred_st_detr + apply_detrend_data(
                    pred_st_detr,
                    coeffs_Y,
                    meta_Y,
                )

                # --------------------------------------------------------------
                # 2.7 Reverse standardization using training fold reference only
                # --------------------------------------------------------------
                pred_det = reverse_standardize(
                    pred_st,
                    y_train_raw,
                    clim_year_start,
                    clim_year_end,
                )

                pred_det = pred_det.transpose("T", "Y", "X") * mask
                pred_det = pred_det.clip(min=0.0)

                hindcast_list.append(pred_det)

            # ------------------------------------------------------------------
            # 3. Concatenate strict hindcasts
            # ------------------------------------------------------------------
            hindcast_det = xr.concat(hindcast_list, dim="T")
            hindcast_det = hindcast_det.sortby("T")
            hindcast_det = hindcast_det.transpose("T", "Y", "X") * mask
            hindcast_det = hindcast_det.clip(min=0.0)

            # Keep exact temporal coordinate order from Predictant
            hindcast_det = hindcast_det.reindex(T=Predictant_safe["T"])

            # ------------------------------------------------------------------
            # 4. Compute probabilities from strict deterministic hindcasts
            # ------------------------------------------------------------------
            hindcast_prob = model.compute_prob(
                Predictant_safe,
                clim_year_start,
                clim_year_end,
                hindcast_det,
                **params_prob,
            )

            hindcast_prob = hindcast_prob.transpose("probability", "T", "Y", "X")
            hindcast_prob = hindcast_prob * mask
            hindcast_prob = hindcast_prob.clip(min=0.0, max=1.0)

            # Normalize PB + PN + PA = 1
            prob_sum = hindcast_prob.sum(dim="probability")
            hindcast_prob = xr.where(prob_sum > 0, hindcast_prob / prob_sum, np.nan)
            hindcast_prob = hindcast_prob * mask

            return hindcast_det, hindcast_prob

        elif isinstance(model, WAS_KernelCCA_base):
                    # WAS_KCCA / WAS_KGCCA / WAS_EOF_KCCA. compute_model est fold-safe et
                    # renvoie le hindcast déjà en ECHELLE PHYSIQUE -> on ne standardise PAS
                    # le prédictand ici et on ne reverse PAS. Le prédicteur peut être un
                    # champ unique OU une LISTE de champs (WAS_EOF_KCCA réduit la liste en PCs).
                    import inspect
                    all_params = {**model_params}
                    cm_args = set(inspect.signature(model.compute_model).parameters.keys())
                    params_models = {k: v for k, v in all_params.items() if k in cm_args}
                    params_prob   = {k: v for k, v in all_params.items() if k not in cm_args}
        
                    is_list_pred = isinstance(Predictor, (list, tuple))
                    Tref = Predictor[0] if is_list_pred else Predictor
        
                    if "M" in Predictant.dims:
                        Predictant = Predictant.isel(M=0, drop=True)
                    Predictant = Predictant.transpose("T", "Y", "X")
                    mask = xr.where(np.isfinite(Predictant.isel(T=0)), 1.0, np.nan)
        
                    # Tuner UNE fois avant la CV (no-op linéaire) ; pour WAS_EOF_KCCA passe la liste :
                    #   model.compute_hyperparameters(Predictor, Predictant, clim_year_start, clim_year_end)
        
                    print("Cross-validation ongoing")
                    hindcast_det = []
                    for i, (train_index, test_index) in enumerate(
                            tqdm(self.custom_cv.split(Tref['T'], self.nb_omit), total=n_splits), start=1):
                        if is_list_pred:
                            X_train = [f.isel(T=train_index) for f in Predictor]
                            X_test  = [f.isel(T=test_index)  for f in Predictor]
                        else:
                            X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                        y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                        pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                        hindcast_det.append(pred_det)
        
                    hindcast_det = xr.concat(hindcast_det, dim="T").sortby("T").reindex(T=Predictant['T'])
                    hindcast_det = xr.where(hindcast_det < 0, 0, hindcast_det) * mask
                    hindcast_prob = model.compute_prob(
                        Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob) * mask
                    return hindcast_det, hindcast_prob

        elif isinstance(model, WAS_Analog):

            # revoir l'option dutiliser download_and_process ici, enfin deviter les repetitions

            all_params = {**model_params}
            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(np.unique(Predictant['T'].dt.year), self.nb_omit), total=n_splits), start=1):
                pred_det = model.compute_model(Predictant, train_index, test_index)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)
            return hindcast_det, hindcast_prob

        elif isinstance(model, WAS_mme_xcELM):
            # revoir xcast pour la standardisation
            all_params = {**model_params}
            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['S'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(S=train_index), Predictor.isel(S=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det.load(), hindcast_prob.load()

        elif isinstance(model, WAS_mme_xcELR):
            # revoir xcast pour la standardisation
            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['S'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(S=train_index), Predictor.isel(S=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hd = hindcast_det.load()
            return hd, hd


        elif isinstance(model, WAS_mme_logistic):
        
                    mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
        
                    if "M" in Predictant.coords:
                        Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
                    Predictor['T'] = Predictant['T']
        
                    verify = WAS_Verification()
                    Predictant_class = verify.compute_class(Predictant, clim_year_start, clim_year_end)
                    Predictor_st = standardize_timeseries(Predictor, clim_year_start, clim_year_end)
        
                    best_params = model_params.get('best_params')
                    cluster_da  = model_params.get('cluster_da')
                    if best_params is None or cluster_da is None:
                        best_params, cluster_da = model.compute_hyperparameters(
                            Predictor, Predictant, clim_year_start, clim_year_end)
        
                    print("Cross-validation ongoing")
                    for i, (train_index, test_index) in enumerate(
                            tqdm(self.custom_cv.split(Predictor_st['T'], self.nb_omit), total=n_splits), start=1):
                        X_train, X_test = Predictor_st.isel(T=train_index), Predictor_st.isel(T=test_index)
                        y_train, y_test = Predictant_class.isel(T=train_index), Predictant.isel(T=test_index)
                        pred_det, pred_prob = model.compute_model(
                            X_train, y_train, X_test, y_test, clim_year_start, clim_year_end,
                            best_params=best_params, cluster_da=cluster_da)
                        hindcast_det.append(pred_det)
                        hindcast_prob.append(pred_prob)
        
                    hindcast_det = xr.concat(hindcast_det, dim="T").sortby("T")
                    hindcast_det['T'] = Predictant['T']
                    hindcast_prob = xr.concat(hindcast_prob, dim="T").sortby("T")
                    hindcast_prob['T'] = Predictant['T']
        
                    return hindcast_det * mask, hindcast_prob * mask
        
        elif isinstance(model, WAS_mme_ELR):
            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            Predictor['T'] = Predictant['T']
            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
    
            # Standardize the PREDICTOR only. Keep the PREDICTAND RAW: the ELR fit is
            # invariant to predictand scaling, and raw values let g(q) act in physical
            # units (so g='sqrt' is valid for precipitation).
            Predictor_st = standardize_timeseries(Predictor, clim_year_start, clim_year_end)
    
            # fixed climatological tercile boundaries in PHYSICAL (raw) units so the
            # forecast categories match WAS_Verification.compute_class exactly
            clim_raw = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
            terc = clim_raw.quantile([1/3, 2/3], dim="T")
            t1 = terc.isel(quantile=0).drop_vars("quantile")
            t2 = terc.isel(quantile=1).drop_vars("quantile")
    
            print("Cross-validation ongoing")
            # OPTIONAL: tune (l2, g) ONCE before the loop. It stores the winners in
            # self.l2 / self.threshold_transform, which compute_model then uses for
            # every fold with no extra plumbing (pass the same fixed terciles):
            #   model.compute_hyperparameters(Predictor, Predictant,
            #                                 clim_year_start, clim_year_end,
            #                                 clim_terciles=(t1, t2), score="rps")
            for i, (train_index, test_index) in enumerate(
                    tqdm(self.custom_cv.split(Predictor_st['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor_st.isel(T=train_index), Predictor_st.isel(T=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)  # RAW
                pred_prob = model.compute_model(X_train, y_train, X_test, y_test,
                                                clim_year_start, clim_year_end,
                                                clim_terciles=(t1, t2))
                hindcast_prob.append(pred_prob)
    
            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob['T'] = Predictant['T']
            hindcast_prob = (hindcast_prob * mask).clip(min=0, max=1)
            return hindcast_prob, hindcast_prob

        elif isinstance(model, (WAS_mme_NGR_Gaussian, WAS_mme_NGR_NonGaussian)):
            all_params = {**model_params}
            params_prob = {
                key: value for key, value in all_params.items()
                if key in model.compute_model.__code__.co_varnames
            }
        
            Predictor = Predictor.assign_coords(T=Predictant['T'].values)
            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0, drop=True)
        
            member_dim = params_prob.get('member_dim') or "M"
            time_dim   = params_prob.get('time_dim')   or "T"
            lat_dim    = params_prob.get('lat_dim')    or "Y"
            lon_dim    = params_prob.get('lon_dim')    or "X"
        
            quantiles = params_prob.get('quantiles')
            return_synthetic_ensemble = bool(params_prob.get('return_synthetic_ensemble', False))
        
            clim_obs = Predictant.sel(T=slice(str(clim_year_start), str(clim_year_end)))
        
            hindcast_prob, hindcast_det = [], []
            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(
                tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1
            ):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train = Predictant.isel(T=train_index)
        
                out = model.compute_model(
                    X_train, y_train, X_test,
                    obs_for_terciles=clim_obs,
                    quantiles=quantiles,
                    clim_terciles=True,
                    return_synthetic_ensemble=return_synthetic_ensemble,
                    member_dim=member_dim, time_dim=time_dim,
                    lat_dim=lat_dim, lon_dim=lon_dim,
                )
                hindcast_prob.append(out['tercile_probability'])
                hindcast_det.append(out['calibrated_mean'])
        
            hindcast_prob = xr.concat(hindcast_prob, dim="T").sortby("T")
            hindcast_det  = xr.concat(hindcast_det,  dim="T").sortby("T")
            return hindcast_det, hindcast_prob   
            
        elif isinstance(model, WAS_mme_FullBMA):
                    import inspect
        
                    all_params = {**model_params}
        
                    # Extraction robuste des arguments valides pour compute_model
                    cm_sig = inspect.signature(model.compute_model)
                    cm_args = set(cm_sig.parameters.keys())
        
                    params_models = {
                        k: v for k, v in all_params.items() if k in cm_args
                    }
        
                    # dist_map est un paramètre requis par WAS_mme_FullBMA
                    if 'dist_map' not in params_models:
                        raise ValueError("Le paramètre 'dist_map' est requis pour WAS_mme_FullBMA et doit être fourni.")
                    dist_map = params_models.pop('dist_map')
        
                    mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
                    Predictor['T'] = Predictant['T']
                    
                    if "M" in Predictant.coords:
                        Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
        
                    print("Cross-validation ongoing (FullBMA)")
                    hindcast_det = []
                    hindcast_prob = []
        
                    for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                        X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                        y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                        
                        # compute_model retourne un xarray.Dataset contenant les prédictions
                        ds_out = model.compute_model(
                            X_train=X_train, 
                            y_train=y_train, 
                            X_test=X_test, 
                            dist_map=dist_map, 
                            **params_models
                        )
                        
                        # Extraction du hindcast déterministe (Moyenne du mélange)
                        hindcast_det.append(ds_out['predictive_mean'])
                        
                        # Extraction des probabilités (si les arguments climatologiques ont été fournis)
                        if 'tercile_probability' in ds_out:
                            hindcast_prob.append(ds_out['tercile_probability'])
        
                    # Concaténation et Masquage
                    hindcast_det = xr.concat(hindcast_det, dim="T").sortby("T")
                    hindcast_det['T'] = Predictant['T']
                    hindcast_det = hindcast_det * mask
        
                    if hindcast_prob:
                        hindcast_prob = xr.concat(hindcast_prob, dim="T").sortby("T")
                        hindcast_prob['T'] = Predictant['T']
                        hindcast_prob = hindcast_prob * mask
                        return hindcast_det, hindcast_prob
                        
                    return hindcast_det, hindcast_det
    
        elif isinstance(model, WAS_mme_FastBMA):
            import inspect

            all_params = {**model_params}

            # FastBMA gère l'entraînement et la prédiction en deux temps (fit et predict_probabilistic)
            fit_args = set(inspect.signature(model.fit).parameters.keys())
            pred_args = set(inspect.signature(model.predict_probabilistic).parameters.keys())

            params_fit = {k: v for k, v in all_params.items() if k in fit_args}
            params_pred = {k: v for k, v in all_params.items() if k in pred_args}

            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
            Predictor['T'] = Predictant['T']
            
            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()

            print("Cross-validation ongoing (FastBMA)")
            hindcast_det = []
            hindcast_prob = []

            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)

                # Ajustement et prédiction
                model.fit(hcst_grid=X_train, obs_grid=y_train, **params_fit)
                ds_out = model.predict_probabilistic(new_forecasts=X_test, **params_pred)

                hindcast_det.append(ds_out['predictive_mean'])
                if 'tercile_probability' in ds_out:
                    hindcast_prob.append(ds_out['tercile_probability'])

            hindcast_det = xr.concat(hindcast_det, dim="T").sortby("T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det * mask

            if hindcast_prob:
                hindcast_prob = xr.concat(hindcast_prob, dim="T").sortby("T")
                hindcast_prob['T'] = Predictant['T']
                hindcast_prob = hindcast_prob * mask
                return hindcast_det, hindcast_prob
                
            return hindcast_det, hindcast_det

        
        elif isinstance(model, WAS_mme_gaussian_process):

            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            else:
                Predictant = Predictant
            all_params = {**model_params}

            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            Predictor['T'] = Predictant['T']
            verify = WAS_Verification()
            Predictant_class = verify.compute_class(Predictant, clim_year_start, clim_year_end)
            Predictor_st = standardize_timeseries(Predictor, clim_year_start, clim_year_end)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor_st.isel(T=train_index), Predictor_st.isel(T=test_index)
                y_train, y_test = Predictant_class.isel(T=train_index), Predictant.isel(T=test_index)
                pred_det, pred_prob = model.compute_model(X_train, y_train, X_test, y_test, clim_year_start, clim_year_end, **model_params)
                hindcast_det.append(pred_det)
                hindcast_prob.append(pred_prob)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob['T'] = Predictant['T']

            return hindcast_det, hindcast_prob

        elif isinstance(model, WAS_Min2009_ProbWeighted):

            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            else:
                Predictant = Predictant
            all_params = {**model_params}

            params_prob = {
                key: value for key, value in all_params.items()
                if key in model.compute_pmme_probabilities.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            masks = params_models.get('masks')
            ensemble_sizes = params_prob.get('ensemble_sizes')
            climatology = params_prob.get('climatology')

            # mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            Predictor['T'] = Predictant['T']

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                forecasts = {k: X_test.sel(M=v) for k, v in masks.items()}
                hindcasts = {k: X_train.sel(M=v) for k, v in masks.items()}
                pred_prob = model.compute_pmme_probabilities(forecasts, hindcasts, climatology, ensemble_sizes)
                pred_prob  = xr.concat([pred_prob["PB"], pred_prob["PN"], pred_prob["PA"]], dim=xr.DataArray(["PB", "PN", "PA"], dims="probability", name="probability"),)
                hindcast_prob.append(pred_prob)

            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob['T'] = Predictant['T']

            return hindcast_prob, hindcast_prob

        # ── same_kind_model2: non-neural classifiers ──────────────────────
        elif any(isinstance(model, cls) for cls in same_kind_model2):
        
            all_params   = {**model_params}
            params_models = {
                key: value for key, value in all_params.items()
                if key in model.compute_model.__code__.co_varnames
            }
            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            mask = (
                xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan)
                .drop_vars(["T"])
                .squeeze()
                .to_numpy()
            )

            Predictant_class, _, _ = model.compute_class(
                Predictant, clim_year_start, clim_year_end
            )

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(
                tqdm(
                    self.custom_cv.split(Predictor["T"], self.nb_omit),
                    total=n_splits,
                ),
                start=1,
            ):
                X_train       = Predictor.isel(T=train_index)
                X_test        = Predictor.isel(T=test_index)
                y_class_train = Predictant_class.isel(T=train_index)

                pred_prob = model.compute_model(
                    X_train, y_class_train, X_test, **params_models
                )
                hindcast_prob.append(pred_prob)

            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob["T"] = Predictant["T"]
            hindcast_prob = (
                hindcast_prob
                .transpose("probability", "T", "Y", "X")
                * mask
            )

            return hindcast_prob, hindcast_prob

        # ── WAS_PCR wrapping a same_kind_model2 classifier ───────────────
        elif (
            isinstance(model, WAS_PCR)
            and any(isinstance(model.__dict__["reg_model"], cls) for cls in same_kind_model2)
        ):

            inner_model = model.__dict__["reg_model"]

            all_params    = {**model_params}
            params_models = {
                key: value for key, value in all_params.items()
                if key in inner_model.compute_model.__code__.co_varnames
            }
            params_prob = {
                key: value for key, value in all_params.items()
                if key not in inner_model.compute_model.__code__.co_varnames
            }

            mask = (
                xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan)
                .drop_vars(["T"])
                .squeeze()
                .to_numpy()
            )

            Predictant_class, _, _ = inner_model.compute_class(
                Predictant, clim_year_start, clim_year_end
            )

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(
                tqdm(
                    self.custom_cv.split(
                        _pcr_T_index(Predictor), self.nb_omit
                    ),
                    total=n_splits,
                ),
                start=1,
            ):
                X_train       = _pcr_isel_T(Predictor, train_index)
                X_test        = _pcr_isel_T(Predictor, test_index)
                y_class_train = Predictant_class.isel(T=train_index)

                pred_prob = model.compute_model(
                    X_train, y_class_train, X_test, **params_models
                )
                hindcast_prob.append(pred_prob)

            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob["T"] = Predictant["T"]
            hindcast_prob = (
                hindcast_prob
                .transpose("probability", "T", "Y", "X")
                * mask
            )

            return hindcast_prob, hindcast_prob
            
             
        # ── count regression models (Poisson, NB, ZINB, Hurdle) ──────────
        elif any(isinstance(model, cls) for cls in same_kind_count):

            # Param split — same co_varnames introspection as same_kind_model1
            all_params    = {**model_params}
            params_models = {
                k: v for k, v in all_params.items()
                if k in model.compute_model.__code__.co_varnames
            }
            params_prob = {
                k: v for k, v in all_params.items()
                if k not in model.compute_model.__code__.co_varnames
            }

            mask = (
                xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan)
                .drop_vars(["T"])
                .squeeze()
                .to_numpy()
            )

            if "M" in Predictant.dims:
                Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

            # No standardize_timeseries — Poisson/NB require raw non-negative
            # counts; standardisation produces negative / non-integer values
            # that break the log-link GLM.
            
            Predictant = Predictant.transpose("T", "Y", "X")

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(
                tqdm(
                    self.custom_cv.split(Predictor["T"], self.nb_omit),
                    total=n_splits,
                ),
                start=1,
            ):
                X_train = Predictor.isel(T=train_index)
                X_test  = Predictor.isel(T=test_index)
                y_train = Predictant.isel(T=train_index)   # raw counts
                y_test  = Predictant.isel(T=test_index)    # raw counts

                # compute_model returns (Y, X) for T=1 folds
                pred_det = model.compute_model(
                    X_train, y_train, X_test, y_test, **params_models
                )

                # Restore T dimension before concatenation
                if "T" not in pred_det.dims:
                    pred_det = pred_det.expand_dims(
                        T=Predictant.isel(T=test_index)["T"].values
                    )
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det["T"] = Predictant["T"]
            hindcast_det = hindcast_det.transpose("T", "Y", "X") * mask

            # No reverse_standardize — predictions are already in raw count units
            hindcast_prob = model.compute_prob(
                Predictant, clim_year_start, clim_year_end,
                hindcast_det, **params_prob
            )
            return hindcast_det * mask, hindcast_prob * mask

        # ── WAS_PCR wrapping a count model ────────────────────────────────
        elif (
            isinstance(model, WAS_PCR)
            and any(isinstance(model.__dict__["reg_model"], cls) for cls in same_kind_count)
        ):
            inner_model = model.__dict__["reg_model"]

            all_params    = {**model_params}
            params_models = {
                k: v for k, v in all_params.items()
                if k in inner_model.compute_model.__code__.co_varnames
            }
            params_prob = {
                k: v for k, v in all_params.items()
                if k not in inner_model.compute_model.__code__.co_varnames
            }

            mask = (
                xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan)
                .drop_vars(["T"])
                .squeeze()
                .to_numpy()
            )

            if "M" in Predictant.dims:
                Predictant = Predictant.isel(M=0).drop_vars("M").squeeze()

            Predictant = Predictant.transpose("T", "Y", "X")

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(
                tqdm(
                    self.custom_cv.split(
                        _pcr_T_index(Predictor), self.nb_omit
                    ),
                    total=n_splits,
                ),
                start=1,
            ):
                X_train = _pcr_isel_T(Predictor, train_index)
                X_test  = _pcr_isel_T(Predictor, test_index)
                y_train = Predictant.isel(T=train_index)
                y_test  = Predictant.isel(T=test_index)

                pred_det = model.compute_model(
                    X_train, y_train, X_test, y_test, **params_models
                )

                if "T" not in pred_det.dims:
                    pred_det = pred_det.expand_dims(
                        T=Predictant.isel(T=test_index)["T"].values
                    )
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det["T"] = Predictant["T"]
            hindcast_det = hindcast_det.transpose("T", "Y", "X") * mask

            hindcast_prob = inner_model.compute_prob(
                Predictant, clim_year_start, clim_year_end,
                hindcast_det, **params_prob
            )
            return hindcast_det * mask, hindcast_prob * mask

        elif any(isinstance(model, i) for i in same_kind_model3):

            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            else:
                Predictant = Predictant

            all_params = {**model_params}

            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            Predictor['T'] = Predictant['T']
            Predictor_st = standardize_timeseries(Predictor, clim_year_start, clim_year_end)

            Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor_st['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor_st.isel(T=train_index), Predictor_st.isel(T=test_index)
                y_train, y_test = Predictant_st.isel(T=train_index), Predictant_st.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det*mask, hindcast_prob*mask


        ## Revenir sur  ici
        elif any(isinstance(model, i) for i in same_kind_model1):

            all_params = {**model_params}

            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.compute_model.__code__.co_varnames
            }

            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

            Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant_st.isel(T=train_index), Predictant_st.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det*mask
            hindcast_det = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det*mask, hindcast_prob*mask


######## WAS_PCR Gestion

        elif (isinstance(model, WAS_PCR) and any(isinstance(model.__dict__['reg_model'], i) for i in same_kind_model1)):

            all_params = {**model_params}

            params_prob = {
                key: value for key, value in all_params.items()
                if key not in model.__dict__['reg_model'].compute_model.__code__.co_varnames
            }
            params_models = {
                key: value for key, value in all_params.items()
                if key not in params_prob
            }

            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

            Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(_pcr_T_index(Predictor), self.nb_omit), total=n_splits), start=1):
                X_train, X_test = _pcr_isel_T(Predictor, train_index), _pcr_isel_T(Predictor, test_index)
                y_train, y_test = Predictant_st.isel(T=train_index), Predictant_st.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det.transpose('T', 'Y', 'X')*mask

            hindcast_det = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)

            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det.transpose('T', 'Y', 'X')*mask, hindcast_prob*mask

        else:
            print("not defined models for cross-validation")
