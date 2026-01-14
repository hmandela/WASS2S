import numpy as np
import xarray as xr
from tqdm import tqdm
from wass2s.was_linear_models import *
from wass2s.was_eof import *
from wass2s.was_pcr import *
from wass2s.was_cca import *
from wass2s.was_machine_learning import *
from wass2s.was_analog import *
from wass2s.utils import *
from wass2s.was_mme import *

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
            
        # n_samples = len(X)
        # indices = np.arange(n_samples)

        # for i in range(n_samples):
        #     test_indices = [i]
        #     train_indices = indices[:i]
        #     if len(train_indices) < self.n_splits:
        #         train_indices = np.concatenate([indices[i+1:], indices[:i]])

        #     yield train_indices[nb_omit:], test_indices

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

    # def get_model_params(self, model):
    #     """
    #     Retrieve parameters required for the model's compute_model method.

    #     Extracts parameters dynamically from the model's attributes that match the
    #     argument names of its compute_model method.

    #     Parameters
    #     ----------
    #     model : object
    #         The forecasting model instance to inspect.

    #     Returns
    #     -------
    #     dict
    #         Dictionary of parameter names and values to pass to the model's compute_model and compute_prob method.
    #     """
    #     params = {}
    #     compute_model_params = model.compute_model.__code__.co_varnames[1:model.compute_model.__code__.co_argcount]
    #     for param in compute_model_params:
    #         if hasattr(model, param):
    #             params[param] = getattr(model, param)
    #     return params

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
                           WAS_LassoLars_Model, WAS_ElasticNet_Model]
        
        same_kind_model1_1 = [WAS_SVR, WAS_RandomForest_XGBoost_ML_Stacking,
                            WAS_MLP, WAS_Stacking_Ridge,
                            WAS_RandomForest_XGBoost_Stacking_MLP]
        
        same_kind_model2 = [WAS_mme_MLP, WAS_mme_hpELM,
                            WAS_mme_MLP_, WAS_mme_hpELM_,
                           WAS_mme_Stacking_, WAS_mme_Stacking]

        same_kind_model3 = [WAS_mme_XGBoosting, WAS_mme_RF,
                            WAS_mme_XGBoosting_, WAS_mme_RF_]
        

        if isinstance(model, WAS_CCA):

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
            Predictor_ = (Predictor - trend_data(Predictor).fillna(trend_data(Predictor)[-3])).fillna(0)
            Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)
            Predictant_ = (Predictant_st - trend_data(Predictant_st).fillna(trend_data(Predictant_st)[-3])).fillna(0)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor_['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor_.isel(T=train_index), Predictor_.isel(T=test_index)
                X_train_, X_test_ = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant_.isel(T=train_index), Predictant_.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test_, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant_['T']
            hindcast_det = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return xr.where(hindcast_det<0, 0, hindcast_det)*mask, xr.where(hindcast_prob<0, 0, hindcast_prob)*mask

        # elif isinstance(model, WAS_KCCA):

        #     all_params = {**model_params}
        #     params_prob = {
        #         key: value for key, value in all_params.items() 
        #         if key not in model.compute_model.__code__.co_varnames
        #     }

        #     params_models = {
        #         key: value for key, value in all_params.items() 
        #         if key not in params_prob
        #     } 

            
        #     mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()
            
        #     Predictant_st = standardize_timeseries(Predictant, clim_year_start, clim_year_end)

        #     print("Cross-validation ongoing")
        #     for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor_['T'], self.nb_omit), total=n_splits), start=1):

        #         X_train_, X_test_ = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                
        #         y_train, y_test = Predictant_.isel(T=train_index), Predictant_.isel(T=test_index)
                
        #         pred_det = model.compute_model(X_train, y_train, X_test_, y_test, **params_models)
        #         hindcast_det.append(pred_det)

        #     hindcast_det = xr.concat(hindcast_det, dim="T")
        #     hindcast_det['T'] = Predictant_['T']
        #     hindcast_det = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)
        #     hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

        #     return xr.where(hindcast_det<0, 0, hindcast_det)*mask, xr.where(hindcast_prob<0, 0, hindcast_prob)*mask


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
            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            else:
                Predictant = Predictant

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

        elif isinstance(model, WAS_LogisticRegression_Model) or (
            isinstance(model, WAS_PCR) and isinstance(model.__dict__['reg_model'], WAS_LogisticRegression_Model)):
            
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
            
            if "M" in Predictant.coords:
                Predictant = Predictant.isel(M=0).drop_vars('M').squeeze()
            else:
                Predictant = Predictant

            verify = WAS_Verification()
            Predictant_class = verify.compute_class(Predictant, clim_year_start, clim_year_end)
            # Predictor_st = standardize_timeseries(Predictor, clim_year_start, clim_year_end)

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor_st.isel(T=test_index)
                y_train, y_test = Predictant_class.isel(T=train_index), Predictant.isel(T=test_index)
                pred_prob = model.compute_model(X_train, y_train, X_test, y_test, clim_year_start, clim_year_end, **model_params)
                hindcast_prob.append(pred_prob)

            hindcast_prob = xr.concat(hindcast_prob, dim="T")
            hindcast_prob['T'] = Predictant['T']

            return hindcast_prob, hindcast_prob

        elif isinstance(model, WAS_PoissonRegression) or (
            isinstance(model, WAS_PCR) and isinstance(model.__dict__['reg_model'], WAS_PoissonRegression)):
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


            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det*mask
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det*mask, hindcast_prob*mask
        

        elif any(isinstance(model, i) for i in same_kind_model2):
            
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
                X_train, X_test = Predictor_st.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant_st.isel(T=train_index), Predictant_st.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det*mask
            hindcast_det_ = reverse_standardize(hindcast_det, Predictant, clim_year_start, clim_year_end)
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det*mask, hindcast_prob*mask

        
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
            
            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det*mask
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det*mask, hindcast_prob*mask

        elif any(isinstance(model, i) for i in same_kind_model1_1) or (
            isinstance(model, WAS_PCR) and any(isinstance(model.__dict__['reg_model'], i) for i in same_kind_model1_1)
        ):
            all_params = {**model_params}
            
            if isinstance(model, WAS_PCR):              
                params_prob = {
                    key: value for key, value in all_params.items() 
                    if key not in model.__dict__['reg_model'].compute_model.__code__.co_varnames
                }
                params_models = {
                    key: value for key, value in all_params.items() 
                    if key not in params_prob
                }
            else:
                params_prob = {
                    key: value for key, value in all_params.items() 
                    if key not in model.compute_model.__code__.co_varnames
                }
    
                params_models = {
                    key: value for key, value in all_params.items() 
                    if key not in params_prob
                } 
            
            mask = xr.where(~np.isnan(Predictant.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze().to_numpy()

            print("Cross-validation ongoing")
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
                y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
                pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
                hindcast_det.append(pred_det)

            hindcast_det = xr.concat(hindcast_det, dim="T")
            hindcast_det['T'] = Predictant['T']
            hindcast_det = hindcast_det*mask
            hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

            return hindcast_det*mask, hindcast_prob*mask


        elif any(isinstance(model, i) for i in same_kind_model1) or (
            isinstance(model, WAS_PCR) and any(isinstance(model.__dict__['reg_model'], i) for i in same_kind_model1)
        ):
            all_params = {**model_params}
            
            if isinstance(model, WAS_PCR):              
                params_prob = {
                    key: value for key, value in all_params.items() 
                    if key not in model.__dict__['reg_model'].compute_model.__code__.co_varnames
                }
                params_models = {
                    key: value for key, value in all_params.items() 
                    if key not in params_prob
                }
            else:
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
            
        else:
            print("not defined models for cross-validation")
            
        #     all_params = {**model_params}
            

        #     params_prob = {
        #         key: value for key, value in all_params.items() 
        #         if key not in model.compute_model.__code__.co_varnames
        #     }
        #     print(params_prob)

        #     params_models = {
        #         key: value for key, value in all_params.items() 
        #         if key not in params_prob
        #     } 
        #     print(params_models)
            
        #     print("Cross-validation ongoing")
        #     for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
        #         X_train, X_test = Predictor.isel(T=train_index), Predictor.isel(T=test_index)
        #         y_train, y_test = Predictant.isel(T=train_index), Predictant.isel(T=test_index)
        #         if 'y_test' in model.compute_model.__code__.co_varnames:
        #             pred_det = model.compute_model(X_train, y_train, X_test, y_test, **params_models)
        #         else:
        #             pred_det = model.compute_model(X_train, y_train, X_test, **params_models)
        #         hindcast_det.append(pred_det)

        #     hindcast_det = xr.concat(hindcast_det, dim="T")
        #     hindcast_det['T'] = Predictant['T']

        #     if any([isinstance(model, i) for i in same_prob_method]):
        #         hindcast_det = hindcast_det.transpose('T', 'Y', 'X')
        #     if isinstance(model, WAS_LogisticRegression_Model):
        #         hindcast_det = hindcast_det.transpose('probability', 'T', 'Y', 'X')
        #     if isinstance(model, WAS_PCR) and any([isinstance(model.__dict__['reg_model'], i) for i in same_prob_method]):
        #         hindcast_det = hindcast_det.transpose('T', 'Y', 'X')
        #     if isinstance(model, WAS_PCR) and isinstance(model.__dict__['reg_model'], WAS_LogisticRegression_Model):
        #         hindcast_det = hindcast_det.transpose('probability', 'T', 'Y', 'X')

        #     if clim_year_start and clim_year_end and hasattr(model, 'compute_prob'):
        #         hindcast_prob = model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det, **params_prob)

        #     if hasattr(model, 'compute_prob') and hindcast_prob is not None and 'T' in hindcast_prob.dims and hindcast_prob.sizes['T'] > 0:
        #         return hindcast_det, hindcast_prob
        #     else:
        #         return hindcast_det

