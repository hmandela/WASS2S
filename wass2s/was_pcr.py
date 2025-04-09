from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor as VIF
from statsmodels.stats.anova import anova_lm
import xarray as xr 
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
import xeofs as xe
import xarray as xr
import numpy as np
import scipy.signal as sig
from scipy.interpolate import CubicSpline
from multiprocessing import cpu_count
from dask.distributed import Client
import dask.array as da
from wass2s.was_linear_models import *
from wass2s.was_eof import *
from wass2s.was_machine_learning import *

### Complete WAS_PCR with PCR with multiple eof zone!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

class WAS_PCR:

    """
    A class for performing Principal Component Regression (PCR) using EOF analysis and variable regression models.

    This class integrates the WAS_EOF for dimensionality reduction through Empirical Orthogonal Function (EOF)
    analysis and allows the use of different regression models for predicting a target variable based on the
    principal components.

    Attributes
    ----------
    eof_model : WAS_EOF
        The EOF analysis model used for dimensionality reduction.
    reg_model : object
        A regression model (e.g., WAS_LinearRegression_Model, WAS_Ridge_Model, etc.) used for regression on the PCs.
    """

    def __init__(self, regression_model, n_modes=None, use_coslat=True, standardize=False,
                 #detrend=True, 
                 opti_explained_variance=None, L2norm=True):
        """
        Initializes the WAS_PCR class with EOF and a flexible regression model.

        Parameters
        ----------
        regression_model : object
            An instance of any regression model class (e.g., WAS_Ridge_Model, WAS_Lasso_Model).
        n_modes : int, optional
            Number of EOF modes to retain, passed to WAS_EOF.
        use_coslat : bool, optional
            Whether to apply cosine latitude weighting in EOF analysis, passed to WAS_EOF.
        standardize : bool, optional
            Whether to standardize the input data, passed to WAS_EOF.
        detrend : bool, optional
            Whether to detrend the input data, passed to WAS_EOF.
        opti_explained_variance : float, optional
            Target cumulative explained variance to determine the number of EOF modes.
        L2norm : bool, optional
            Whether to normalize EOF components and scores to have L2 norm, passed to WAS_EOF.
        multivariate : bool, optional
            Whether to perform multivariate EOF analysis.
        compute_individual : bool, optional
            Whether to compute separate EOFs for each variable in a multivariate list.
        """
        # same_probb = [WAS_LinearRegression_Model, WAS_Ridge_Model, WAS_Lasso_Model, WAS_LassoLars_Model, WAS_ElasticNet_Model,
        #              WAS_RandomForest_XGBoost_ML_Stacking, WAS_MLP, WAS_Stacking_Ridge, WAS_RandomForest_XGBoost_Stacking_MLP]
        
        self.eof_model = WAS_EOF(n_modes=n_modes, use_coslat=use_coslat, standardize=standardize, #detrend=detrend,
                                 opti_explained_variance=opti_explained_variance, L2norm=L2norm)
        
        self.reg_model = regression_model  # Set the regression model passed as an argument

    def compute_model(self, X_train, y_train, X_test, y_test=None, alpha=None, l1_ratio=None, **kwargs):
        s_eofs, s_pcs, _, _ = self.eof_model.fit(X_train, dim="T")
        X_test = X_test.fillna(X_train.mean())
        X_test = X_test.drop_vars('T').squeeze().expand_dims({'T': ["1991-01-01"]})
        X_train_pcs = s_pcs.rename({"mode": "features"}).transpose('T', 'features')
        X_test_pcs = self.eof_model.transform(X_test).drop_vars('T').squeeze().rename({"mode": "features"})#.transpose('T', 'features')
        
        if isinstance(self.reg_model, WAS_LinearRegression_Model): 
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test)
        if isinstance(self.reg_model, (WAS_Ridge_Model, WAS_Lasso_Model, WAS_LassoLars_Model)):
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, alpha)
        if isinstance(self.reg_model, WAS_ElasticNet_Model): 
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, alpha, l1_ratio)
        if isinstance(self.reg_model, WAS_LogisticRegression_Model): 
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, alpha)
        if isinstance(self.reg_model, WAS_SVR): 
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, epsilon, C, degree_array=None)
        if isinstance(self.reg_model, WAS_RandomForest_XGBoost_ML_Stacking):
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, best_param_da)
        if isinstance(self.reg_model, WAS_MLP):
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, hl_array, act_array, lr_array, maxiter_array)
        if isinstance(self.reg_model, WAS_Stacking_Ridge):
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, best_param_da)
        if isinstance(self.reg_model, WAS_RandomForest_XGBoost_Stacking_MLP):
            result = self.reg_model.compute_model(X_train_pcs, y_train, X_test_pcs, y_test, best_param_da)
        return result

    def compute_prob(self, Predictant, clim_year_start, clim_year_end, hindcast_det):
        same_probb = [WAS_LinearRegression_Model, WAS_Ridge_Model, WAS_Lasso_Model, WAS_LassoLars_Model, WAS_ElasticNet_Model,
                     WAS_RandomForest_XGBoost_ML_Stacking, WAS_MLP, WAS_Stacking_Ridge, WAS_RandomForest_XGBoost_Stacking_MLP,
                     WAS_SVR, WAS_PolynomialRegression, WAS_PoissonRegression]

        if any(isinstance(self.reg_model, i) for i in same_probb):
            result = self.reg_model.compute_prob(Predictant, clim_year_start, clim_year_end, hindcast_det)
        if isinstance(self.reg_model, WAS_LogisticRegression_Model): 
            result = None
        # if isinstance(self.reg_model, WAS_QuantileRegression_Model): 
        #     result = self.reg_model.compute_prob(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det)
        return result

    def forecast(self, Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year, alpha=None, l1_ratio=None):

        s_eofs, s_pcs, _, _ = self.eof_model.fit(Predictor, dim="T")
        Predictor_for_year = Predictor_for_year.fillna(Predictor.mean())
        # Predictor_for_year = Predictor_for_year.drop_vars('T').squeeze().expand_dims({'T': ["1991-01-01"]})
        Predictor_for_year_pcs = self.eof_model.transform(Predictor_for_year).rename({"mode": "features"}).transpose('T', 'features') #.drop_vars('T').squeeze()
        Predictor = s_pcs.rename({"mode": "features"}).transpose('T', 'features')
        
        if isinstance(self.reg_model, (WAS_LinearRegression_Model, WAS_PolynomialRegression, WAS_PoissonRegression)): 
            result = self.reg_model.forecast( Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs)
            
        if isinstance(self.reg_model, (WAS_Ridge_Model, WAS_Lasso_Model, WAS_LassoLars_Model)):
            result = self.reg_model.forecast( Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs, alpha)
        if isinstance(self.reg_model, WAS_ElasticNet_Model): 
            result = self.reg_model.forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs, alpha, l1_ratio)
        if isinstance(self.reg_model, WAS_SVR): 
            result = self.reg_model.forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs,
                                             epsilon, C, kernel_array, degree_array, gamma_array)

        if isinstance(self.reg_model, WAS_RandomForest_XGBoost_ML_Stacking): 
            result = self.reg_model.forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs, best_param_da)       
        
        if isinstance(self.reg_model, WAS_MLP): 
            result = self.reg_model.forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs, hl_array, act_array, lr_array)  
            
        if isinstance(self.reg_model, WAS_RandomForest_XGBoost_Stacking_MLP): 
            result = self.reg_model.forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs, best_param_da)  

        if isinstance(self.reg_model, WAS_Stacking_Ridge): 
            result = self.reg_model.forecast(Predictant, clim_year_start, clim_year_end, Predictor, hindcast_det, Predictor_for_year_pcs, best_param_da)  
        
        if isinstance(self.reg_model, WAS_LogisticRegression_Model): 
            result = None
 
        # if isinstance(self.reg_model, WAS_QuantileRegression_Model): 
        #     result = None
        #if isinstance():
        return result
