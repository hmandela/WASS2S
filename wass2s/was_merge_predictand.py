import warnings
import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from wass2s.utils import *


class WAS_Merging:
    def __init__(
        self,
        df: pd.DataFrame,
        da: xr.Dataset,
        date_month_day: str = "08-01"
    ):

        self.df = df
        self.da = da
        self.date_month_day = date_month_day

    def adjust_duplicates(self, series: pd.Series, increment: float = 0.00001) -> pd.Series:
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series

    def transform_cpt(
        self,
        df: pd.DataFrame,
        missing_value: float = -999.0
    ) -> xr.DataArray:

        # --- 1) Extract metadata: first 2 rows (LAT, LON) ---
        metadata = (
            df
            .iloc[:2]
            .set_index("STATION")  # -> index = ["LAT", "LON"]
            .T
            .reset_index()         # station names in 'index'
        )
        metadata.columns = ["STATION", "LAT", "LON"]
        
        # Adjust duplicates in LAT / LON
        metadata["LAT"] = self.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = self.adjust_duplicates(metadata["LON"])
        
        # --- 2) Extract the data part: from row 2 downward ---
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)
        
        # --- 3) Convert wide -> long ---
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )
        
        # --- 4) Turn YEAR into a date (e.g. YYYY-08-01) ---
        long_data["DATE"] = pd.to_datetime(
            long_data["YEAR"].astype(str) + f"-{self.date_month_day}",
            format="%Y-%m-%d"
        )
        
        # --- 5) Merge with metadata on STATION to attach (LAT, LON) ---
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")
        
        # --- 6) Convert to xarray DataArray ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        # Optional: mask out missing_value
        if missing_value is not None:
            rainfall_data_array = rainfall_data_array.where(rainfall_data_array != missing_value)
            
        return rainfall_data_array

    def transform_cdt(self, df: pd.DataFrame) -> xr.DataArray:

        # --- 1) Extract metadata (first 3 rows: LON, LAT, ELEV) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]
        
        # Adjust duplicates
        metadata["LON"] = self.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = self.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = self.adjust_duplicates(metadata["ELEV"])
        
        # --- 2) Extract actual data from row 3 onward, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})
        
        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")
        
        # --- 3) Merge with metadata to attach (LON, LAT, ELEV) ---
        final_df = pd.merge(data_long, metadata, on="STATION", how="left")
        
        # Ensure 'DATE' is a proper datetime (assuming "YYYYmmdd" format)
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")
        
        # --- 4) Convert to xarray, rename coords ---
        rainfall_data_array = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        
        return rainfall_data_array

    def auto_select_kriging_parameters(
        self,
        df: pd.DataFrame,
        x_col: str = 'X',
        y_col: str = 'Y',
        z_col: str = 'residuals',
        variogram_models: list = None,
        nlags_range=range(3, 10),
        n_splits: int = 5,
        random_state: int = 42,
        verbose: bool = False,
        enable_plotting: bool = False
    ):
        """
        Automatically selects the best variogram_model and nlags for Ordinary Kriging 
        using cross-validation.

        Returns:
         - best_model (str)
         - best_nlags (int)
         - ok_best (OrdinaryKriging object)
         - results_df (pd.DataFrame)
        """
        if variogram_models is None:
            variogram_models = ['linear', 'power', 'gaussian', 'spherical', 'exponential']
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        best_score = np.inf
        best_model = None
        best_nlags = None
        results = []

        for model in variogram_models:
            for nlags in nlags_range:
                cv_errors = []
                for train_index, test_index in kf.split(df):
                    train, test = df.iloc[train_index], df.iloc[test_index]
                    try:
                        ok = OrdinaryKriging(
                            x=train[x_col],
                            y=train[y_col],
                            z=train[z_col],
                            variogram_model=model,
                            nlags=nlags,
                            verbose=False,
                            enable_plotting=False
                        )
                        z_pred, ss = ok.execute('points', test[x_col].values, test[y_col].values)
                        mse = mean_squared_error(test[z_col], z_pred)
                        cv_errors.append(mse)
                    except Exception as e:
                        # E.g., convergence issues
                        cv_errors.append(np.inf)
                        if verbose:
                            print(f"Exception for model {model} with nlags {nlags}: {e}")
                        break
                
                avg_error = np.mean(cv_errors)
                results.append({
                    'variogram_model': model,
                    'nlags': nlags,
                    'cv_mse': avg_error
                })

                if avg_error < best_score:
                    best_score = avg_error
                    best_model = model
                    best_nlags = nlags

                if verbose:
                    print(f"Model: {model}, nlags: {nlags}, CV MSE: {avg_error:.4f}")

        results_df = pd.DataFrame(results).sort_values(by='cv_mse').reset_index(drop=True)

        if verbose:
            print(f"\nBest Variogram Model: {best_model}")
            print(f"Best nlags: {best_nlags}")
            print(f"Best CV MSE: {best_score:.4f}")

        # Fit the best model on the entire dataset
        try:
            ok_best = OrdinaryKriging(
                x=df[x_col],
                y=df[y_col],
                z=df[z_col],
                variogram_model=best_model,
                nlags=best_nlags,
                verbose=verbose,
                enable_plotting=enable_plotting
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fit the best Ordinary Kriging model: {e}")

        return best_model, best_nlags, ok_best, results_df

    def simple_bias_adjustment(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = False
    ) -> (xr.DataArray, pd.DataFrame or None):
        """
        Performs spatial bias adjustment for each time slice. Optionally does
        leave-one-out (LOO) cross-validation to compute RMSE at each time.

        Returns
        -------
        xr.DataArray
            Concatenated bias-adjusted values along the time dimension.
        pd.DataFrame or None
            A DataFrame containing the LOO RMSE for each time if do_cross_validation=True.
            Otherwise, returns None.
        """
        # 1. Read & transform CSV data
        df = self.df # pd.read_csv(self.input_csv_path, sep="\t")
        df_seas = self.transform_cpt(df, missing_value=missing_value)

        # 2. Read gridded NetCDF data
        estim = self.da #xr.open_dataset(self.input_nc_path)
        estim.name = "Estimation"
        mask = estim.mean(dim="T").squeeze()
        mask = xr.where(np.isnan(mask), np.nan, 1)
        
        # 3. Interpolate NetCDF data onto station coordinates
        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
        )
        
        # 4. Align on time
        df_seas['T'] = df_seas['T'].astype('datetime64[ns]')
        estim_stations['T'] = estim_stations['T'].astype('datetime64[ns]')
        estim['T'] = estim['T'].astype('datetime64[ns]')

        df_seas, estim_stations = xr.align(df_seas, estim_stations)

        sp_bias_adj = []
        cv_rmse_records = []

        # 5. Process each time step
        for t_val in df_seas['T'].values:
            # a. Merge data for the same time
            merged_df = pd.merge(
                df_seas.sel(T=t_val).to_dataframe(),
                estim_stations.sel(T=t_val).to_dataframe(),
                on=["T", "Y", "X"],
                how="outer"
            ).reset_index()

            station_var = list(df_seas.data_vars.keys())[0]
            estim_var = "Estimation" #list(estim_stations.data_vars.keys())[0]
            
            # b. Keep only rows with station data
            merged_df = merged_df.dropna(subset=[station_var])
            if merged_df.empty:
                continue

            # c. Compute residuals
            merged_df['residuals'] = merged_df[station_var] - merged_df[estim_var]
            merged_df = merged_df.dropna(subset=['residuals'])
            merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)

            if merged_df['residuals'].isna().all():
                continue

            print(f"\nYear = {t_val}, number in-situ = {merged_df.shape[0]}")

            # d. Auto-select kriging parameters
            best_variogram, best_nlags, ok_model, cv_results = self.auto_select_kriging_parameters(
                merged_df,
                x_col='X',
                y_col='Y',
                z_col='residuals',
                variogram_models=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                nlags_range=range(3, 10),
                n_splits=5,
                random_state=42,
                verbose=False,
                enable_plotting=False
            )

            # LOO cross-validation
            loo_rmse = np.nan
            if do_cross_validation:
                X_data = merged_df['X'].values
                Y_data = merged_df['Y'].values
                Z_data = merged_df['residuals'].values
                n_points = len(merged_df)

                loo_predictions = []
                for j in range(n_points):
                    train_mask = np.ones(n_points, dtype=bool)
                    train_mask[j] = False

                    X_train = X_data[train_mask]
                    Y_train = Y_data[train_mask]
                    Z_train = Z_data[train_mask]

                    ok_loo = OrdinaryKriging(
                        X_train,
                        Y_train,
                        Z_train,
                        variogram_model=best_variogram,
                        nlags=best_nlags
                    )
                    zhat, _ = ok_loo.execute('points', [X_data[j]], [Y_data[j]])
                    loo_predictions.append(zhat[0])

                loo_errors = Z_data - np.array(loo_predictions)
                loo_rmse = np.sqrt(np.nanmean(loo_errors**2))
                print(f"  LOO RMSE for T={t_val}: {loo_rmse:.4f}")

                cv_rmse_records.append({
                    'time': t_val,
                    'LOO_RMSE': loo_rmse,
                    'num_stations': n_points
                })

            # e. Krige residuals over the entire grid
            z_pred, ss = ok_model.execute('grid', estim.X.values, estim.Y.values)
            z_pred_da = xr.DataArray(
                z_pred,
                coords={'Y': estim['Y'], 'X': estim['X']},
                dims=['Y', 'X']
            )
            z_pred_da['T'] = t_val

            # f. Add residual field to the original dataset slice
            if isinstance(estim, xr.Dataset):
                estim_i = estim.sel(T=t_val).to_array().drop_vars('variable')
            else:
                estim_i = estim.sel(T=t_val)
                
            tmp = estim_i + z_pred_da
            tmp = xr.where(tmp < 0, 0, tmp)  # floor negative values at zero
            sp_bias_adj.append(tmp)

        # 6. Concatenate along time dimension
        if sp_bias_adj:
            result = xr.concat(sp_bias_adj, dim="T")
        else:
            result = xr.DataArray()

        if do_cross_validation and cv_rmse_records:
            cv_df = pd.DataFrame(cv_rmse_records)
        else:
            cv_df = None

        return result*mask, cv_df

    def regression_kriging(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = False
    ) -> (xr.DataArray, pd.DataFrame or None):
        """
        Performs spatial bias adjustment for each time slice using:
          1) Linear Regression
          2) Kriging of residuals

        Optionally performs LOO cross-validation for each time.
        """
        df = self.df # pd.read_csv(self.input_csv_path, sep="\t")
        df_seas = self.transform_cpt(df,  missing_value=missing_value)
        
        estim = self.da # xr.open_dataset(self.input_nc_path)
        estim.name = "Estimation"
        mask = estim.mean(dim="T").squeeze()
        mask = xr.where(np.isnan(mask), np.nan, 1)
        
        gridx, gridy = np.meshgrid(estim.X.values, estim.Y.values)

        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
        )

        df_seas['T'] = df_seas['T'].astype('datetime64[ns]')
        estim_stations['T'] = estim_stations['T'].astype('datetime64[ns]')
        estim['T'] = estim['T'].astype('datetime64[ns]')

        df_seas, estim_stations = xr.align(df_seas, estim_stations)

        sp_regress_krig = []
        cv_rmse_records = []

        for t_val in df_seas['T'].values:
            merged_df = pd.merge(
                df_seas.sel(T=t_val).to_dataframe(),
                estim_stations.sel(T=t_val).to_dataframe(),
                on=["T", "Y", "X"],
                how="outer"
            ).reset_index()
            
            station_var = list(df_seas.data_vars.keys())[0]
            estim_var = "Estimation" #list(estim_stations.data_vars.keys())[0]

            # Drop rows with no station observation
            merged_df = merged_df.dropna(subset=[station_var])
            if merged_df.empty:
                continue

            X = merged_df[['X', 'Y', estim_var]].fillna(0)
            y = merged_df[[station_var]].fillna(0)

            # 1) Linear regression
            reg_model = LinearRegression()
            reg_model.fit(X, y)

            merged_df['regression_prediction'] = reg_model.predict(X)
            merged_df['residuals'] = merged_df[station_var] - merged_df['regression_prediction']
            merged_df = merged_df.dropna(subset=['residuals'])
            
            merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if merged_df['residuals'].isna().all():
                continue
            print(f"\nYear = {t_val}, number in-situ = {merged_df.shape[0]}")

            # 2) Auto-select kriging parameters
            best_variogram, best_nlags, ok_model, cv_results = self.auto_select_kriging_parameters(
                merged_df,
                x_col='X',
                y_col='Y',
                z_col='residuals',
                variogram_models=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                nlags_range=range(3, 10),
                n_splits=5,
                random_state=42,
                verbose=False,
                enable_plotting=False
            )

            # Cross-validation on residuals could go here if desired.

            # Krige the residual field
            z_pred, ss = ok_model.execute('grid', estim.X.values, estim.Y.values)

            # Predict from the regression model on the entire grid
            # We assume the "estim_var" is the variable needed for regression
            reg_input = estim.sel(T=t_val).to_dataframe().reset_index()[['X', 'Y',estim_var]].fillna(0)
            regression_pred_grid = reg_model.predict(reg_input)
            regression_pred_grid = regression_pred_grid.reshape(gridx.shape)

            final_prediction_ok = regression_pred_grid + z_pred
            final_prediction_ok = np.where(final_prediction_ok < 0, 0, final_prediction_ok)

            final_prediction_da = xr.DataArray(
                final_prediction_ok,
                coords={'Y': estim['Y'], 'X': estim['X']},
                dims=['Y', 'X']
            )
            final_prediction_da['T'] = t_val

            sp_regress_krig.append(final_prediction_da)

        if sp_regress_krig:
            result = xr.concat(sp_regress_krig, dim="T")
        else:
            result = xr.DataArray()

        if do_cross_validation and cv_rmse_records:
            cv_df = pd.DataFrame(cv_rmse_records)
        else:
            cv_df = None

        return result*mask, cv_df

    def neural_network_kriging_(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = False
    ) -> (xr.DataArray, pd.DataFrame or None):
        """
        Performs spatial bias adjustment for each time slice using:
          1) Neural Network
          2) Kriging of residuals

        Optionally performs LOO cross-validation for each time.
        """
        df = self.df # pd.read_csv(self.input_csv_path, sep="\t")
        df_seas = self.transform_cpt(self.df,  missing_value=missing_value)
        
        estim = self.da # xr.open_dataset(self.input_nc_path)
        estim.name = "Estimation"
        mask = estim.mean(dim="T").squeeze()
        mask = xr.where(np.isnan(mask), np.nan, 1)

        
        
        gridx, gridy = np.meshgrid(estim.X.values, estim.Y.values)

        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
        )
        
        df_seas['T'] = df_seas['T'].astype('datetime64[ns]')
        estim_stations['T'] = estim_stations['T'].astype('datetime64[ns]')
        estim['T'] = estim['T'].astype('datetime64[ns]')

        df_seas, estim_stations = xr.align(df_seas, estim_stations)

        df_seas_ = standardize_timeseries(df_seas)
        estim_stations_ = standardize_timeseries(estim_stations)
        estim_ = standardize_timeseries(estim)
        
        sp_neural_krig = []
        cv_rmse_records = []

        for t_val in df_seas['T'].values:
            merged_df = pd.merge(
                df_seas_.sel(T=t_val).to_dataframe(),
                estim_stations_.sel(T=t_val).to_dataframe(),
                on=["T", "Y", "X"],
                how="outer"
            ).reset_index()
            
            station_var = list(df_seas_.data_vars.keys())[0]
            estim_var = "Estimation" #list(estim_stations.data_vars.keys())[0]

            # Drop rows with no station observation or no grid estimate
            merged_df = merged_df.dropna(subset=[station_var, estim_var])
            
            if merged_df.empty:
                continue

            X_vals = merged_df[['X', 'Y', estim_var]].fillna(0).to_numpy()
            y_vals = merged_df[[station_var]].fillna(0).to_numpy().ravel()

            # Perform hyperparameter tuning for the MLPRegressor
            param_grid = {
                'hidden_layer_sizes': [(5,), (10,), (10, 50), (100, 50), (150, 75)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'max_iter': [5000, 10000],
            }
            base_model = MLPRegressor(random_state=42)
            grid_search = GridSearchCV(base_model, 
                                       param_grid, 
                                       cv=3,
                                       scoring='neg_mean_squared_error',
                                       n_jobs=-1)
            grid_search.fit(X_vals, y_vals)
            nn_model = grid_search.best_estimator_

            # nn_model = MLPRegressor(
            #     hidden_layer_sizes=(100, 50),
            #     activation='relu',
            #     solver='adam',
            #     max_iter=5000,
            #     random_state=42
            # )
            # nn_model.fit(X_vals, y_vals)

            merged_df['neural_prediction'] = nn_model.predict(X_vals)
            merged_df['residuals'] = merged_df[station_var] - merged_df['neural_prediction']
            merged_df = merged_df.dropna(subset=['residuals'])
            merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if merged_df['residuals'].isna().all():
                continue
            print(f"\nYear = {t_val}, number in-situ = {merged_df.shape[0]}")

            best_variogram, best_nlags, ok_model, cv_results = self.auto_select_kriging_parameters(
                merged_df,
                x_col='X',
                y_col='Y',
                z_col='residuals',
                variogram_models=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                nlags_range=range(3, 10),
                n_splits=5,
                random_state=42,
                verbose=False,
                enable_plotting=False
            )

            # (Optional) LOO cross-validation 

            # Predict residuals over the entire grid
            z_pred, ss = ok_model.execute('grid', estim.X.values, estim.Y.values)

            # Predict from the NN model on the entire grid
            nn_input = estim_.sel(T=t_val).to_dataframe().reset_index()[['X', 'Y', estim_var]].fillna(0)
            nn_pred_grid = nn_model.predict(nn_input)
            nn_pred_grid = nn_pred_grid.reshape(gridx.shape)

            # Combine the NN prediction + Kriged residuals
            final_prediction_ok = nn_pred_grid + z_pred
            final_prediction_ok = np.where(final_prediction_ok < 0, 0, final_prediction_ok)
            
            final_prediction_da = xr.DataArray(
                final_prediction_ok,
                coords={'Y': estim['Y'], 'X': estim['X']},
                dims=['Y', 'X']
            )
            final_prediction_da['T'] = t_val

            sp_neural_krig.append(final_prediction_da)

        if sp_neural_krig:
            result = xr.concat(sp_neural_krig, dim="T")
        else:
            result = xr.DataArray()

        result = reverse_standardize(result, estim)

        if do_cross_validation and cv_rmse_records:
            cv_df = pd.DataFrame(cv_rmse_records)
        else:
            cv_df = None

        return result*mask, cv_df

    def neural_network_kriging(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = False
    ) -> (xr.DataArray, pd.DataFrame | None):
        """
        Performs a two-step spatial bias adjustment for each time slice, leveraging:
        
        1) **Neural Network** (MLPRegressor) to capture non-linear relationships between
           the large-scale estimations (predictors) and in-situ observations (predictands).
        2) **Kriging** of the residuals (observation - NN_prediction) to spatially interpolate
           the remaining error structure under the assumption that these residuals are 
           stationary and exhibit spatial autocorrelation.
        
        Parameters
        ----------
        missing_value : float, optional
            Value used to fill missing data in the input station dataset, by default -999.0
        do_cross_validation : bool, optional
            Whether or not to perform cross-validation (e.g., leave-one-out or k-fold) 
            during kriging parameter selection, by default False.
    
        Returns
        -------
        xr.DataArray
            Bias-adjusted field over the domain, with the same spatial dimensions as the 
            original input (and time dimension if applicable).
        pd.DataFrame or None
            If `do_cross_validation=True`, returns a DataFrame with RMSE records from CV; 
            otherwise, None.
    
        Notes
        -----
        - The hyperparameter search for the MLP uses GridSearchCV with MSE-based scoring.
        - Kriging assumes the residuals are reasonably stationary and spatially correlated.
        """
    
        df = self.df  # Station data (already loaded)
        df_seas = self.transform_cpt(self.df, missing_value=missing_value)
    
        # Load or reference the gridded dataset
        estim = self.da
        estim.name = "Estimation"
    
        # Create a mask for valid points (non-NaN across time)
        mask = estim.mean(dim="T").squeeze()
        mask = xr.where(np.isnan(mask), np.nan, 1)
    
        gridx, gridy = np.meshgrid(estim.X.values, estim.Y.values)
    
        # 1) Interpolate the model estimates to the station locations for direct comparison
        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
        )
    
        # Ensure consistent datetime64 dtypes
        df_seas['T'] = df_seas['T'].astype('datetime64[ns]')
        estim_stations['T'] = estim_stations['T'].astype('datetime64[ns]')
        estim['T'] = estim['T'].astype('datetime64[ns]')
    
        # Align station data and interpolated estimates along T
        df_seas, estim_stations = xr.align(df_seas, estim_stations)
    
        # (Optional) Standardize data so that NN training deals with normalized values
        df_seas_ = standardize_timeseries(df_seas)
        estim_stations_ = standardize_timeseries(estim_stations)
        estim_ = standardize_timeseries(estim)
    
        sp_neural_krig = []
        cv_rmse_records = []
    
        # 2) Loop over each time slice
        for t_val in df_seas['T'].values:
            # Merge station data with model-based data for this time
            merged_df = pd.merge(
                df_seas_.sel(T=t_val).to_dataframe(),
                estim_stations_.sel(T=t_val).to_dataframe(),
                on=["T", "Y", "X"],
                how="outer"
            ).reset_index()
    
            station_var = list(df_seas_.data_vars.keys())[0]
            estim_var = "Estimation"
    
            # Drop rows with no station observation or no estimate
            merged_df = merged_df.dropna(subset=[station_var, estim_var])
    
            if merged_df.empty:
                continue
    
            # Prepare inputs and labels for the Neural Network
            X_vals = merged_df[['X', 'Y', estim_var]].fillna(0).to_numpy()
            y_vals = merged_df[[station_var]].fillna(0).to_numpy().ravel()
    
            # 2a) Hyperparameter tuning for the MLPRegressor
            param_grid = {
                'hidden_layer_sizes': [(5,), (10,), (10, 50), (100, 50), (150, 75)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam'],
                'max_iter': [5000, 10000],
            }
            base_model = MLPRegressor(random_state=42)
            grid_search = GridSearchCV(
                base_model, 
                param_grid, 
                cv=3,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            grid_search.fit(X_vals, y_vals)
    
            nn_model = grid_search.best_estimator_
    
            # Compute station residuals = Observed - NN_prediction
            merged_df['neural_prediction'] = nn_model.predict(X_vals)
            merged_df['residuals'] = merged_df[station_var] - merged_df['neural_prediction']
            merged_df = merged_df.dropna(subset=['residuals'])
            merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if merged_df['residuals'].isna().all():
                continue
    
            print(f"\nTime = {t_val}, station count = {merged_df.shape[0]}")
    
            # 2b) Auto-selection of Kriging parameters, 
            #     e.g., best variogram model & nlags via cross-validation
            best_variogram, best_nlags, ok_model, cv_results = self.auto_select_kriging_parameters(
                merged_df,
                x_col='X',
                y_col='Y',
                z_col='residuals',
                variogram_models=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                nlags_range=range(3, 10),
                n_splits=5,
                random_state=42,
                verbose=False,
                enable_plotting=False
            )
    
            # 2c) Krige the residuals over the grid
            z_pred, ss = ok_model.execute('grid', estim.X.values, estim.Y.values)
    
            # Get the NN model's predictions on the full domain
            nn_input = estim_.sel(T=t_val).to_dataframe().reset_index()[['X', 'Y', estim_var]].fillna(0)
            nn_pred_grid = nn_model.predict(nn_input)
            nn_pred_grid = nn_pred_grid.reshape(gridx.shape)
    
            # Combine NN prediction + kriged residual
            final_prediction_ok = nn_pred_grid + z_pred
            final_prediction_ok = np.where(final_prediction_ok < 0, 0, final_prediction_ok)
    
            # Create an xarray DataArray
            final_prediction_da = xr.DataArray(
                final_prediction_ok,
                coords={'Y': estim['Y'], 'X': estim['X']},
                dims=['Y', 'X']
            )
            final_prediction_da['T'] = t_val
    
            sp_neural_krig.append(final_prediction_da)
    
        # 3) Concatenate results over time dimension
        if sp_neural_krig:
            result = xr.concat(sp_neural_krig, dim="T")
        else:
            result = xr.DataArray()
    
        # 4) Reverse the standardization to get back to original scale
        result = reverse_standardize(result, estim)
    
        # 5) If cross-validation was enabled, return records
        if do_cross_validation and cv_rmse_records:
            cv_df = pd.DataFrame(cv_rmse_records)
        else:
            cv_df = None
    
        # Apply mask to keep only valid domain
        return result * mask, cv_df

    
    def multiplicative_bias(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = False
    ) -> (xr.DataArray, pd.DataFrame or None):

        """
        Apply multiplicative bias correction to gridded predictions using ground observations.

        This method performs bias adjustment for each time step by comparing standardized observations
        and model estimates, interpolating residuals using kriging, and applying a multiplicative correction
        across the spatial domain.

        Parameters
        ----------
        missing_value : float, optional
            Value used to represent missing data in the input observational CSV, by default -999.0.
        do_cross_validation : bool, optional
            If True, perform Leave-One-Out Cross-Validation to estimate kriging performance, by default False.

        Returns
        -------
        xr.DataArray
            The bias-corrected spatial dataset over time (with dimensions: T, Y, X), masked over valid areas.
        pd.DataFrame or None
            DataFrame containing LOOCV RMSE per time step, or None if cross-validation was not performed.
        """

        df = self.df # pd.read_csv(self.input_csv_path, sep="\t")
        df_seas = self.transform_cpt(self.df,  missing_value=missing_value)  
        
        estim = self.da # xr.open_dataset(self.input_nc_path)
        mask = estim.mean(dim="T").squeeze()
        mask = xr.where(np.isnan(mask), np.nan, 1)
                
        gridx, gridy = np.meshgrid(estim.X.values, estim.Y.values)

        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
        )

        f_hdcst = []
        [f_hdcst.append((df_seas.sel(T=df_seas['T'] != df_seas['T'].isel(T=i)).mean("T") / estim_stations.sel(T=estim_stations['T'] != estim_stations['T'].isel(T=i)).mean("T"))) for i in range(0,len(estim_stations['T']))]
        
        f_hdcst = xr.concat(f_hdcst, dim="T") 
        f_hdcst['T'] = estim_stations['T']
        
        df_seas['T'] = df_seas['T'].astype('datetime64[ns]')
        f_hdcst['T'] = f_hdcst['T'].astype('datetime64[ns]')
        estim['T'] = estim['T'].astype('datetime64[ns]')


        sp_mult_bias = []
        cv_rmse_records = []

        for t_val in df_seas['T'].values:
            
            merged_df = f_hdcst.sel(T=t_val).to_dataframe().reset_index()
            merged_df = merged_df[~merged_df["Observation"].isna() & np.isfinite(merged_df["Observation"])]

            if merged_df.empty:
                continue

            if merged_df['Observation'].isna().all():
                continue

            print(f"\nYear = {t_val}, number in-situ = {merged_df.shape[0]}")

            # d. Auto-select kriging parameters
            best_variogram, best_nlags, ok_model, cv_results = self.auto_select_kriging_parameters(
                merged_df,
                x_col='X',
                y_col='Y',
                z_col='Observation',
                variogram_models=['linear', 'power', 'gaussian', 'spherical', 'exponential'],
                nlags_range=range(3, 10),
                n_splits=5,
                random_state=42,
                verbose=False,
                enable_plotting=False
            )

            # (Optional) LOO cross-validation
            loo_rmse = np.nan
            if do_cross_validation:
                X_data = merged_df['X'].values
                Y_data = merged_df['Y'].values
                Z_data = merged_df['Observation'].values
                n_points = len(merged_df)

                loo_predictions = []
                for j in range(n_points):
                    train_mask = np.ones(n_points, dtype=bool)
                    train_mask[j] = False

                    X_train = X_data[train_mask]
                    Y_train = Y_data[train_mask]
                    Z_train = Z_data[train_mask]

                    ok_loo = OrdinaryKriging(
                        X_train,
                        Y_train,
                        Z_train,
                        variogram_model=best_variogram,
                        nlags=best_nlags
                    )
                    zhat, _ = ok_loo.execute('points', [X_data[j]], [Y_data[j]])
                    loo_predictions.append(zhat[0])

                loo_errors = Z_data - np.array(loo_predictions)
                loo_rmse = np.sqrt(np.nanmean(loo_errors**2))
                print(f"  LOO RMSE for T={t_val}: {loo_rmse:.4f}")

                cv_rmse_records.append({
                    'time': t_val,
                    'LOO_RMSE': loo_rmse,
                    'num_stations': n_points
                })

            # e. Krige residuals over the entire grid
            z_pred, ss = ok_model.execute('grid', estim.X.values, estim.Y.values)
            z_pred_da = xr.DataArray(
                z_pred,
                coords={'Y': estim['Y'], 'X': estim['X']},
                dims=['Y', 'X']
            )
            z_pred_da['T'] = t_val

            # f. Add residual field to the original dataset slice
            if isinstance(estim, xr.Dataset):
                estim_i = estim.sel(T=t_val).to_array().drop_vars('variable')
            else:
                estim_i = estim.sel(T=t_val)
                
            tmp = estim_i*z_pred_da
            tmp = xr.where(tmp < 0, 0, tmp)  # floor negative values at zero
            sp_mult_bias.append(tmp)

        # 6. Concatenate along time dimension
        if sp_mult_bias:
            result = xr.concat(sp_mult_bias, dim="T")
        else:
            result = xr.DataArray()

        if do_cross_validation and cv_rmse_records:
            cv_df = pd.DataFrame(cv_rmse_records)
        else:
            cv_df = None
        return result*mask, cv_df

    def plot_merging_comparaison(self, df_Obs, da_estimated, da_corrected, missing_value = -999.0):
        
        da_Obs = self.transform_cpt(df_Obs, missing_value=missing_value)
        
        da_estimated_ = da_estimated.interp(
            X=da_Obs.coords['X'],
            Y=da_Obs.coords['Y'],
        )
        da_estimated_.name = "Estimation"
        
        da_corrected_ = da_corrected.interp(
            X=da_Obs.coords['X'],
            Y=da_Obs.coords['Y'],
        )
        da_corrected_.name = "Correction"
            
        merged_df = pd.merge(
            pd.merge(da_Obs.to_dataframe(), 
                     da_estimated_.to_dataframe(), 
                     on=["T", "Y", "X"], 
                     how="outer"),
            da_corrected_.to_dataframe(),
            on=["T", "Y", "X"],
            how="outer"
        ).reset_index()
        
        df = merged_df.dropna(subset=["Observation",  "Estimation", "Correction"])
        
        # Create a 1-row, 2-column figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        
        # Define limits for square axes
        x_min, x_max = df["Observation"].min(), df["Observation"].max()
        y_min, y_max = df["Observation"].min(), df["Observation"].max()
        # y_min = min(df["Estimation"].min(), df["Correction"].min())
        # y_max = max(df["Estimation"].max(), df["Correction"].max())
        
        # Set equal axis limits
        axes[0].set_xlim(x_min, x_max)
        axes[0].set_ylim(y_min, y_max)
        axes[1].set_xlim(x_min, x_max)
        axes[1].set_ylim(y_min, y_max)
        
        # Define y=x reference line range
        y_line = np.linspace(x_min, x_max, 100)
        
        # Panel 1: Scatterplot for Observations vs Estimated
        sns.scatterplot(data=df, x="Observation", y="Estimation", ax=axes[0], color="blue")
        axes[0].plot(y_line, y_line, linestyle="--", color="black", label="y = x")  # y=x line
        axes[0].set_title("Observation vs Estimation")
        axes[0].set_xlabel("Observation")
        axes[0].set_ylabel("Estimation")
        axes[0].set_aspect('equal', adjustable='box')  # Make square
        axes[0].legend()
        
        # Panel 2: Scatterplot for Observations vs Corrected
        sns.scatterplot(data=df, x="Observation", y="Correction", ax=axes[1], color="red")
        axes[1].plot(y_line, y_line, linestyle="--", color="black", label="y = x")  # y=x line
        axes[1].set_title("Observation vs Correction")
        axes[1].set_xlabel("Observation")
        axes[1].set_ylabel("Correction")
        axes[1].set_aspect('equal', adjustable='box')  # Make square
        axes[1].legend()
        
        # Adjust layout
        plt.tight_layout()
        plt.show()

        

import warnings
import numpy as np
import pandas as pd
import xarray as xr
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.spatial.distance import cdist
from scipy.linalg import solve

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging

warnings.filterwarnings("ignore")


class WAS_Merging_:
    """
    Merging class for station observations with gridded estimates,
    using geostatistical and machine learning methods.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        da: xr.DataArray | xr.Dataset,
        dem: xr.DataArray | None = None,
        date_month_day: str = "08-01",
        varname: str | None = None,
        projection: str = "utm",      # "utm" or "laea"
        utm_zone: int | None = None,
        hemisphere: str | None = None,
        # kriging speed
        kriging_cv: bool = True,
        kriging_fixed: dict | None = None,  # {"variogram_model": "spherical", "nlags": 6}
        kriging_cache: bool = True,
        kriging_cache_key: str = "global",  # "global" or "per_time"
        # memory control
        max_grid_points_warn: int = 200_000,
        # verbose
        verbose: bool = True,
    ):
        self.df = df
        self.dem = dem
        self.date_month_day = date_month_day
        self.verbose = verbose

        self.da = self._coerce_to_dataarray(da, varname=varname)
        self._validate_grids()

        self.projection = projection.lower()
        self.utm_zone = utm_zone
        self.hemisphere = hemisphere

        self.kriging_cv = bool(kriging_cv)
        self.kriging_fixed = kriging_fixed
        self.kriging_cache = bool(kriging_cache)
        self.kriging_cache_key = kriging_cache_key

        self.max_grid_points_warn = int(max_grid_points_warn)

        self._to_proj = None
        self._to_geo = None

        self._grid_cache = None
        self._kriging_param_cache = {}
        self.unique_stations = None

        self._init_projection()
        self._cache_projected_grid()

    # --------------------- base guards ---------------------

    def _coerce_to_dataarray(self, da: xr.DataArray | xr.Dataset, varname: str | None) -> xr.DataArray:
        if isinstance(da, xr.DataArray):
            return da
        if not isinstance(da, xr.Dataset):
            raise TypeError("da must be an xarray.DataArray or xarray.Dataset")
        if varname is None:
            if len(da.data_vars) != 1:
                raise ValueError("da has multiple variables; set varname=...")
            varname = list(da.data_vars)[0]
        if varname not in da:
            raise KeyError(f"'{varname}' not found in dataset.")
        return da[varname]

    def _validate_grids(self):
        for d in ("T", "Y", "X"):
            if d not in self.da.dims:
                raise ValueError("da must have dims (T, Y, X) with coords 'T','Y','X'.")
        if self.dem is not None:
            if not isinstance(self.dem, xr.DataArray):
                raise TypeError("dem must be an xarray.DataArray.")
            if ("Y" not in self.dem.dims) or ("X" not in self.dem.dims):
                raise ValueError("dem must have dims (Y, X).")
            try:
                self.dem = self.dem.interp(Y=self.da["Y"], X=self.da["X"])
            except Exception:
                pass

    def _mask_like_estim(self, estim: xr.DataArray) -> xr.DataArray:
        return xr.where(np.isnan(estim.mean(dim="T")), np.nan, 1.0)

    def _expand_T(self, da2d: xr.DataArray, t) -> xr.DataArray:
        return da2d.expand_dims(T=[t])

    def _xy_stack(self, y, x):
        return np.column_stack([np.asarray(y, float), np.asarray(x, float)])


    def adjust_duplicates(self, series: pd.Series, increment: float = 0.00001) -> pd.Series:
        """
        If any values in the Series repeat, nudge them by a tiny increment
        so that all are unique (to avoid indexing collisions).
        """
        counts = series.value_counts()
        for val, count in counts[counts > 1].items():
            duplicates = series[series == val].index
            for i, idx in enumerate(duplicates):
                series.at[idx] += increment * i
        return series


    def transform_cpt(
        self,
        df: pd.DataFrame,
        missing_value: float = -999.0
    ) -> xr.Dataset:
        """
        Convert a CPT‑format station data frame to an xarray Dataset.
        Expects first two rows: station names, then latitudes, then longitudes.
        (The second row is used for LAT, the third for LON – but here we assume
         the input has exactly two metadata rows: first row station names,
         second row LAT, third row LON. The class will handle it as in the first version.)
        """
        # --- 1) Extract metadata: first 2 rows (LAT, LON) ---
        metadata = (
            df
            .iloc[:2]
            .set_index("STATION")  # -> index = ["LAT", "LON"]
            .T
            .reset_index()         # station names in 'index'
        )
        metadata.columns = ["STATION", "LAT", "LON"]

        # Adjust duplicates in LAT / LON
        metadata["LAT"] = self.adjust_duplicates(metadata["LAT"])
        metadata["LON"] = self.adjust_duplicates(metadata["LON"])

        # --- 2) Extract the data part: from row 2 downward ---
        data_part = df.iloc[2:].copy()
        data_part = data_part.rename(columns={"STATION": "YEAR"})
        data_part["YEAR"] = data_part["YEAR"].astype(int)

        # --- 3) Convert wide -> long ---
        long_data = data_part.melt(
            id_vars="YEAR",
            var_name="STATION",
            value_name="VALUE"
        )

        # --- 4) Turn YEAR into a date (e.g. YYYY-08-01) ---
        long_data["DATE"] = pd.to_datetime(
            long_data["YEAR"].astype(str) + f"-{self.date_month_day}",
            format="%Y-%m-%d"
        )

        # --- 5) Merge with metadata on STATION to attach (LAT, LON) ---
        final_df = pd.merge(long_data, metadata, on="STATION", how="left")

        # --- 6) Convert to xarray Dataset (single variable "Observation") ---
        ds = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        # Optional: mask out missing_value
        if missing_value is not None:
            ds = ds.where(ds["Observation"] != missing_value)

        return ds

    def transform_cdt(
        self,
        df: pd.DataFrame,
        missing_value: float = -999.0
    ) -> xr.Dataset:
        """
        Convert a CDT‑format station data frame to an xarray Dataset.
        Expects first three rows: station names, longitudes, latitudes, elevations.
        """
        # --- 1) Extract metadata (first 3 rows: LON, LAT, ELEV) ---
        metadata = df.iloc[:3].set_index("ID").T.reset_index()
        metadata.columns = ["STATION", "LON", "LAT", "ELEV"]

        # Adjust duplicates
        metadata["LON"] = self.adjust_duplicates(metadata["LON"])
        metadata["LAT"] = self.adjust_duplicates(metadata["LAT"])
        metadata["ELEV"] = self.adjust_duplicates(metadata["ELEV"])

        # --- 2) Extract actual data from row 3 onward, rename ID -> DATE ---
        data_part = df.iloc[3:].rename(columns={"ID": "DATE"})

        # Melt to long form
        data_long = data_part.melt(id_vars=["DATE"], var_name="STATION", value_name="VALUE")

        # --- 3) Merge with metadata to attach (LON, LAT, ELEV) ---
        final_df = pd.merge(data_long, metadata, on="STATION", how="left")

        # Ensure 'DATE' is a proper datetime (assuming "YYYYmmdd" format)
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], format="%Y%m%d", errors="coerce")

        # --- 4) Convert to xarray Dataset ---
        ds = (
            final_df[["DATE", "LAT", "LON", "VALUE"]]
            .set_index(["DATE", "LAT", "LON"])
            .to_xarray()
            .rename({"VALUE": "Observation", "LAT": "Y", "LON": "X", "DATE": "T"})
        )
        if missing_value is not None:
            ds = ds.where(ds["Observation"] != missing_value)

        return ds

    # --------------------- projection ---------------------

    def _init_projection(self):
        try:
            from pyproj import CRS, Transformer
        except Exception as e:
            raise ImportError("pyproj is required. Install: pip/conda install pyproj") from e

        lat0 = float(np.nanmean(self.da["Y"].values))
        lon0 = float(np.nanmean(self.da["X"].values))

        if self.projection == "utm":
            zone = self.utm_zone
            if zone is None:
                zone = int(np.floor((lon0 + 180.0) / 6.0) + 1)
                zone = max(1, min(zone, 60))
            hemi = self.hemisphere
            if hemi is None:
                hemi = "north" if lat0 >= 0 else "south"
            epsg = 32600 + zone if hemi == "north" else 32700 + zone
            crs_proj = CRS.from_epsg(epsg)

        elif self.projection == "laea":
            proj4 = f"+proj=laea +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m +no_defs"
            crs_proj = CRS.from_proj4(proj4)

        else:
            raise ValueError("projection must be 'utm' or 'laea'.")

        crs_geo = CRS.from_epsg(4326)
        self._to_proj = Transformer.from_crs(crs_geo, crs_proj, always_xy=True)
        self._to_geo = Transformer.from_crs(crs_proj, crs_geo, always_xy=True)

    def _cache_projected_grid(self):
        lon = self.da["X"].values
        lat = self.da["Y"].values
        gx, gy = np.meshgrid(lon, lat)  # lon2d, lat2d (deg)
        xp, yp = self._to_proj.transform(gx, gy)  # meters
        self._grid_cache = {
            "gx": gx, "gy": gy,
            "xp": xp, "yp": yp,
            "xp_flat": xp.ravel().astype(float),
            "yp_flat": yp.ravel().astype(float),
        }
        npts = self._grid_cache["xp_flat"].size
        if npts > self.max_grid_points_warn:
            warnings.warn(
                f"Grid has {npts:,} points; full distance matrices (OI/Barnes) may be heavy. "
                f"Use chunking options in OI/Barnes if needed.",
                RuntimeWarning
            )

    def _project_points(self, lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        xp, yp = self._to_proj.transform(np.asarray(lon, float), np.asarray(lat, float))
        return np.asarray(xp, float), np.asarray(yp, float)

    # --------------------- station sampling ---------------------
    # obs is a Dataset with variable "Observation"

    def _station_df_for_time(self, obs: xr.Dataset, estim: xr.DataArray, t, add_dem: bool = False) -> pd.DataFrame:
        df = obs.sel(T=t).to_dataframe().reset_index().dropna()
        if df.empty:
            return df

        xp = xr.DataArray(df["X"].values, dims="points")
        yp = xr.DataArray(df["Y"].values, dims="points")
        est_pts = estim.sel(T=t).interp(X=xp, Y=yp)
        df["Estimation"] = np.asarray(est_pts.values, dtype=float)

        if add_dem and (self.dem is not None):
            dem_pts = self.dem.interp(X=xp, Y=yp)
            df["DEM"] = np.asarray(dem_pts.values, dtype=float)

        Xp, Yp = self._project_points(df["X"].values, df["Y"].values)
        df["Xp"] = Xp
        df["Yp"] = Yp

        return df.dropna(subset=["Observation", "Estimation", "Xp", "Yp"])

    # --------------------- kriging selection + cache ---------------------

    def _kriging_cache_id(self, method: str, t=None) -> str:
        if not self.kriging_cache:
            return ""
        if self.kriging_cache_key == "per_time" and t is not None:
            return f"{method}:{pd.to_datetime(t).date()}"
        return f"{method}:global"

    def _fit_ok(self, x, y, z, variogram_model, nlags):
        return OrdinaryKriging(
            x, y, z,
            variogram_model=variogram_model,
            nlags=int(nlags),
            enable_plotting=False,
            verbose=False,
        )

    def auto_select_kriging_parameters(
        self,
        df: pd.DataFrame,
        method_key: str,
        t=None,
        x_col: str = "Xp",
        y_col: str = "Yp",
        z_col: str = "residuals",
        variogram_models=None,
        nlags_range=range(3, 10),
        n_splits=5,
        random_state=42,
    ):
        df = df[[x_col, y_col, z_col]].dropna().copy()
        if len(df) < max(5, n_splits):
            raise ValueError("Not enough points for kriging.")

        x = df[x_col].values.astype(float)
        y = df[y_col].values.astype(float)
        z = df[z_col].values.astype(float)

        # fixed parameters
        if (not self.kriging_cv) and (self.kriging_fixed is not None):
            vm = self.kriging_fixed.get("variogram_model", "spherical")
            nl = int(self.kriging_fixed.get("nlags", 6))
            return self._fit_ok(x, y, z, vm, nl), {"variogram": vm, "nlags": nl, "cv_mse": None}

        # cache hit
        cid = self._kriging_cache_id(method_key, t=t)
        if cid and (cid in self._kriging_param_cache):
            vm, nl = self._kriging_param_cache[cid]
            return self._fit_ok(x, y, z, vm, nl), {"variogram": vm, "nlags": nl, "cv_mse": "cached"}

        if variogram_models is None:
            variogram_models = ["linear", "power", "gaussian", "spherical", "exponential"]

        kf = KFold(n_splits=min(n_splits, len(df)), shuffle=True, random_state=random_state)

        best_score = np.inf
        best_model = None
        best_nlags = None

        for model in variogram_models:
            for nlags in nlags_range:
                errs = []
                for tr, te in kf.split(df):
                    try:
                        ok = self._fit_ok(x[tr], y[tr], z[tr], model, nlags)
                        z_pred, _ = ok.execute("points", x[te], y[te])
                        z_pred = np.asarray(z_pred, float).ravel()
                        errs.append(mean_squared_error(z[te], z_pred))
                    except Exception:
                        errs.append(np.inf)
                mse = float(np.mean(errs))
                if mse < best_score:
                    best_score, best_model, best_nlags = mse, model, int(nlags)

        if cid:
            self._kriging_param_cache[cid] = (best_model, best_nlags)

        ok_best = self._fit_ok(x, y, z, best_model, best_nlags)
        return ok_best, {"variogram": best_model, "nlags": best_nlags, "cv_mse": best_score}

    def _krige_grid_points(self, df_planar: pd.DataFrame, z_col: str, estim: xr.DataArray, method_key: str, t=None):
        ok, info = self.auto_select_kriging_parameters(df_planar, method_key=method_key, t=t, z_col=z_col)
        xp = self._grid_cache["xp_flat"]
        yp = self._grid_cache["yp_flat"]
        z_pred_flat, _ = ok.execute("points", xp, yp)
        z_pred = np.asarray(z_pred_flat, float).reshape(estim.sizes["Y"], estim.sizes["X"])
        return z_pred, info

    # --------------------- merging methods ---------------------
    

    def simple_bias_adjustment(self, missing_value=-999.0):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "SBA", "n_times": 0}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if m_df.empty:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            m_df["residuals"] = m_df["Observation"] - m_df["Estimation"]
            try:
                z_res, info = self._krige_grid_points(m_df, "residuals", estim, method_key="SBA", t=t)
            except Exception:
                continue
            res2d = estim.sel(T=t) + xr.DataArray(z_res, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def regression_kriging(self, missing_value=-999.0, use_dem: bool = False, dem_fill: str = "mask"):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "RK", "n_times": 0, "use_dem": bool(use_dem), "dem_fill": dem_fill}

        dem2d = self.dem if (use_dem and self.dem is not None) else None
        if dem2d is not None and dem_fill == "mean":
            dem2d = dem2d.fillna(float(dem2d.mean().values))

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=use_dem)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            if use_dem and ("DEM" in m_df.columns) and (dem2d is not None):
                if dem_fill == "mean":
                    m_df["DEM"] = m_df["DEM"].fillna(float(np.nanmean(m_df["DEM"].values)))
                else:
                    m_df = m_df.dropna(subset=["DEM"])
                if len(m_df) < 5:
                    continue

                reg = LinearRegression().fit(m_df[["Estimation", "DEM"]].values, m_df["Observation"].values)

                
                estim_t = estim.sel(T=t).values
                dem_t = dem2d.values
                X_grid = np.column_stack([estim_t.ravel(), dem_t.ravel()])
                X_grid = np.nan_to_num(X_grid, nan=0.0)
                trend = reg.predict(X_grid).reshape(estim.sizes["Y"], estim.sizes["X"])

                m_df["residuals"] = m_df["Observation"] - reg.predict(m_df[["Estimation", "DEM"]].values)
            else:
                reg = LinearRegression().fit(m_df[["Estimation"]].values, m_df["Observation"].values)

                estim_t = estim.sel(T=t).values
                estim_t_no_nan = np.nan_to_num(estim_t, nan=0.0)
                trend = reg.predict(estim_t_no_nan.reshape(-1, 1)).reshape(estim.sizes["Y"], estim.sizes["X"])
                m_df["residuals"] = m_df["Observation"] - reg.predict(m_df[["Estimation"]].values)

            try:
                z_res, info = self._krige_grid_points(m_df, "residuals", estim, method_key="RK", t=t)
            except Exception:
                continue

            res2d = xr.DataArray(trend + z_res, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def conditional_merging(self, missing_value=-999.0):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "CM", "n_times": 0}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            try:
                z_obs, _ = self._krige_grid_points(m_df, "Observation", estim, method_key="CM_obs", t=t)
                z_est, _ = self._krige_grid_points(m_df, "Estimation", estim, method_key="CM_est", t=t)
            except Exception:
                continue
            cm_field = z_obs + (estim.sel(T=t).values - z_est)
            res2d = xr.DataArray(cm_field, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def kriging_with_external_drift(
        self,
        missing_value=-999.0,
        variogram_model="linear",
        drift_terms: tuple[str, ...] = ("Estimation",),  # subset of ("Estimation","DEM")
    ):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        if ("DEM" in drift_terms) and (self.dem is None):
            raise ValueError("DEM drift requested but dem is None.")

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []
        diag = {"method": "KED", "n_times": 0, "variogram_model": variogram_model, "drift_terms": drift_terms}

        xp_grid = self._grid_cache["xp_flat"]
        yp_grid = self._grid_cache["yp_flat"]

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=("DEM" in drift_terms))
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            specified_drift = []
            if "Estimation" in drift_terms:
                specified_drift.append(m_df["Estimation"].values.astype(float))
            if "DEM" in drift_terms:
                specified_drift.append(m_df["DEM"].values.astype(float))

            try:
                uk = UniversalKriging(
                    m_df["Xp"].values.astype(float),
                    m_df["Yp"].values.astype(float),
                    m_df["Observation"].values.astype(float),
                    variogram_model=variogram_model,
                    drift_terms=["specified"],
                    specified_drift=specified_drift,
                    enable_plotting=False,
                    verbose=False,
                )

                grid_drifts = []
                if "Estimation" in drift_terms:
                    grid_drifts.append(estim.sel(T=t).values.ravel().astype(float))
                if "DEM" in drift_terms:
                    grid_drifts.append(self.dem.values.ravel().astype(float))

                z_pred_flat, _ = uk.execute(
                    "points",
                    xp_grid, yp_grid,
                    specified_drift_arrays=grid_drifts,
                )
                z_pred = np.asarray(z_pred_flat, dtype=float).reshape(estim.sizes["Y"], estim.sizes["X"])
            except Exception:
                z_pred = estim.sel(T=t).values

            res2d = xr.DataArray(z_pred, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def optimal_interpolation(self, missing_value=-999.0, L_m: float = 150_000.0, epsilon: float = 0.1, chunk_size: int | None = None):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        grid_xy = self._xy_stack(self._grid_cache["yp_flat"], self._grid_cache["xp_flat"])
        t_common = np.intersect1d(obs["T"].values, estim["T"].values)

        out = []
        diag = {"method": "OI", "n_times": 0, "L_m": float(L_m), "epsilon": float(epsilon), "chunk_size": chunk_size}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            stn_xy = self._xy_stack(m_df["Yp"].values, m_df["Xp"].values)
            residuals = (m_df["Observation"].values - m_df["Estimation"].values).astype(float)

            dist_stn = cdist(stn_xy, stn_xy)
            C = np.exp(-(dist_stn**2) / (2.0 * (float(L_m) ** 2))) + float(epsilon) * np.eye(len(stn_xy))

            try:
                w = solve(C, residuals, assume_a="pos")
            except Exception:
                continue

            z_pred_flat = np.empty(grid_xy.shape[0], dtype=float)
            if chunk_size is None:
                dist_grid = cdist(grid_xy, stn_xy)
                Wg = np.exp(-(dist_grid**2) / (2.0 * (float(L_m) ** 2)))
                z_pred_flat[:] = Wg @ w
            else:
                cs = int(chunk_size)
                for i0 in range(0, grid_xy.shape[0], cs):
                    i1 = min(i0 + cs, grid_xy.shape[0])
                    dist_grid = cdist(grid_xy[i0:i1], stn_xy)
                    Wg = np.exp(-(dist_grid**2) / (2.0 * (float(L_m) ** 2)))
                    z_pred_flat[i0:i1] = Wg @ w

            z_pred = z_pred_flat.reshape(estim.sizes["Y"], estim.sizes["X"])
            res2d = estim.sel(T=t) + xr.DataArray(z_pred, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    def barnes_interpolation(self, missing_value=-999.0, kappa_m: float = 200_000.0, gamma: float = 0.3, chunk_size: int | None = None):
        obs = self.transform_cpt(self.df, missing_value)
        estim = self.da
        mask = self._mask_like_estim(estim)

        grid_xy = self._xy_stack(self._grid_cache["yp_flat"], self._grid_cache["xp_flat"])
        t_common = np.intersect1d(obs["T"].values, estim["T"].values)

        out = []
        diag = {"method": "Barnes", "n_times": 0, "kappa_m": float(kappa_m), "gamma": float(gamma), "chunk_size": chunk_size}

        for t in t_common:
            m_df = self._station_df_for_time(obs, estim, t, add_dem=False)
            if len(m_df) < 5:
                continue

            if self.verbose:
                print(f"\nYear = {pd.to_datetime(t).year}, number in-situ = {m_df.shape[0]}")

            stn_xy = self._xy_stack(m_df["Yp"].values, m_df["Xp"].values)
            residuals = (m_df["Observation"].values - m_df["Estimation"].values).astype(float)

            dist_stn = cdist(stn_xy, stn_xy)
            w1s = np.exp(-(dist_stn**2) / (float(kappa_m) ** 2))
            s1s = np.sum(w1s, axis=1)
            s1s[s1s == 0] = 1e-12
            z1s = (w1s @ residuals) / s1s
            new_res = residuals - z1s

            z1_flat = np.empty(grid_xy.shape[0], dtype=float)
            z2_flat = np.empty(grid_xy.shape[0], dtype=float)

            def _barnes_pass(vals, scale_m2):
                outv = np.empty(grid_xy.shape[0], dtype=float)
                if chunk_size is None:
                    d = cdist(grid_xy, stn_xy)
                    w = np.exp(-(d**2) / scale_m2)
                    s = np.sum(w, axis=1)
                    s[s == 0] = 1e-12
                    outv[:] = (w @ vals) / s
                else:
                    cs = int(chunk_size)
                    for i0 in range(0, grid_xy.shape[0], cs):
                        i1 = min(i0 + cs, grid_xy.shape[0])
                        d = cdist(grid_xy[i0:i1], stn_xy)
                        w = np.exp(-(d**2) / scale_m2)
                        s = np.sum(w, axis=1)
                        s[s == 0] = 1e-12
                        outv[i0:i1] = (w @ vals) / s
                return outv

            z1_flat = _barnes_pass(residuals, float(kappa_m) ** 2)
            z2_flat = _barnes_pass(new_res, (float(gamma) * float(kappa_m)) ** 2)

            z_pred = (z1_flat + z2_flat).reshape(estim.sizes["Y"], estim.sizes["X"])
            res2d = estim.sel(T=t) + xr.DataArray(z_pred, coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(np.maximum(res2d, 0), t))
            diag["n_times"] += 1

        if not out:
            return xr.DataArray(np.nan), diag
        return xr.concat(out, dim="T") * mask, diag

    # --------------------- ML (uses projected distances) -------------------------
    # obs is a Dataset

    def _prepare_ml(self, missing_value, k_nearest=10):
        obs = self.transform_cpt(self.df, missing_value)   # Dataset
        estim = self.da

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        rows = []
        for t in t_common:
            df_t = obs.sel(T=t).to_dataframe().reset_index().dropna()
            if df_t.empty:
                continue
            # Add Estimation
            xp = xr.DataArray(df_t["X"].values, dims="points")
            yp = xr.DataArray(df_t["Y"].values, dims="points")
            est_pts = estim.sel(T=t).interp(X=xp, Y=yp)
            df_t["Estimation"] = np.asarray(est_pts.values, dtype=float)

            if self.dem is not None:
                dem_pts = self.dem.interp(X=xp, Y=yp)
                df_t["DEM"] = np.asarray(dem_pts.values, dtype=float)

            Xp, Yp = self._project_points(df_t["X"].values, df_t["Y"].values)
            df_t["Xp"] = Xp
            df_t["Yp"] = Yp

            df_t = df_t.dropna(subset=["Observation", "Estimation", "Xp", "Yp"])
            if not df_t.empty:
                df_t["T"] = t
                rows.append(df_t)

        if not rows:
            raise ValueError("No station samples after alignment.")

        g_df = pd.concat(rows, ignore_index=True)
        stn_unique = g_df[["Yp", "Xp"]].drop_duplicates().reset_index(drop=True)
        self.unique_stations = stn_unique.values.astype(float)

        feats = ["X", "Y", "Estimation"]
        if (self.dem is not None) and ("DEM" in g_df.columns):
            feats.append("DEM")

        obs_xy = g_df[["Yp", "Xp"]].values.astype(float)
        stn_xy = self.unique_stations
        dmat = cdist(obs_xy, stn_xy, metric="euclidean")

        k = int(min(k_nearest, dmat.shape[1]))
        knn = np.partition(dmat, kth=k - 1, axis=1)[:, :k]
        knn.sort(axis=1)

        for i in range(k):
            col = f"DIST_{i+1}"
            g_df[col] = knn[:, i]
            feats.append(col)

        return g_df, feats, estim, obs, k

    def _predict_grid(self, model, estim, obs, feats, k_nearest):
        mask = self._mask_like_estim(estim)

        xp_grid = self._grid_cache["xp_flat"]
        yp_grid = self._grid_cache["yp_flat"]
        grid_xy = np.column_stack([yp_grid, xp_grid]).astype(float)

        d_grid = cdist(grid_xy, self.unique_stations, metric="euclidean")
        k = int(min(k_nearest, d_grid.shape[1]))
        knn_grid = np.partition(d_grid, kth=k - 1, axis=1)[:, :k]
        knn_grid.sort(axis=1)

        t_common = np.intersect1d(obs["T"].values, estim["T"].values)
        out = []

        for t in t_common:
            X_pred = pd.DataFrame(
                {
                    "X": self._grid_cache["gx"].ravel(),
                    "Y": self._grid_cache["gy"].ravel(),
                    "Estimation": estim.sel(T=t).values.ravel(),
                }
            )
            if self.dem is not None:
                X_pred["DEM"] = self.dem.values.ravel()

            for i in range(k):
                X_pred[f"DIST_{i+1}"] = knn_grid[:, i]

            pred = model.predict(X_pred[feats]).reshape(estim.sizes["Y"], estim.sizes["X"])
            da2d = xr.DataArray(np.maximum(pred, 0), coords=[estim["Y"], estim["X"]], dims=["Y", "X"])
            out.append(self._expand_T(da2d, t))

        if not out:
            return xr.DataArray(np.nan)
        return xr.concat(out, dim="T") * mask

    def random_forest_merging(self, missing_value=-999.0, k_nearest=10):
        g_df, feats, estim, obs, k = self._prepare_ml(missing_value, k_nearest=k_nearest)
        model = RandomForestRegressor(
            n_estimators=1200,
            max_features="sqrt",
            min_samples_leaf=3,
            n_jobs=-1,
            random_state=42,
        )
        model.fit(g_df[feats], g_df["Observation"])
        return self._predict_grid(model, estim, obs, feats, k_nearest=k), {"method": "RF", "n_samples": len(g_df), "k_nearest": k}

    def xgboost_merging(self, missing_value=-999.0, tune=True, k_nearest=10):
        g_df, feats, estim, obs, k = self._prepare_ml(missing_value, k_nearest=k_nearest)

        if tune:
            p_grid = {
                "max_depth": [3, 5, 7],
                "learning_rate": [0.02, 0.05, 0.1],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }
            base = xgb.XGBRegressor(
                n_estimators=600,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
            gs = GridSearchCV(base, p_grid, cv=3, n_jobs=-1)
            gs.fit(g_df[feats], g_df["Observation"])
            model = gs.best_estimator_
            info = {"method": "XGB", "best_params": gs.best_params_, "n_samples": len(g_df), "k_nearest": k}
        else:
            model = xgb.XGBRegressor(
                n_estimators=900,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
            )
            model.fit(g_df[feats], g_df["Observation"])
            info = {"method": "XGB", "best_params": None, "n_samples": len(g_df), "k_nearest": k}

        return self._predict_grid(model, estim, obs, feats, k_nearest=k), info

    # --------------------- helper for validation features ---------------------

    def _add_knn_features(self, df, k_nearest, train_stations_only=False):
        """
        Add k‑nearest distance features to a station DataFrame.
        If train_stations_only=True, uses self.unique_stations (already computed from training).
        Otherwise, computes unique stations from the current DataFrame.
        """
        df = df.copy()
        if not train_stations_only:
            # Compute unique stations from this DataFrame (training set)
            stn_unique = df[["Yp", "Xp"]].drop_duplicates().reset_index(drop=True)
            self.unique_stations = stn_unique.values.astype(float)
        else:
            # Use already stored unique stations (from training)
            pass

        obs_xy = df[["Yp", "Xp"]].values.astype(float)
        dmat = cdist(obs_xy, self.unique_stations, metric="euclidean")
        k = int(min(k_nearest, dmat.shape[1]))
        knn = np.partition(dmat, kth=k - 1, axis=1)[:, :k]
        knn.sort(axis=1)

        for i in range(k):
            col = f"DIST_{i+1}"
            df[col] = knn[:, i]
        return df, k

    # --------------------- validation (LOYO) ---------------------

    def validate_merging_methods(self, missing_value=-999.0, min_points_year=5, k_nearest=10):
        """
        Leave-one-year-out validation of all merging methods at station points.

        Parameters
        ----------
        missing_value : float
            Missing value indicator for observations (passed to transform_cpt).
        min_points_year : int
            Minimum number of test points required to include a year.
        k_nearest : int
            Number of nearest stations to use for RF/XGB feature engineering.

        Returns
        -------
        pd.DataFrame
            Mean RMSE per method, sorted ascending.
        """
        obs = self.transform_cpt(self.df, missing_value)   # Dataset
        estim = self.da
        years = np.unique(pd.to_datetime(obs["T"].values).year)

        results = []

        for yr in years:
            is_test = (pd.to_datetime(obs["T"].values).year == yr)
            t_test = obs["T"].values[is_test]
            t_train = obs["T"].values[~is_test]

            if len(t_test) == 0 or len(t_train) == 0:
                continue

            # Collect training and test data frames for this year
            train_rows, test_rows = [], []
            for t in np.intersect1d(t_train, estim["T"].values):
                df_t = self._station_df_for_time(obs, estim, t, add_dem=False)
                if not df_t.empty:
                    train_rows.append(df_t)
            for t in np.intersect1d(t_test, estim["T"].values):
                df_t = self._station_df_for_time(obs, estim, t, add_dem=False)
                if not df_t.empty:
                    test_rows.append(df_t)

            if not train_rows or not test_rows:
                continue

            train_df = pd.concat(train_rows, ignore_index=True)
            test_df = pd.concat(test_rows, ignore_index=True)

            if len(test_df) < min_points_year:
                continue

            # ----- Simple Bias Adjustment (SBA) -----
            try:
                tr = train_df.copy()
                tr["residuals"] = tr["Observation"] - tr["Estimation"]
                ok, _ = self.auto_select_kriging_parameters(tr, method_key="VAL_SBA", z_col="residuals")
                z_pred, _ = ok.execute("points", test_df["Xp"].values, test_df["Yp"].values)
                pred = test_df["Estimation"].values + np.asarray(z_pred, float).ravel()
                rmse = float(np.sqrt(mean_squared_error(test_df["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "SBA", "RMSE": rmse})
            except Exception:
                pass

            # ----- Regression Kriging (RK) -----
            try:
                reg = LinearRegression().fit(train_df[["Estimation"]].values, train_df["Observation"].values)
                tr = train_df.copy()
                tr["residuals"] = tr["Observation"] - reg.predict(tr[["Estimation"]].values)
                ok, _ = self.auto_select_kriging_parameters(tr, method_key="VAL_RK", z_col="residuals")
                z_pred, _ = ok.execute("points", test_df["Xp"].values, test_df["Yp"].values)
                pred = reg.predict(test_df[["Estimation"]].values) + np.asarray(z_pred, float).ravel()
                rmse = float(np.sqrt(mean_squared_error(test_df["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "RK", "RMSE": rmse})
            except Exception:
                pass

            # ----- Conditional Merging (CM) -----
            try:
                ok_obs, _ = self.auto_select_kriging_parameters(train_df, method_key="VAL_CM_obs", z_col="Observation")
                ok_est, _ = self.auto_select_kriging_parameters(train_df, method_key="VAL_CM_est", z_col="Estimation")
                z_obs, _ = ok_obs.execute("points", test_df["Xp"].values, test_df["Yp"].values)
                z_est, _ = ok_est.execute("points", test_df["Xp"].values, test_df["Yp"].values)
                pred = np.asarray(z_obs, float).ravel() + (test_df["Estimation"].values - np.asarray(z_est, float).ravel())
                rmse = float(np.sqrt(mean_squared_error(test_df["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "CM", "RMSE": rmse})
            except Exception:
                pass

            # ----- Kriging with External Drift (KED) -----
            try:
                uk = UniversalKriging(
                    train_df["Xp"].values.astype(float),
                    train_df["Yp"].values.astype(float),
                    train_df["Observation"].values.astype(float),
                    variogram_model="linear",
                    drift_terms=["specified"],
                    specified_drift=[train_df["Estimation"].values.astype(float)],
                    enable_plotting=False,
                    verbose=False,
                )
                z_pred, _ = uk.execute(
                    "points",
                    test_df["Xp"].values.astype(float),
                    test_df["Yp"].values.astype(float),
                    specified_drift_arrays=[test_df["Estimation"].values.astype(float)],
                )
                pred = np.asarray(z_pred, float).ravel()
                rmse = float(np.sqrt(mean_squared_error(test_df["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "KED", "RMSE": rmse})
            except Exception:
                pass

            # ----- Optimal Interpolation (OI) -----
            try:
                stn_xy_train = self._xy_stack(train_df["Yp"].values, train_df["Xp"].values)
                res_train = (train_df["Observation"].values - train_df["Estimation"].values).astype(float)
                dist_stn = cdist(stn_xy_train, stn_xy_train)
                L_m = 150_000.0   # default length scale (meters)
                eps = 0.1
                C = np.exp(-(dist_stn**2) / (2.0 * L_m**2)) + eps * np.eye(len(stn_xy_train))
                w = solve(C, res_train, assume_a="pos")

                stn_xy_test = self._xy_stack(test_df["Yp"].values, test_df["Xp"].values)
                dist_te = cdist(stn_xy_test, stn_xy_train)
                Wg = np.exp(-(dist_te**2) / (2.0 * L_m**2))
                res_pred = Wg @ w
                pred = test_df["Estimation"].values + res_pred
                rmse = float(np.sqrt(mean_squared_error(test_df["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "OI", "RMSE": rmse})
            except Exception:
                pass

            # ----- Barnes Interpolation -----
            try:
                stn_xy_train = self._xy_stack(train_df["Yp"].values, train_df["Xp"].values)
                res_train = (train_df["Observation"].values - train_df["Estimation"].values).astype(float)

                dist_te = cdist(stn_xy_test, stn_xy_train)   # using test points already computed
                kappa = 200_000.0
                gamma = 0.3

                w1 = np.exp(-(dist_te**2) / kappa**2)
                s1 = np.sum(w1, axis=1)
                s1[s1 == 0] = 1e-12
                z1 = (w1 @ res_train) / s1

                dist_stn = cdist(stn_xy_train, stn_xy_train)
                w1s = np.exp(-(dist_stn**2) / kappa**2)
                s1s = np.sum(w1s, axis=1)
                s1s[s1s == 0] = 1e-12
                z1s = (w1s @ res_train) / s1s
                new_res = res_train - z1s

                w2 = np.exp(-(dist_te**2) / (gamma * kappa)**2)
                s2 = np.sum(w2, axis=1)
                s2[s2 == 0] = 1e-12
                z2 = (w2 @ new_res) / s2

                res_pred = z1 + z2
                pred = test_df["Estimation"].values + res_pred
                rmse = float(np.sqrt(mean_squared_error(test_df["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "Barnes", "RMSE": rmse})
            except Exception:
                pass

            # ----- Random Forest (RF) -----
            try:
                train_f, k_used = self._add_knn_features(train_df, k_nearest)
                test_f, _ = self._add_knn_features(test_df, k_nearest, train_stations_only=True)

                feats = ["X", "Y", "Estimation"] + [f"DIST_{i+1}" for i in range(k_used)]

                if self.dem is not None:
                    xp_tr = xr.DataArray(train_f["X"].values, dims="points")
                    yp_tr = xr.DataArray(train_f["Y"].values, dims="points")
                    train_f["DEM"] = np.asarray(self.dem.interp(X=xp_tr, Y=yp_tr).values, float)
                    xp_te = xr.DataArray(test_f["X"].values, dims="points")
                    yp_te = xr.DataArray(test_f["Y"].values, dims="points")
                    test_f["DEM"] = np.asarray(self.dem.interp(X=xp_te, Y=yp_te).values, float)
                    feats.append("DEM")

                rf = RandomForestRegressor(
                    n_estimators=800,
                    max_features="sqrt",
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=42,
                )
                rf.fit(train_f[feats], train_f["Observation"])
                pred = rf.predict(test_f[feats])
                rmse = float(np.sqrt(mean_squared_error(test_f["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "RF", "RMSE": rmse})
            except Exception:
                pass

            # ----- XGBoost (XGB) -----
            try:
                train_f, k_used = self._add_knn_features(train_df, k_nearest)
                test_f, _ = self._add_knn_features(test_df, k_nearest, train_stations_only=True)

                feats = ["X", "Y", "Estimation"] + [f"DIST_{i+1}" for i in range(k_used)]

                if self.dem is not None:
                    xp_tr = xr.DataArray(train_f["X"].values, dims="points")
                    yp_tr = xr.DataArray(train_f["Y"].values, dims="points")
                    train_f["DEM"] = np.asarray(self.dem.interp(X=xp_tr, Y=yp_tr).values, float)
                    xp_te = xr.DataArray(test_f["X"].values, dims="points")
                    yp_te = xr.DataArray(test_f["Y"].values, dims="points")
                    test_f["DEM"] = np.asarray(self.dem.interp(X=xp_te, Y=yp_te).values, float)
                    feats.append("DEM")

                xgb_model = xgb.XGBRegressor(
                    n_estimators=700,
                    learning_rate=0.05,
                    max_depth=5,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                )
                xgb_model.fit(train_f[feats], train_f["Observation"])
                pred = xgb_model.predict(test_f[feats])
                rmse = float(np.sqrt(mean_squared_error(test_f["Observation"], pred)))
                results.append({"Year": int(yr), "Method": "XGB", "RMSE": rmse})
            except Exception:
                pass

        if not results:
            return pd.DataFrame(columns=["Method", "RMSE"])
        return pd.DataFrame(results).groupby("Method")["RMSE"].mean().sort_values()

    # --------------------- plotting ---------------------

    def plot_merging_comparison(self, da_corrected, da_estimated=None, missing_value=-999.0):
        """
        Create a side-by-side scatter plot comparing observations against
        the original estimation and a corrected field.

        Parameters
        ----------
        da_corrected : xr.DataArray
            Corrected field (output from a merging method) with dimensions (T, Y, X).
        da_estimated : xr.DataArray, optional
            Original estimation grid. If None, uses self.da.
        missing_value : float, optional
            Missing value indicator for observations (passed to transform_cpt).
        """
        # Get observation Dataset
        ds_obs = self.transform_cpt(self.df, missing_value=missing_value)   # Dataset

        if da_estimated is None:
            da_estimated = self.da

        # Interpolate estimation and correction to observation points
        da_estimated_interp = da_estimated.interp(X=ds_obs["X"], Y=ds_obs["Y"])
        da_corrected_interp = da_corrected.interp(X=ds_obs["X"], Y=ds_obs["Y"])

        # Convert to DataFrames
        df_obs = ds_obs.to_dataframe().reset_index()   # columns: "Y", "X", "T", "Observation"
        df_est = da_estimated_interp.to_dataframe(name="Estimation").reset_index()
        df_cor = da_corrected_interp.to_dataframe(name="Correction").reset_index()

        merged = pd.merge(pd.merge(df_obs, df_est, on=["T", "Y", "X"]), df_cor, on=["T", "Y", "X"])
        df = merged.dropna(subset=["Observation", "Estimation", "Correction"])

        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Common limits for square axes
        x_min, x_max = df["Observation"].min(), df["Observation"].max()
        y_min, y_max = x_min, x_max  # use same for y to keep square

        for ax in axes:
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal', adjustable='box')

        # y=x reference line
        line_x = np.linspace(x_min, x_max, 100)

        # Left: Observation vs Estimation
        sns.scatterplot(data=df, x="Observation", y="Estimation", ax=axes[0], color="blue", alpha=0.6)
        axes[0].plot(line_x, line_x, 'k--', label='1:1 line')
        axes[0].set_title("Observation vs Estimation")
        axes[0].set_xlabel("Observation")
        axes[0].set_ylabel("Estimation")
        axes[0].legend()

        # Right: Observation vs Correction
        sns.scatterplot(data=df, x="Observation", y="Correction", ax=axes[1], color="red", alpha=0.6)
        axes[1].plot(line_x, line_x, 'k--', label='1:1 line')
        axes[1].set_title("Observation vs Correction")
        axes[1].set_xlabel("Observation")
        axes[1].set_ylabel("Correction")
        axes[1].legend()

        plt.tight_layout()
        plt.show()