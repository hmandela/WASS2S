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

            X = merged_df[[estim_var]].fillna(0)
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
            reg_input = estim.sel(T=t_val).to_dataframe().reset_index()[[estim_var]].fillna(0)
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

            X_vals = merged_df[[estim_var]].fillna(0).to_numpy()
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
            nn_input = estim_.sel(T=t_val).to_dataframe().reset_index()[[estim_var]].fillna(0)
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
    
        **Theoretical Rationale**:
        --------------------------
        - **Neural Network Step**:
          We use a Multi-Layer Perceptron (MLP) to learn non-linear mappings from model estimates
          (e.g., from a climate or weather model) to station observations. The NN effectively
          captures large-scale, non-linear bias patterns that linear methods may miss.
    
        - **Kriging (Geostatistics) Step**:
          After removing the large-scale bias via NN, the residual field often still contains
          spatially correlated local biases or errors. We assume these residuals follow a
          stationary random field. Ordinary Kriging then interpolates these residuals across
          the domain, leveraging a semi-variogram fitted to station residuals. By adding the
          kriged residual field back to the NN predictions, we account for both large-scale
          and local spatial biases.
    
        **Procedure**:
          1. Transform station data (df_seas) and gridded estimates (estim) into a standardized
             space if needed (optional).
          2. For each time slice:
             a. Align station and grid data, dropping NaNs.
             b. Train a Neural Network model (MLPRegressor) on (X=NN_inputs, y=station_obs).
             c. Compute station residuals = (station_obs - nn_prediction).
             d. Fit a variogram to station residuals, then perform Ordinary Kriging on the
                entire domain.
             e. Combine (NN grid prediction) + (kriged residuals) for final bias-adjusted field.
    
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
        - Negative predictions are clamped to zero, assuming a non-negative variable
          (e.g., precipitation).
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
            X_vals = merged_df[[estim_var]].fillna(0).to_numpy()
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
            nn_input = estim_.sel(T=t_val).to_dataframe().reset_index()[[estim_var]].fillna(0)
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

        
