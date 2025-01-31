import warnings
import pandas as pd
import xarray as xr
import numpy as np

from pykrige.ok import OrdinaryKriging
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor


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
        do_cross_validation: bool = True
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
        
        # 3. Interpolate NetCDF data onto station coordinates
        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
            method="nearest"
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
            estim_var = list(estim_stations.data_vars.keys())[0]
            
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

            print(f"\nTime = {t_val}, merged_df shape = {merged_df.shape}")

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

            # (Optional) LOO cross-validation
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
            estim_i = estim.sel(T=t_val).to_array().drop_vars('variable').squeeze()
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

        return result, cv_df

    def regression_kriging(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = True
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
        gridx, gridy = np.meshgrid(estim.X.values, estim.Y.values)

        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
            method="nearest"
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
            estim_var = list(estim_stations.data_vars.keys())[0]

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

            # (Optional) Cross-validation on residuals could go here if desired.

            # Krige the residual field
            z_pred, ss = ok_model.execute('grid', estim.X.values, estim.Y.values)

            # Predict from the regression model on the entire grid
            # We assume the "estim_var" is the variable needed for regression
            reg_input = estim.sel(T=t_val).to_array().squeeze().to_dataframe().reset_index()[[estim_var]].fillna(0)
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

        return result, cv_df

    def neural_network_kriging(
        self,
        missing_value: float = -999.0,
        do_cross_validation: bool = True
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
        gridx, gridy = np.meshgrid(estim.X.values, estim.Y.values)

        estim_stations = estim.interp(
            X=df_seas.coords['X'],
            Y=df_seas.coords['Y'],
            method="nearest"
        )

        df_seas['T'] = df_seas['T'].astype('datetime64[ns]')
        estim_stations['T'] = estim_stations['T'].astype('datetime64[ns]')
        estim['T'] = estim['T'].astype('datetime64[ns]')

        df_seas, estim_stations = xr.align(df_seas, estim_stations)

        sp_neural_krig = []
        cv_rmse_records = []

        for t_val in df_seas['T'].values:
            merged_df = pd.merge(
                df_seas.sel(T=t_val).to_dataframe(),
                estim_stations.sel(T=t_val).to_dataframe(),
                on=["T", "Y", "X"],
                how="outer"
            ).reset_index()
            
            station_var = list(df_seas.data_vars.keys())[0]
            estim_var = list(estim_stations.data_vars.keys())[0]

            # Drop rows with no station observation or no grid estimate
            merged_df = merged_df.dropna(subset=[station_var, estim_var])
            if merged_df.empty:
                continue

            X_vals = merged_df[[estim_var]].fillna(0).to_numpy()
            y_vals = merged_df[[station_var]].fillna(0).to_numpy().ravel()

            nn_model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                max_iter=5000,
                random_state=42
            )
            nn_model.fit(X_vals, y_vals)

            merged_df['neural_prediction'] = nn_model.predict(X_vals)
            merged_df['residuals'] = merged_df[station_var] - merged_df['neural_prediction']
            merged_df = merged_df.dropna(subset=['residuals'])
            merged_df.replace([np.inf, -np.inf], np.nan, inplace=True)
            if merged_df['residuals'].isna().all():
                continue
            print(f"\nTime = {t_val}, merged_df shape = {merged_df.shape}")

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
            nn_input = estim.sel(T=t_val).to_array().squeeze().to_dataframe().reset_index()[[estim_var]].fillna(0)
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

        if do_cross_validation and cv_rmse_records:
            cv_df = pd.DataFrame(cv_rmse_records)
        else:
            cv_df = None

        return result, cv_df
