import numpy as np
import xarray as xr
from scipy.stats import pearsonr, norm, linregress
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc
from sklearn.utils import resample
import calendar
from pathlib import Path
from scipy import stats
from scipy.stats import lognorm
from scipy.stats import gamma
import os
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import properscoring 
import xskillscore as xs
from wass2s.utils import *
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import ListedColormap

class WAS_Verification:
    def __init__(self, dist_method="gamma"):
        """
        Initialize the WAS_Verification class with predefined scoring metrics and their metadata.
        """
        self.scores = {
            "KGE": ("Kling Gupta Efficiency", -1, 1, "det_score", "RdBu_r", self.kling_gupta_efficiency),
            "Pearson": ("Pearson Correlation", -1, 1, "det_score", "RdBu_r", self.pearson_corr),
            "IOA": ("Index Of Agreement", 0, 1, "det_score", "RdBu_r", self.index_of_agreement),
            "MAE": ("Mean Absolute Error", 0, 100, "det_score", "viridis", self.mean_absolute_error),
            "RMSE": ("Root Mean Square Error", 0, 100, "det_score", "viridis", self.root_mean_square_error),
            "NSE": ("Nash Sutcliffe Efficiency", None, 1, "det_score", "RdBu_r", self.nash_sutcliffe_efficiency),
            "TAYLOR_DIAGRAM": ("Taylor Diagram", None, None, "all_grid_det_score", None, self.taylor_diagram),
            "GROC": ("Generalized Discrimination Score", 0, 1, "prob_score", "RdBu_r", self.calculate_groc),
            "RPSS": ("Ranked Probability Skill Score", -1, 1, "prob_score", "RdBu_r", self.calculate_rpss), 
            "IGS": ("Ignorance Score", 0, None, "prob_score", "RdBu", self.ignorance_score),
            "RES": ("Resolution", 0, None, "prob_score", "RdBu_r", self.resolution_score_grid),
            "REL": ("Reliability", None, None, "prob_score", None, self.reliability_score_grid),
            "RELIABILITY_DIAGRAM": ("Reliability Diagram", None, None, "all_grid_prob_score", None, self.reliability_diagram),
            "ROC_CURVE": ("ROC CURVE", None, None, "all_grid_prob_score", None, self.plot_roc_curves),
            "CRPS": ("Continuous Ranked Probability Score with the ensemble distribution", 0, 100, "ensemble_score", "RdBu", self.compute_crps)
        }

        self.dist_method=dist_method

    def get_scores_metadata(self):
        """
        Return the metadata for all scores.
        """
        return self.scores

    # ------------------------
    # Deterministic Metrics
    # ------------------------
    
    def kling_gupta_efficiency(self, y_true, y_pred):
        """Compute Kling-Gupta Efficiency (KGE)."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            r = np.corrcoef(y_true_clean, y_pred_clean)[0, 1]
            alpha = np.std(y_pred_clean) / np.std(y_true_clean)
            beta = np.mean(y_pred_clean) / np.mean(y_true_clean)
            return 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
        else:
            return np.nan

    def pearson_corr(self, y_true, y_pred):
        """Compute Pearson Correlation Coefficient."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask] 
            return pearsonr(y_true_clean, y_pred_clean)[0]
        else:
            return np.nan

    def index_of_agreement(self, y_true, y_pred):
        """Compute Index of Agreement (IOA)."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask] 
            numerator = np.sum((y_pred_clean - y_true_clean)**2)
            denominator = np.sum((np.abs(y_pred_clean - np.mean(y_true_clean)) + 
                                  np.abs(y_true_clean - np.mean(y_true_clean)))**2)
            return 1 - (numerator / denominator)
        else:
            return np.nan

    def mean_absolute_error(self, y_true, y_pred):
        """Compute Mean Absolute Error (MAE)."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]         
            mae = np.mean(np.abs(y_true_clean - y_pred_clean))
            return mae
        else:
            return np.nan

    def root_mean_square_error(self, y_true, y_pred):
        """Compute Root Mean Square Error (RMSE)."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            mse = np.mean((y_true_clean - y_pred_clean) ** 2)
            rmse = np.sqrt(mse)
            return rmse
        else:
            return np.nan

    def nash_sutcliffe_efficiency(self, y_true, y_pred):
        """Compute Nash-Sutcliffe Efficiency (NSE)."""
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_pred_clean = y_pred[mask]
            numerator = np.sum((y_true_clean - y_pred_clean) ** 2)
            denominator = np.sum((y_true_clean - np.mean(y_true_clean)) ** 2)
            nse = 1 - (numerator / denominator)
            return nse
        else:
            return np.nan

    def taylor_diagram(self, y_true, y_pred):
        """Placeholder for Taylor Diagram implementation."""
        # Implement Taylor Diagram as needed
        pass

    def compute_deterministic_score(self, score_func, obs, pred):
        """Apply a deterministic scoring function over xarray DataArrays."""
        obs, pred = xr.align(obs, pred)
        return xr.apply_ufunc(
            score_func,
            obs,
            pred,
            input_core_dims=[('T',), ('T',)],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float'],
            dask_gufunc_kwargs={"allow_rechunk":True},
        )

    # ------------------------
    # Probabilistic Metrics
    # ------------------------
    
    def classify(self, y, index_start, index_end):
        """Classify data into terciles."""
        mask = np.isfinite(y)
        if np.any(mask):
            terciles = np.nanpercentile(y[index_start:index_end], [33, 67])
            y_class = np.digitize(y, bins=terciles, right=True)
            return y_class, terciles[0], terciles[1]
        else:
            return np.full(y.shape[0], np.nan), np.nan, np.nan

    def compute_class(self, Predictant, clim_year_start, clim_year_end):
        """Compute class labels based on terciles."""
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop
        Predictant_class, tercile_33, tercile_67 = xr.apply_ufunc(
            self.classify,
            Predictant,
            input_core_dims=[('T',)],
            kwargs={'index_start': index_start, 'index_end': index_end},
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('T',), (), ()],
            output_dtypes=['float', 'float', 'float']
        )
        return Predictant_class.transpose('T', 'Y', 'X')

    def classify_data_into_terciles(self, y, T1, T2):
        """Alternative method to classify data into terciles."""
        mask = np.isfinite(y)
        if np.any(mask):
            classified_data = np.zeros(y.shape)
            classified_data[y < T1] = 0
            classified_data[y > T2] = 2
            classified_data[(y >= T1) & (y <= T2)] = 1
            return classified_data
        else:
            return np.nan

    def calculate_groc(self, y_true, y_probs, index_start, index_end, n_classes=3):
        """Compute Generalized Receiver Operating Characteristic (GROC)."""
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            terciles = np.nanpercentile(y_true_clean[index_start:index_end], [33, 67])
            y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
            groc = 0.0
            for i in range(n_classes):
                y_true_i = (y_true_clean_class == i).astype(int)
                fpr, tpr, _ = roc_curve(y_true_i, y_probs_clean[i, :])
                groc += auc(fpr, tpr)
            return groc / n_classes
        else:
            return np.nan

    def calculate_rpss__(self, y_true, y_probs, index_start, index_end):
        """Compute Ranked Probability Skill Score (RPSS)."""
        encoder = OneHotEncoder(categories=[np.array([0, 1, 2])], sparse_output=False)
        climatology = np.array([[1/3, 1/3, 1/3]] * y_probs.shape[1])
        mask = np.isfinite(y_true)& np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            terciles = np.nanpercentile(y_true_clean[index_start:index_end], [33, 67])
            y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
            one_hot_encoded_outcomes = encoder.fit_transform(y_true_clean_class.reshape(-1, 1))
            cumulative_forecast = np.cumsum(np.swapaxes(y_probs_clean, 0, 1), axis=1)
            cumulative_reference = np.cumsum(climatology, axis=1)
            cumulative_outcome = np.cumsum(one_hot_encoded_outcomes, axis=1)

            rps_forecast = np.mean(np.sum((cumulative_forecast - cumulative_outcome) ** 2, axis=1))
            rps_reference = np.mean(np.sum((cumulative_reference - cumulative_outcome) ** 2, axis=1))
            return 1 - (rps_forecast / rps_reference)
        else:
            return np.nan
    def calculate_rpss(self, y_true, y_probs, index_start, index_end):
        """Compute Ranked Probability Skill Score (RPSS)."""
        encoder = OneHotEncoder(categories=[np.array([0, 1, 2])], sparse_output=False)
    
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            # Compute thresholds on clean subset
            terciles = np.nanpercentile(y_true_clean[index_start:index_end], [33, 67])
            y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
            # One-hot encoding
            one_hot_encoded_outcomes = encoder.fit_transform(y_true_clean_class.reshape(-1, 1))
            # Cumulative versions
            cumulative_forecast = np.cumsum(np.swapaxes(y_probs_clean, 0, 1), axis=1)
            cumulative_outcome = np.cumsum(one_hot_encoded_outcomes, axis=1)
            # Climatology matched to outcome shape
            climatology = np.full_like(one_hot_encoded_outcomes, 1/3)
            cumulative_reference = np.cumsum(climatology, axis=1)
            # Compute RPS and RPSS
            rps_forecast = np.mean(np.sum((cumulative_forecast - cumulative_outcome) ** 2, axis=1))
            rps_reference = np.mean(np.sum((cumulative_reference - cumulative_outcome) ** 2, axis=1))
            return 1 - (rps_forecast / rps_reference)
        else:
            return np.nan

    
    def ignorance_score(self, y_true, y_probs, index_start, index_end):
        """Compute Ignorance Score using (Weijs, 2010)."""
        mask = np.isfinite(y_true)& np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
            terciles = np.nanpercentile(y_true_clean[index_start:index_end], [33, 67])
            y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
    
            n = y_true_clean.shape[1]
            ignorance_sum = 0.0
            for i in range(n):
                y_true_clean_category = int(y_true_clean[i])
                prob = y_probs_clean[y_true_clean_category, i]
                if prob > 0:
                    ignorance_sum += -np.log2(prob)
                else:
                    ignorance_sum += np.nan  # Handle log(0) as NaN
            return ignorance_sum / n if n > 0 else np.nan 
        else:
            return np.nan 

    def resolution_score_grid(self, y_true, y_probs, index_start, index_end, 
                              bins=np.array([0.000, 0.025, 0.050, 0.100, 0.150, 0.200, 
                                             0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 
                                             0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 
                                             0.850, 0.900, 0.950, 0.975, 1.000])):
        """Compute Resolution Score on Grid using (Weijs, 2010)."""
        mask = np.isfinite(y_true)& np.isfinite(y_probs).all(axis=0)
        if np.any(mask) and np.sum(mask) > 2:
            return np.nan
        y_true_clean = y_true[mask]
        y_probs_clean = y_probs[:, mask]
        terciles = np.nanpercentile(y_true_clean[index_start:index_end], [33, 67])
        y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
    
        n_categories, n_instances = y_probs_clean.shape
        
        # Calculate the overall observed relative frequency for each category
        y_bar = [np.mean(y_true_clean_class == k) for k in range(n_categories)]
        
        # Initialize the total resolution score
        resolution_sum = 0.0
        
        # Loop over each category
        for k in range(n_categories):
            # Get the forecast probabilities for category k
            y_probs_clean_k = y_probs_clean[k, :]
            
            # Bin the forecast probabilities for category k using the custom bins
            binned_indices = np.digitize(y_probs_clean_k, bins) - 1  # Bin index for each forecast probability
            
            # Calculate the resolution for each bin
            for b in range(len(bins) - 1):
                # Mask for forecasts in the current bin
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)  # Number of forecasts in this bin
                
                if n_kb > 0:
                    # Calculate y_k (observed relative frequency for the bin)
                    y_kb = np.mean(y_true_clean_class[bin_mask] == k)
                    # Compute the resolution components for this bin, avoiding log(0)
                    if y_kb > 0 and y_kb < 1:
                        term1 = y_kb * np.log2(y_kb / y_bar[k]) if y_bar[k] > 0 else 0
                        term2 = (1 - y_kb) * np.log2((1 - y_kb) / (1 - y_bar[k])) if y_bar[k] < 1 else 0
                        resolution_sum += (n_kb / n_instances) * (term1 + term2)
        # Calculate the maximum possible resolution value for each category and take the average
        max_resolution = np.mean([-np.log2(y_bar[k] * (1 - y_bar[k])) if 0 < y_bar[k] < 1 else 0 for k in range(n_categories)])
        return resolution_sum #, max_resolution

    def reliability_score_grid(self, y_true, y_probs, index_start, index_end,
                               bins=np.array([0.000, 0.025, 0.050, 0.100,
                                              0.150, 0.200, 0.250, 0.300,
                                              0.350, 0.400, 0.450, 0.500,
                                              0.550, 0.600, 0.650, 0.700,
                                              0.750, 0.800, 0.850, 0.900,
                                              0.950, 0.975, 1.000])):
        """Compute Reliability Score on Grid using (Weijs, 2010)."""
        mask = np.isfinite(y_true)& np.isfinite(y_probs).all(axis=0)
        if not np.any(mask):
            return np.nan
        y_true_clean = y_true[mask]
        y_probs_clean = y_probs[:, mask]
        terciles = np.nanpercentile(y_true_clean[index_start:index_end], [33, 67])
        y_true_clean_class = np.digitize(y_true_clean, bins=terciles, right=True)
                
        n_categories, n_instances = y_probs_clean.shape
        reliability_sum = 0.0
    
        # Loop over each category
        for k in range(n_categories):
            # Get the forecast probabilities for category k
            y_probs_clean_k = y_probs_clean[k, :]
    
            # Bin the forecast probabilities for category k using the custom bins
            binned_indices = np.digitize(y_probs_clean_k, bins) - 1  # Bin index for each forecast probability
    
            # Calculate reliability for each bin
            for b in range(len(bins) - 1):
                # Mask for forecasts in the current bin
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)  # Number of forecasts in this bin
    
                if n_kb > 0:
                    # Calculate y_k (observed relative frequency for the bin)
                    y_kb = np.mean(y_true_clean_class[bin_mask] == k)
                    # Average forecast probability for the bin
                    p_kb = np.mean(y_probs_clean_k[bin_mask])
    
                    # Compute the reliability components for this bin, avoiding log(0)
                    if y_kb > 0 and y_kb < 1 and p_kb > 0 and p_kb < 1:
                        term1 = y_kb * np.log2(y_kb / p_kb) if p_kb > 0 else 0
                        term2 = (1 - y_kb) * np.log2((1 - y_kb) / (1 - p_kb)) if p_kb < 1 else 0
                        reliability_sum += (n_kb / n_instances) * (term1 + term2)
    
        return reliability_sum

    def resolution_and_reliability_over_all_grid(self, dir_to_save_score, y_true, y_probs, clim_year_start, clim_year_end,
                                                 bins=np.array([0.000, 0.025, 0.050, 0.100, 0.150, 0.200, 
                                                                0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 
                                                                0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 
                                                                0.850, 0.900, 0.950, 0.975, 1.000])):
        """
        Compute both Resolution and Reliability scores over all grid points using (Weijs, 2010).
        Reference: https://web.unbc.ca/~chengy/reliability.html
        """
        y_true, y_probs = xr.align(y_true, y_probs)
        y_true_class = self.compute_class(y_true, clim_year_start, clim_year_end)
        observed_outcomes = y_true_class.stack(flat_dim=("T", "Y", "X")).values
        predicted_probs = y_probs.stack(flat_dim=("T", "Y", "X")).values
        mask = ~np.isnan(observed_outcomes)& np.isfinite(predicted_probs).all(axis=0)
        observed_classes = observed_outcomes[mask]
        predicted_probabilities = predicted_probs[:, mask]
        
        #### Resolution
        n_categories, n_instances = predicted_probabilities.shape
        y_bar = [np.mean(observed_classes == k) for k in range(n_categories)]
        resolution_sum = 0.0
        
        for k in range(n_categories):
            y_probs_clean_k = predicted_probabilities[k, :]
            binned_indices = np.digitize(y_probs_clean_k, bins) - 1
            for b in range(len(bins) - 1):
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)
                if n_kb > 0:
                    y_kb = np.mean(observed_classes[bin_mask] == k)
                    if y_kb > 0 and y_kb < 1:
                        term1 = y_kb * np.log2(y_kb / y_bar[k]) if y_bar[k] > 0 else 0
                        term2 = (1 - y_kb) * np.log2((1 - y_kb) / (1 - y_bar[k])) if y_bar[k] < 1 else 0
                        resolution_sum += (n_kb / n_instances) * (term1 + term2)
    
        #### Reliability
        reliability_sum = 0.0
        for k in range(n_categories):
            y_probs_clean_k = predicted_probabilities[k, :]
            binned_indices = np.digitize(y_probs_clean_k, bins) - 1
            for b in range(len(bins) - 1):
                bin_mask = binned_indices == b
                n_kb = np.sum(bin_mask)
                if n_kb > 0:
                    y_kb = np.mean(observed_classes[bin_mask] == k)
                    p_kb = np.mean(y_probs_clean_k[bin_mask])
                    if y_kb > 0 and y_kb < 1 and p_kb > 0 and p_kb < 1:
                        term1 = y_kb * np.log2(y_kb / p_kb) if p_kb > 0 else 0
                        term2 = (1 - y_kb) * np.log2((1 - y_kb) / (1 - p_kb)) if p_kb < 1 else 0
                        reliability_sum += (n_kb / n_instances) * (term1 + term2)
    
        return resolution_sum, reliability_sum

    def reliability_diagram(self, modelname, dir_to_save_score,  y_true, y_probs, clim_year_start, clim_year_end,
                            bins=np.array([0.100, 0.150, 0.200, 
                                           0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 
                                           0.550, 0.600, 0.650, 0.700])):
        """Plot Reliability Diagrams."""
        labels = ["Below-normal", "Near-normal", "Above-normal"]
        y_true, y_probs = xr.align(y_true, y_probs)
        y_true_class = self.compute_class(y_true, clim_year_start, clim_year_end)
        observed_outcomes = y_true_class.stack(flat_dim=("T", "Y", "X")).values
        predicted_probs = y_probs.stack(flat_dim=("T", "Y", "X")).values
        mask = ~np.isnan(observed_outcomes) & np.isfinite(predicted_probs).all(axis=0)
        observed_classes = observed_outcomes[mask]
        predicted_probabilities = predicted_probs[:, mask]
    
        resolution, reliability = self.resolution_and_reliability_over_all_grid(dir_to_save_score,
            y_true, y_probs, clim_year_start, clim_year_end, bins
        )
    
        n_bins = len(bins) - 1
        observed_freqs = {0: np.zeros(n_bins), 1: np.zeros(n_bins), 2: np.zeros(n_bins)}
        forecast_counts = {0: np.zeros(n_bins), 1: np.zeros(n_bins), 2: np.zeros(n_bins)}
    
        for i in range(predicted_probabilities.shape[1]):
            for tercile in range(3):
                prob = predicted_probabilities[tercile, i]
                obs = observed_classes[i]
            
                if np.isnan(prob) or np.isnan(obs):
                    continue  # Skip missing values
                
                bin_index = np.digitize(prob, bins) - 1
                bin_index = min(bin_index, n_bins - 1)
                
                forecast_counts[tercile][bin_index] += 1
                
                if obs == tercile:
                    observed_freqs[tercile][bin_index] += 1
    
        for tercile in range(3):
            observed_freqs[tercile] = np.divide(
                observed_freqs[tercile],
                forecast_counts[tercile],
                out=np.zeros_like(observed_freqs[tercile]),
                where=forecast_counts[tercile] != 0
            )
    
        # Plot Reliability Diagrams
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        titles = ["Below-normal", "Near-normal", "Above-normal"]
    
        for idx, tercile in enumerate(range(3)):
            ax = axs[idx]
            ax.plot(bins[:-1] * 100, observed_freqs[tercile] * 100, 'k-', lw=2, color="black", label="Reliability Curve")
            
            # Calculate and plot least squares regression
            non_zero_mask = forecast_counts[tercile] > 0
            if np.any(non_zero_mask):
                slope, intercept, _, _, _ = linregress(bins[:-1][non_zero_mask], observed_freqs[tercile][non_zero_mask])
                ax.plot(bins[:-1] * 100, (slope * bins[:-1] + intercept) * 100, 'k--', color="black", lw=1, label="Regression Fit")
            
            # Perfect reliability line
            ax.plot([0, 100], [0, 100], 'r:', color="red", lw=1.5, label="Perfect Reliability")
            
            # Add relative frequency line (horizontal and vertical dashed lines)
            total_observed = np.sum(observed_classes == tercile)
            total_forecasts = len(observed_classes)
            relative_frequency = (total_observed / total_forecasts) * 100
            ax.axhline(relative_frequency, linestyle='--', color="blue", lw=0.8, label="Relative Frequency")
            ax.axvline(relative_frequency, linestyle='--', color="blue", lw=0.8)
    
            # Add no skill line
            no_skill_x = np.linspace(0, 100, 100)
            no_skill_y = 0.5 * no_skill_x + relative_frequency / 2  
            ax.plot(no_skill_x, no_skill_y, 'b--', color="orange", lw=2, label="No Skill Line")
    
            # Shade areas between no skill line and vertical line at relative frequency
            ax.fill_between(no_skill_x, no_skill_y, 0, where=(no_skill_x <= relative_frequency),
                            color='gray', alpha=0.2)
            ax.fill_between(no_skill_x, no_skill_y, 100, where=(no_skill_x >= relative_frequency),
                            color='gray', alpha=0.2)
    
            # Annotate Reliability and Resolution scores
            ax.text(0.05, 0.78, f"REL: {reliability:.2f}", transform=ax.transAxes, fontsize=10)
            ax.text(0.05, 0.71, f"RES: {resolution:.2f}", transform=ax.transAxes, fontsize=10)
            
            # Histogram of forecast probabilities
            ax.bar(bins[:-1] * 100, (forecast_counts[tercile] / total_forecasts) * 100, width=5, color="grey", alpha=0.5, align="edge")
            
            ax.set_title(titles[tercile])
            ax.set_xlim([0, 100])
            ax.set_ylim([0, 100])
            ax.set_xlabel("Forecast Probability (%)")
            ax.set_ylabel("Observed Relative Frequency (%)")
            # ax.legend(loc="upper left")
        
        fig.suptitle("Reliability Diagrams", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{dir_to_save_score}/RELIABILITY_{modelname}_.png", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_roc_curves(self, modelname, dir_to_save_score, y_true, y_probs, clim_year_start, clim_year_end, 
                        n_bootstraps=200, ci=0.95):
        """Plot ROC Curves with Confidence Intervals for probabilistic forecasts."""
        labels = ["Below-normal", "Near-normal", "Above-normal"]
        y_true, y_probs = xr.align(y_true, y_probs)
        y_true_class = self.compute_class(y_true, clim_year_start, clim_year_end)
        
        observed_outcomes = y_true_class.stack(flat_dim=("T", "Y", "X")).values
        predicted_probs = y_probs.stack(flat_dim=("T", "Y", "X")).values
        mask = ~np.isnan(observed_outcomes)& np.isfinite(predicted_probs).all(axis=0)
        observed_outcomes = observed_outcomes[mask]
        predicted_probs = predicted_probs[:, mask]
    
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
        for i, ax in enumerate(axes):
            binary_labels = (observed_outcomes == i).astype(int)
            fpr, tpr, _ = roc_curve(binary_labels, predicted_probs[i, :])
            roc_auc = auc(fpr, tpr)
            
            # Bootstrap resampling to calculate confidence intervals
            tprs_bootstrap = []
            mean_fpr = np.linspace(0, 1, 100)
            
            for _ in range(n_bootstraps):
                indices = resample(np.arange(len(predicted_probs[i])), replace=True)
                boot_binary_labels = binary_labels[indices]
                boot_predicted_probs = predicted_probs[i, indices]
                
                if np.unique(boot_binary_labels).size < 2:
                    continue  # Skip if only one class present in the bootstrap sample
                
                boot_fpr, boot_tpr, _ = roc_curve(boot_binary_labels, boot_predicted_probs)
                interp_tpr = np.interp(mean_fpr, boot_fpr, boot_tpr)
                interp_tpr[0] = 0.0
                tprs_bootstrap.append(interp_tpr)
            
            if not tprs_bootstrap:
                print(f"No valid bootstrap samples for {labels[i]} category.")
                continue
            
            tprs_bootstrap = np.array(tprs_bootstrap)
            mean_tpr = tprs_bootstrap.mean(axis=0)
            mean_tpr[-1] = 1.0
            lower_tpr = np.percentile(tprs_bootstrap, (1 - ci) / 2 * 100, axis=0)
            upper_tpr = np.percentile(tprs_bootstrap, (1 + ci) / 2 * 100, axis=0)
            
            # Plot the mean ROC curve and confidence interval
            ax.plot(fpr, tpr, color='red', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax.fill_between(mean_fpr, lower_tpr, upper_tpr, color='red', alpha=0.2, 
                            label=f'{int(ci*100)}% CI')
            ax.plot([0, 1], [0, 1], 'k--', lw=2, label='No Skill')
            
            # Enhance subplot aesthetics
            for spine in ax.spines.values():
                spine.set_edgecolor('black')
                spine.set_linewidth(1.5)
                spine.set_visible(True)
            
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_title(f'ROC Curve for {labels[i]} Category')
            ax.set_xlabel('False Positive Rate')
            if i == 0:
                ax.set_ylabel('True Positive Rate')
            ax.legend(loc="lower right")
            ax.grid(True)
        fig.suptitle(f"ROC Curves for {modelname}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(f"{dir_to_save_score}/ROC_{modelname}_.png", dpi=300, bbox_inches='tight')
        plt.show()

    # ------------------------
    # Ensemble Metrics
    # ------------------------
    
    def compute_crps(self, y_true, y_pred, member_dim='number', dim="T"):
        """Compute Continuous Ranked Probability Score (CRPS)."""
        y_true, y_pred = xr.align(y_true, y_pred)
        return xs.crps_ensemble(y_true, y_pred, member_dim=member_dim, dim=dim)

    # ------------------------
    # Probabilistic Scoring
    # ------------------------
    
    def compute_probabilistic_score(self, score_func, obs, prob_pred, clim_year_start, clim_year_end):
        """Apply a probabilistic scoring function over xarray DataArrays."""
        index_start = obs.get_index("T").get_loc(str(clim_year_start)).start
        index_end = obs.get_index("T").get_loc(str(clim_year_end)).stop
        obs, prob_pred = xr.align(obs, prob_pred)
        return xr.apply_ufunc(
            score_func,
            obs,
            prob_pred,
            input_core_dims=[('T',), ('probability', 'T')],
            vectorize=True,
            kwargs={'index_start': index_start, 'index_end': index_end},
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float'],
            dask_gufunc_kwargs={"allow_rechunk":True},
        )

    # ------------------------
    # Plotting Utilities
    # ------------------------
    
    def plot_model_score(self, model_metric, score, dir_save_score, figure_name="WAS_MLR"):
        """Plot deterministic scores on a map."""
        dir_save_score = Path(dir_save_score)
        dir_save_score.mkdir(parents=True, exist_ok=True)
        score_meta_data = self.scores[score]
        
        fig, ax = plt.subplots(figsize=(6, 4), subplot_kw={'projection': ccrs.PlateCarree()})
        # Plot the data
        im = model_metric.plot(
            ax=ax, 
            transform=ccrs.PlateCarree(), 
            cmap=score_meta_data[4], 
            vmin=score_meta_data[1] if score_meta_data[1] is not None else None,
            vmax=score_meta_data[2] if score_meta_data[2] is not None else None,
            add_colorbar=False
        )
        # Add coastlines and borders for better visualization
        ax.coastlines(resolution='10m')
        ax.add_feature(cfeature.BORDERS, linestyle='--')
        
        # Add a title
        ax.set_title(f"{figure_name} {score_meta_data[0]}")
        
        # Optionally add gridlines
        gl = ax.gridlines(draw_labels=True, linewidth=0.1, color='gray', alpha=0.5)
        gl.top_labels = False
        gl.right_labels = False
        
        # Create a colorbar that matches the width of the map
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.5)
        # Adjust the colorbar to match the map's extent
        cbar.ax.set_position([ax.get_position().x0, ax.get_position().y0 - 0.1, 
                          ax.get_position().width, 0.039])
        cbar.set_label(score_meta_data[0])
                
        # Save the figure
        plt.savefig(dir_save_score / f"{figure_name}_{score}.png", dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
        plt.close() 

    def plot_models_score(self, model_metrics, score, dir_save_score):
        """Plot multiple model scores on a grid of maps."""
        dir_save_score = Path(dir_save_score)
        dir_save_score.mkdir(parents=True, exist_ok=True)
        score_meta_data = self.scores[score]
        
        # Determine the number of subplots
        n_scores = len(model_metrics)
        n_cols = min(3, n_scores)  # Maximum of 3 columns
        n_rows = int(np.ceil(n_scores / n_cols))
        
        # Create a single figure with a grid of subplots
        fig, axes = plt.subplots(
            n_rows, n_cols, 
            figsize=(n_cols * 6, n_rows * 4),
            subplot_kw={'projection': ccrs.PlateCarree()}
        )
        axes = axes.flatten()  # Flatten to iterate easily
        
        for i, (center, data) in enumerate(model_metrics.items()):
            ax = axes[i]
            # Plot each score on its subplot
            im = data.plot(
                ax=ax, 
                transform=ccrs.PlateCarree(), 
                cmap=score_meta_data[4], 
                vmin=score_meta_data[1] if score_meta_data[1] is not None else None,
                vmax=score_meta_data[2] if score_meta_data[2] is not None else None,
                add_colorbar=False
            )
            
            ax.coastlines(resolution='10m')
            ax.add_feature(cfeature.BORDERS, linestyle='--')
            ax.set_title(f"{center} {score_meta_data[0]}")
            
            gl = ax.gridlines(draw_labels=True, linewidth=0.1, color='gray', alpha=0.5)
            gl.top_labels = False
            gl.right_labels = False

            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.15, shrink=0.7)
            cbar.ax.set_position([ax.get_position().x0, ax.get_position().y0 - 0.12,
                              ax.get_position().width, 0.03])
            cbar.set_label(score_meta_data[0])
    
        # Remove any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
            
        # Adjust layout to reduce separation between subplots
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        
        # Save the figure
        plt.savefig(dir_save_score / f"{score}_all_models.png", dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

    # ------------------------
    # GCM Validation
    # ------------------------

    @staticmethod
    def calculate_tercile_probabilities(best_guess, error_variance, first_tercile, second_tercile, dof):
        """
        Student's t-based method
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))

        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            # Transform thresholds
            first_t = (first_tercile - best_guess) / error_std
            second_t = (second_tercile - best_guess) / error_std

            pred_prob[0, :] = stats.t.cdf(first_t, df=dof)
            pred_prob[1, :] = stats.t.cdf(second_t, df=dof) - stats.t.cdf(first_t, df=dof)
            pred_prob[2, :] = 1 - stats.t.cdf(second_t, df=dof)

        return pred_prob
        
    @staticmethod
    def calculate_tercile_probabilities_weibull_min(best_guess, error_variance, first_tercile, second_tercile, dof):
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
    
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
    
            # Using the weibull_min CDF with best_guess as loc and error_std as scale.
            # Note: Adjust these assumptions if your application requires a different parameterization.
            pred_prob[0, :] = stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std) - \
                               stats.weibull_min.cdf(first_tercile, c=dof, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.weibull_min.cdf(second_tercile, c=dof, loc=best_guess, scale=error_std)
    
        return pred_prob
    @staticmethod
    def calculate_tercile_probabilities_gamma(best_guess, error_variance, T1, T2):
        """
        Gamma-based method
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time), dtype=float)

        # If any input is NaN, fill with NaN
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob

        # Convert inputs to arrays
        best_guess = np.asarray(best_guess, dtype=float)
        error_variance = np.asarray(error_variance, dtype=float)
        T1 = np.asarray(T1, dtype=float)
        T2 = np.asarray(T2, dtype=float)

        alpha = (best_guess**2) / error_variance
        theta = error_variance / best_guess
    
        # Compute CDF at T1, T2
        cdf_t1 = gamma.cdf(T1, a=alpha, scale=theta)
        cdf_t2 = gamma.cdf(T2, a=alpha, scale=theta)
    
        pred_prob[0, :] = cdf_t1
        pred_prob[1, :] = cdf_t2 - cdf_t1
        pred_prob[2, :] = 1.0 - cdf_t2

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_nonparametric(best_guess, error_samples, first_tercile, second_tercile):
        """
        Non-parametric method (require historical errors)
        """
        # best_guess: shape (n_time,)
        # error_samples: shape (n_time,)
        n_time = len(best_guess)
        pred_prob = np.full((3, n_time), np.nan, dtype=float)

        for t in range(n_time):
            if np.isnan(best_guess[t]):
                continue

            # Empirical distribution = best_guess[t] + error_samples[:, t] ---- to see in deep again
            dist = best_guess[t] + error_samples#[:, t]
            dist = dist[np.isfinite(dist)]  # remove NaNs

            if len(dist) == 0:
                continue

            # Probability(X < T1)
            p_below = np.mean(dist < first_tercile)
            # Probability(T1 <= X < T2)
            p_between = np.mean((dist >= first_tercile) & (dist < second_tercile))
            # Probability(X >= T2)
            p_above = 1.0 - (p_below + p_between)

            pred_prob[0, t] = p_below
            pred_prob[1, t] = p_between
            pred_prob[2, t] = p_above

        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_normal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Normal-based method using the Gaussian CDF.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        if np.all(np.isnan(best_guess)):
            pred_prob[:] = np.nan
        else:
            error_std = np.sqrt(error_variance)
            pred_prob[0, :] = stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[1, :] = stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std) - \
                              stats.norm.cdf(first_tercile, loc=best_guess, scale=error_std)
            pred_prob[2, :] = 1 - stats.norm.cdf(second_tercile, loc=best_guess, scale=error_std)
            
        return pred_prob

    @staticmethod
    def calculate_tercile_probabilities_lognormal(best_guess, error_variance, first_tercile, second_tercile):
        """
        Lognormal-based method.
        """
        n_time = len(best_guess)
        pred_prob = np.empty((3, n_time))
        
        # If any input is NaN, fill with NaN
        if np.any(np.isnan(best_guess)) or np.any(np.isnan(error_variance)):
            pred_prob[:] = np.nan
            return pred_prob
        
        # Moment matching for lognormal distribution:
        # Given mean (m) and variance (v), we have:
        # sigma = sqrt(ln(1 + v/m^2)) and mu = ln(m) - sigma^2/2.
        sigma = np.sqrt(np.log(1 + error_variance / (best_guess**2)))
        mu = np.log(best_guess) - sigma**2 / 2
        
        # Use the lognormal CDF from scipy:
        pred_prob[0, :] = lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[1, :] = lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu)) - \
                          lognorm.cdf(first_tercile, s=sigma, scale=np.exp(mu))
        pred_prob[2, :] = 1 - lognorm.cdf(second_tercile, s=sigma, scale=np.exp(mu))
        
        return pred_prob


    def gcm_compute_prob_ensemble_method(self, Obs_data, clim_year_start, clim_year_end, model_data, ensemble="mean"):
        """Compute probabilistic forecasts for GCM data. To Justify after"""
        index_start = Obs_data.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Obs_data.get_index("T").get_loc(str(clim_year_end)).stop
        rainfall_for_tercile = Obs_data.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.3, 0.67], dim='T')
        best_guess_member = getattr(model_data, ensemble)(dim="number")
        error_variance_member = (Obs_data - model_data).var(dim='number')
        model_data_prob = xr.apply_ufunc(
            self.gcm_prob_gamma,
            best_guess_member,
            error_variance_member,
            terciles.isel(quantile=0).drop_vars('quantile'),
            terciles.isel(quantile=1).drop_vars('quantile'),
            input_core_dims=[('T',), ('T',), (), ()],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[('probability', 'T')],
            output_dtypes=['float'],
            dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
        )
        model_data_prob = model_data_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA'])).transpose('probability', 'T', 'Y', 'X')
        return model_data_prob

    def gcm_compute_prob(self, 
                     Predictant, 
                     clim_year_start, 
                     clim_year_end, 
                     hindcast_det):
        """
        Compute tercile probabilities using either 't', 'gamma', 'normal', 'lognormal', or 'nonparam'.

        Parameters:
        -----------
        Predictant : xarray.DataArray
            Observed data array with dimensions (T, Y, X)
        clim_year_start : int
            Start year for climatology
        clim_year_end : int
            End year for climatology
        hindcast_det : xarray.DataArray
            Deterministic forecast (same shape as Predictant, minus M dimension)
        method : str, default = "gamma"
            Method to use for calculating tercile probabilities:
            - "t"
            - "gamma"
            - "normal"
            - "lognormal"
            - "nonparam"
        error_samples : xarray.DataArray or None
            Only required for non-parametric method, shape (ensemble, T, Y, X) or something similar.

        Returns:
        --------
        hindcast_prob : xarray.DataArray
            Probability for each tercile category (PB, PN, PA)
            with dimensions (probability=3, T, Y, X).
        """
        
        # Select climatology slice
        index_start = Predictant.get_index("T").get_loc(str(clim_year_start)).start
        index_end = Predictant.get_index("T").get_loc(str(clim_year_end)).stop

        rainfall_for_tercile = Predictant.isel(T=slice(index_start, index_end))
        terciles = rainfall_for_tercile.quantile([0.33, 0.67], dim='T')

        # We'll pass these thresholds to the methods
        T1 = terciles.isel(quantile=0).drop_vars('quantile')
        T2 = terciles.isel(quantile=1).drop_vars('quantile')
        # Degrees of freedom (used by 't' and weibul methids)
        dof = len(Predictant.get_index("T")) - 2
        
        # ---- CHOOSE THE CALC FUNCTION & ARGUMENTS BASED ON 'method' ----
        if self.dist_method == "t":
                        
            calc_func = self.calculate_tercile_probabilities
            error_variance = (Predictant - hindcast_det).var(dim='T')
            
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True},
            )


        elif self.dist_method == "weibull_min":
            calc_func = self.calculate_tercile_probabilities_weibull_min
            error_variance = (Predictant - hindcast_det).var(dim='T')
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                terciles.isel(quantile=0).drop_vars('quantile'),
                terciles.isel(quantile=1).drop_vars('quantile'),
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk": True}
            )

        elif self.dist_method == "gamma":
            calc_func = self.calculate_tercile_probabilities_gamma
            error_variance = (Predictant - hindcast_det).var(dim='T')
            
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                # kwargs={'dof': dof},
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "normal":
            calc_func = self.calculate_tercile_probabilities_normal
            error_variance = (Predictant - hindcast_det).var(dim='T')
            
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "lognormal":
            calc_func = self.calculate_tercile_probabilities_lognormal
            error_variance = (Predictant - hindcast_det).var(dim='T')
            
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_variance,
                T1,
                T2,
                input_core_dims=[('T',), (), (), ()],
                vectorize=True,
                dask='parallelized',
                output_core_dims=[('probability', 'T')],
                output_dtypes=['float'],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}, "allow_rechunk":True},
            )

        elif self.dist_method == "nonparam":
            calc_func = self.calculate_tercile_probabilities_nonparametric
            error_samples = (Predictant - hindcast_det)
            
            hindcast_prob = xr.apply_ufunc(
                calc_func,
                hindcast_det,
                error_samples,
                T1,
                T2,
                input_core_dims=[('T',), ('T',), (), ()],
                output_core_dims=[('probability','T')],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[float],
                dask_gufunc_kwargs={'output_sizes': {'probability': 3}},
            )

        else:
            raise ValueError(f"Invalid method: {method}. Choose 't', 'gamma', 'normal', 'lognormal', or 'nonparam'.")

        hindcast_prob = hindcast_prob.assign_coords(probability=('probability', ['PB', 'PN', 'PA']))
        return hindcast_prob.transpose('probability', 'T', 'Y', 'X')


    def gcm_validation_compute_(self, center_variable, month_of_initialization, lead_time,
                              dir_model, Obs, year_start, year_end, clim_year_start, clim_year_end,  area, score, dir_to_save_roc_reliability, ensemble_mean=None, gridded=True):

        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        season_months = [((int(month_of_initialization) + int(l) - 1) % 12) + 1 for l in lead_time]
        season = "".join([calendar.month_abbr[month] for month in season_months])
        
        # abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        # season = "".join([
        #     calendar.month_abbr[(int(i) + int(month_of_initialization)) % 12 or 12]
        #     for i in lead_time
        # ])
        
        variables = center_variable[0].split(".")[1]
    
        x_metric = {}
    
        if gridded:
            # Obs_data_ = xr.open_dataset(f"{dir_obs}/Obs_{variables}_{year_start}_{year_end}_{season}.nc") ## Recall the case of reanalysis
            Obs_data_ = Obs
            Obs_data_['T'] = Obs_data_['T'].astype('datetime64[ns]')
            mean_rainfall = Obs_data_.mean(dim="T").squeeze() # .to_array().drop_vars('variable')
            mask_ = xr.where(mean_rainfall <= 20, np.nan, 1)
            mask_ = mask_.where(abs(mask_.Y) <= 22, np.nan)
            
            for i in center_variable:
                score_type = self.scores[score][3]
                center = i.split(".")[0].lower().replace("_", "")
                model_file = f"{dir_model}/hindcast_{center}_{variables}_{abb_mont_ini}Ic_{season}_{lead_time[0]}.nc"
                model_data_ = xr.open_dataset(model_file)
                model_data_['T'] = model_data_['T'].astype('datetime64[ns]')
                
                year_start_ = np.unique(model_data_['T'].dt.year)[0]
                year_end_ = np.unique(model_data_['T'].dt.year)[-1]
                
                Obs_data = Obs_data_.sel(T=slice(str(year_start_),str(year_end_))).interp(Y=model_data_.Y, X=model_data_.X, method="nearest", kwargs={"fill_value": "extrapolate"})
                mask = mask_.interp(Y=model_data_.Y, X=model_data_.X, method="nearest", kwargs={"fill_value": "extrapolate"})
                
                # Obs_data = Obs_data_.interp(Y=model_data_.Y, X=model_data_.X)
                # mask = mask_.interp(Y=model_data_.Y, X=model_data_.X)
                
                Obs_data = Obs_data.where(mask == 1, np.nan)#.to_array().drop_vars("variable").squeeze()
                model_data = model_data_.where(mask == 1, np.nan).to_array().drop_vars("variable").squeeze()

                # model_data, Obs_data = xr.align(model_data, Obs_data, join="left")
                model_data['T'] = Obs_data['T']

                
                if score_type == "det_score":
                    if (ensemble_mean is None) or ("number" in model_data.coords):
                        # print("Deterministic score requires an ensemble mean or median. by default mean will be used")
                        model_data = model_data.mean(dim="number",skipna=True)
                        # ensemble_data = getattr(model_data, ensemble_mean)(dim="number")
                        # continue
                    
                    score_result = self.compute_deterministic_score(
                        self.scores[score][5], Obs_data, model_data
                    )
                    x_metric[f"{center}_{abb_mont_ini}Ic_{season}"] = score_result
                    

                elif score_type == "prob_score":
                    if (ensemble_mean is None) or ("number" in model_data.coords):
                        # print("Deterministic score requires an ensemble mean or median. by default mean will be used")
                        model_data = model_data.mean(dim="number",skipna=True)
                    proba_forecast = self.gcm_compute_prob(Obs_data, clim_year_start, clim_year_end, model_data)
                    score_result = self.compute_probabilistic_score(
                        self.scores[score][5], Obs_data, proba_forecast, clim_year_start, clim_year_end,
                    )
                    x_metric[f"{center}_{abb_mont_ini}Ic_{season}"] = score_result
                
                # elif score_type == "prob_score":
                #     if ensemble_mean is not None:
                #         print("Probabilistic score does not require an ensemble mean or median.")
                #         continue
                #     proba_forecast = self.gcm_compute_prob(Obs_data, model_data, ensemble="mean")
                #     score_result = self.compute_probabilistic_score(
                #         self.scores[score][5], Obs_data, proba_forecast, year_start_, year_end_
                #     )
                #     x_metric[f"{center}_{abb_mont_ini}Ic_{season}"] = score_result

                
                elif score_type == "ensemble_score":
                    if ensemble_mean is not None:
                        print("Ensemble score does not require an ensemble mean or median.")
                    else:
                        score_result = self.compute_crps(Obs_data, model_data, member_dim='number', dim="T")
                        x_metric[f"{center}_{abb_mont_ini}Ic_{season}"] = score_result
    
                elif score_type == "all_grid_prob_score":
                    # if ensemble_mean is not None:
                    #     print("Probabilistic score does not require an ensemble mean or median.")
                    #     continue
                    # proba_forecast = self.gcm_compute_prob(Obs_data, model_data, ensemble="mean")
                    
                    if (ensemble_mean is None) or ("number" in ds.coords):
                        model_data = model_data.mean(dim="number",skipna=True)
                        
                    proba_forecast = self.gcm_compute_prob(Obs_data, clim_year_start, clim_year_end, model_data)  
                    
                    if score == "ROC_CURVE":
                        self.plot_roc_curves(center, dir_to_save_roc_reliability, Obs_data, proba_forecast, clim_year_start, clim_year_end, 
                                             n_bootstraps=1000, ci=0.95)
                    elif score == "RELIABILITY_DIAGRAM":
                        self.reliability_diagram(center, dir_to_save_roc_reliability, Obs_data, proba_forecast, clim_year_start, clim_year_end)
                    else:
                        print(f"Plotting for score {score} is not implemented.")
                        
                elif score_type == "all_grid_det_score":
                    # Implement if needed
                    pass
    
        else:
            print("Non-gridded data validation is not implemented yet.")
    
        return x_metric if self.scores[score][3] in ["det_score", "prob_score", "ensemble_score"] else None


    def gcm_validation_compute(self, models_files_path, Obs, score, month_of_initialization, clim_year_start, clim_year_end, dir_to_save_roc_reliability, lead_time = None, ensemble_mean=None, gridded=True):

        abb_mont_ini = calendar.month_abbr[int(month_of_initialization)]
        if lead_time is None:
            season = ""
        else:
            season_months = [((int(month_of_initialization) + int(l) - 1) % 12) + 1 for l in lead_time]
            season = "".join([calendar.month_abbr[month] for month in season_months])
                    
        x_metric = {}
    
        if gridded:
            Obs_data_ = Obs
            Obs_data_['T'] = Obs_data_['T'].astype('datetime64[ns]')
            
            for i in models_files_path.keys():
                score_type = self.scores[score][3]
                model_data_ = xr.open_dataset(models_files_path[i])
                model_data_['T'] = model_data_['T'].astype('datetime64[ns]')
                
                year_start_ = np.unique(model_data_['T'].dt.year)[0]
                year_end_ = np.unique(model_data_['T'].dt.year)[-1]
                
                Obs_data = Obs_data_.sel(T=slice(str(year_start_),str(year_end_))).interp(Y=model_data_.Y, X=model_data_.X, method="linear", kwargs={"fill_value": "extrapolate"})
                model_data = model_data_.to_array().drop_vars("variable").squeeze()
                model_data['T'] = Obs_data['T']

                
                if score_type == "det_score":
                    if 'number' in model_data.dims:
                        model_data = model_data.mean(dim="number")
                    
                    score_result = self.compute_deterministic_score(
                        self.scores[score][5], Obs_data, model_data
                    )
                    x_metric[f"{i}_{abb_mont_ini}Ic_{season}"] = score_result
                    

                elif score_type == "prob_score":
                    if 'number' in model_data.dims:
                        model_data = model_data.mean(dim="number")

                    proba_forecast = self.gcm_compute_prob(Obs_data, clim_year_start, clim_year_end, model_data)
                    score_result = self.compute_probabilistic_score(
                        self.scores[score][5], Obs_data, proba_forecast, clim_year_start, clim_year_end,
                    )
                    x_metric[f"{i}_{abb_mont_ini}Ic_{season}"] = score_result
                
                
                elif score_type == "ensemble_score":
                    if ensemble_mean is not None:
                        print("Ensemble score does not require an ensemble mean or median.")
                    else:
                        score_result = self.compute_crps(Obs_data, model_data, member_dim='number', dim="T")
                        x_metric[f"{i}_{abb_mont_ini}Ic_{season}"] = score_result
    
                elif score_type == "all_grid_prob_score":
                    
                    if (ensemble_mean is None) or ("number" in ds.coords):
                        model_data = model_data.mean(dim="number",skipna=True)
                        
                    proba_forecast = self.gcm_compute_prob(Obs_data, clim_year_start, clim_year_end, model_data)  
                    
                    if score == "ROC_CURVE":
                        self.plot_roc_curves(center, dir_to_save_roc_reliability, Obs_data, proba_forecast, clim_year_start, clim_year_end, 
                                             n_bootstraps=1000, ci=0.95)
                    elif score == "RELIABILITY_DIAGRAM":
                        self.reliability_diagram(center, dir_to_save_roc_reliability, Obs_data, proba_forecast, clim_year_start, clim_year_end)
                    else:
                        print(f"Plotting for score {score} is not implemented.")
                        
                elif score_type == "all_grid_det_score":
                    # Implement if needed
                    pass
    
        else:
            print("Non-gridded data validation is not implemented yet.")
    
        return x_metric if self.scores[score][3] in ["det_score", "prob_score", "ensemble_score"] else None



    def weighted_gcm_forecasts(
        self,
        Obs,
        best_models,
        scores,
        lead_time,
        model_dir,
        clim_year_start, 
        clim_year_end,
        variable="PRCP" 
    ):
        parts = list(best_models.keys())[0].split("_")
        tmp = xr.open_dataset(f"{model_dir}/hindcast_{parts[0]}_{variable}_{parts[1]}_{parts[2]}_{lead_time[0]}.nc")
        
        
        # 2. Initialize accumulators
        score_sum = None         # Will hold the sum of MAE-derived weights (or any chosen metric)
        hindcast_det = None      # Will hold the weighted sum of hindcasts
        forecast_det = None      # Will hold the weighted sum of forecasts
    
        # 3. Loop over the best models
        for model_name in best_models.keys():
            # 3a. Get the model's score array (MAE) and interpolate it to Obs grid
            score_array = scores["GROC"][model_name]
            score_array = score_array.interp(
                Y=tmp.Y,
                X=tmp.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
    
            # 3b. Build filenames for hindcast/forecast NetCDF files
            #     Model name pattern assumed: something like "Model1_runID_variant"
            #     This is just an example, adapt to your actual naming convention.
            parts = model_name.split("_")
            hindcast_file = (
                f"{model_dir}/hindcast_{parts[0]}_{variable}_{parts[1]}_{parts[2]}_{lead_time[0]}.nc"
            )
            forecast_file = (
                f"{model_dir}/forecast_{parts[0]}_{variable}_{parts[1]}_{parts[2]}_{lead_time[0]}.nc"
            )
    
            # 3c. Open hindcast file
            hincast_data = xr.open_dataset(hindcast_file)
            # If ensemble dimension present, take the mean across ensemble members
            if "number" in hincast_data.coords:
                hincast_data = hincast_data.mean(dim="number", skipna=True)
            # Convert time coordinate to datetime64
            hincast_data["T"] = hincast_data["T"].astype("datetime64[ns]")
            # Interpolate to match Obs grid
            hincast_data = hincast_data.interp(
                Y=tmp.Y,
                X=tmp.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )

            # 3d. Open forecast file
            forecast_data = xr.open_dataset(forecast_file)
            if "number" in forecast_data.coords:
                forecast_data = forecast_data.mean(dim="number", skipna=True)
            forecast_data["T"] = forecast_data["T"].astype("datetime64[ns]")
            forecast_data = forecast_data.interp(
                Y=tmp.Y,
                X=tmp.X,
                method="nearest",
                kwargs={"fill_value": "extrapolate"}
            )
    
            # 3e. Multiply each dataset by its score_array for weighting
            hincast_weighted = hincast_data * score_array
            forecast_weighted = forecast_data * score_array

            # 3f. Accumulate into the running total
            if hindcast_det is None:
                # First iteration
                hindcast_det = hincast_weighted
                forecast_det = forecast_weighted
                score_sum = score_array
            else:
                # Subsequent iterations: add to existing
                hindcast_det = hindcast_det + hincast_weighted
                forecast_det = forecast_det + forecast_weighted
                score_sum = score_sum + score_array
    
        # 4. Convert sums to weighted means
        hindcast_det = hindcast_det / score_sum
        hindcast_det = hindcast_det.interp(
                    Y=Obs.Y,
                    X=Obs.X,
                    method="nearest",
                    kwargs={"fill_value": "extrapolate"}
                )
        
        forecast_det = forecast_det / score_sum
        forecast_det = forecast_det.interp(
                    Y=Obs.Y,
                    X=Obs.X,
                    method="nearest",
                    kwargs={"fill_value": "extrapolate"}
                )
            
        # 5. Scale factor based on mean ratio to observations 
        #    (this is domain/organization-specific logic)
        f = Obs.mean("T") / hindcast_det.mean("T")
        hindcast_det = hindcast_det * f
        forecast_det = forecast_det * f
        
        # f_fcst = Obs.mean("T") / hindcast_det.mean("T")
        
        # f_hdcst = []
        # [f_hdcst.append((Obs.sel(T=Obs['T'] != Obs['T'].isel(T=i)).mean("T", skipna=True) / hindcast_det.sel(T=hindcast_det['T'] != hindcast_det['T'].isel(T=i)).mean("T", skipna=True))) for i in range(0,len(hindcast_det['T']))]
        
        # f_hdcst = xr.concat(f_hdcst, dim="T") 
        # f_hdcst['T'] = hindcast_det['T']

        
        # hindcast_det = hindcast_det * f_hdcst
        # forecast_det = forecast_det * f_fcst
    
        # 6. Convert to DataArray and fix dimensions
        hindcast_det = (
            hindcast_det
            .to_array()                   # combine variables into one array
            .drop_vars("variable")        # remove the 'variable' coordinate
            .squeeze()                    # remove any size-1 dimension
            .transpose("T", "Y", "X")     # rearrange dimension order
            )
    
        forecast_det = (
            forecast_det
            .to_array()
            .drop_vars("variable")
            .squeeze("variable")            # if there's a single variable dimension
            .transpose("T", "Y", "X")
        )
    
        # 7. Compute probabilities using  verification methods
        hindcast_prob = self.gcm_compute_prob(Obs, clim_year_start, clim_year_end, hindcast_det)#.fillna(0.33)
        
        forecast_prob = (
            self.gcm_compute_prob_forecast(Obs, clim_year_start, clim_year_end, hindcast_det, forecast_det)
            .squeeze()
            .drop_vars("T")  
        )
    
        return hindcast_det, hindcast_prob, forecast_prob
######################## Validation annual year        
    @staticmethod
    def classify_percent(p):
        if p >= 150:
            return 1  # Well Above Average
        elif p >= 110:
            return 2  # Above Average
        elif p >= 90:
            return 3  # Near Average
        elif p >= 50:
            return 4  # Below Average
        else:
            return 5  # Well Below Average
            
    def ratio_to_average(self, predictant, clim_year_start, clim_year_end, year):       
        clim_slice = slice(str(clim_year_start), str(clim_year_end))
        clim_mean = predictant.sel(T=clim_slice).mean(dim='T')
        ratio = 100*predictant.sel(T=str(year))/clim_mean
        mask = xr.where(~np.isnan(predictant.isel(T=0)), 1, np.nan)\
                .drop_vars(['T']).squeeze()
        mask.name = None
        # Vectorized classification
        classified = xr.apply_ufunc(
            self.classify_percent,
            ratio,
            input_core_dims=[('T',)],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float']
        )*mask    
 
        # Define colormap and labels
        # cmap = ListedColormap(["darkgreen", "limegreen", "gray", "orange", "red"])
        cmap = ListedColormap(['#1a9641', '#a6d96a', '#ffffbf','#fdae61', '#d7191c', ])
        labels = ["Well Above Avg", "Above Avg", "Near Avg", "Below Avg", "Well Below Avg"]
        
        fig = plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Plot
        im = classified.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            add_colorbar=False
        )
        
        # Add shapefiles or borders
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.COASTLINE)
        
        # Custom legend
        import matplotlib.patches as mpatches
        legend_patches = [mpatches.Patch(color=cmap(i), label=labels[i]) for i in range(5)]
        plt.legend(handles=legend_patches, loc='lower left')
        plt.title("Ratio to Normal [%]")
        plt.show()



    def calculate_rpss_(self, y_true, y_probs):
        """
        Compute Ranked Probability Skill Score (RPSS).
        
        Parameters
        ----------
        y_true : array-like of shape (n_samples,)
            True class labels (already classified as 0, 1, or 2).
        y_probs : array-like of shape (3, n_samples)
            Forecast probabilities for each class.
        
        Returns
        -------
        rpss : float
            Ranked Probability Skill Score.
        """
        encoder = OneHotEncoder(categories=[[0, 1, 2]], sparse_output=False)
    
        # Mask for valid values
        mask = np.isfinite(y_true) & np.isfinite(y_probs).all(axis=0)
    
        if np.any(mask):
            y_true_clean = y_true[mask]
            y_probs_clean = y_probs[:, mask]
    
            # One-hot encode y_true (now no need for digitize or percentile)
            one_hot = encoder.fit_transform(y_true_clean.reshape(-1, 1))
    
            # Compute cumulative distributions
            cumulative_forecast = np.cumsum(np.swapaxes(y_probs_clean, 0, 1), axis=1)
            cumulative_outcome = np.cumsum(one_hot, axis=1)
    
            # Climatology: uniform probability (1/3)
            climatology = np.array([1/3, 1/3, 1/3])
            cumulative_climatology = np.cumsum(climatology)
    
            # Compute RPS
            rps_forecast = np.mean(np.sum((cumulative_forecast - cumulative_outcome) ** 2, axis=1))
            rps_reference = np.mean(np.sum((cumulative_climatology - cumulative_outcome) ** 2, axis=1))
    
            return 1 - (rps_forecast / rps_reference)
        else:
            return np.nan

    def compute_one_year_rpss(self, obs, prob_pred, clim_year_start, clim_year_end, year):
        """Apply a probabilistic scoring function over xarray DataArrays."""
    
        obs = self.compute_class(obs, clim_year_start, clim_year_end)
        obs = obs.sel(T=str(year))
        prob_pred['T'] = obs['T']
        obs, prob_pred = xr.align(obs, prob_pred)
        
        A = xr.apply_ufunc(
            self.calculate_rpss_,
            obs,
            prob_pred,
            input_core_dims=[('T',), ('probability', 'T')],
            vectorize=True,
            dask='parallelized',
            output_core_dims=[()],
            output_dtypes=['float'],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        
        A_ = xr.where(A > 1 + 1e-6, 1, A)
        fig = plt.figure(figsize=(10, 8))
        ax = plt.axes(projection=ccrs.PlateCarree())
        
        # Plot with enhanced colorbar
        A_.plot(
            ax=ax,
            transform=ccrs.PlateCarree(),
            cmap='RdBu_r',
            vmin=-1, vmax=1+ 1e-6,
            cbar_kwargs={
                'label': 'RPSS',
                'shrink': 0.5,
                'extend': 'both',
                'orientation': 'vertical',
                'ticks': [-1, -0.5, 0, 0.5, 1]
            }
        )
        
        # Add base map features
        ax.add_feature(cfeature.BORDERS)
        ax.add_feature(cfeature.COASTLINE)
        ax.set_title(f"Ranked Probability Skill Score - {year}")
        
        plt.tight_layout()
        plt.show()
