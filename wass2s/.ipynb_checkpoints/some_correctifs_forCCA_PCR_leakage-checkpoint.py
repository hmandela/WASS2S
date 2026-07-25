if isinstance(model, WAS_CCA):
    # 1. Parameter Splitting
    all_params = {**model_params}
    params_prob = {
        k: v for k, v in all_params.items()
        if k not in model.compute_model.__code__.co_varnames
    }
    params_models = {
        k: v for k, v in all_params.items()
        if k not in params_prob
    }

    # 2. Prepare Raw Data & Global Mask
    # We work with deep copies to ensure no modification propagates
    Predictant_safe = Predictant.copy(deep=True)
    mask = xr.where(~np.isnan(Predictant_safe.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
    
    print("Starting Rigorous Cross-Validation (No Data Leakage)...")
    hindcast_list = []

    # 3. Cross-Validation Loop
    for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
        
        # A. Split RAW Data
        X_train_raw = Predictor.isel(T=train_index)
        X_test_raw  = Predictor.isel(T=test_index)
        y_train_raw = Predictant_safe.isel(T=train_index)
        # y_test_raw is not needed for training, only for final verification

        # B. Process Predictor (X) inside the loop
        # 1. Detrend Training X
        X_train_det, X_coeffs, X_meta = detrended_data(X_train_raw, dim="T")
        X_train_ready = X_train_det.fillna(0.0)

        # 2. Process Test X using TRAINING statistics
        # We apply the trend learned from X_train to X_test
        X_test_trend = apply_detrend_data(X_test_raw, X_coeffs, X_meta)
        X_test_det = X_test_raw - X_test_trend
        X_test_ready = X_test_det.fillna(0.0)

        # C. Process Predictant (Y) inside the loop
        # 1. Standardize Training Y (Dynamic Climatology)
        # We calculate mean/std strictly on the training set
        y_mean = y_train_raw.mean(dim="T")
        y_std  = y_train_raw.std(dim="T")
        y_train_st = (y_train_raw - y_mean) / y_std

        # 2. Detrend Training Y
        y_train_det, y_coeffs, y_meta = detrended_data(y_train_st, dim="T")
        y_train_ready = y_train_det.fillna(0.0)

        # D. Compute Model
        # The model learns to predict: Standardized & Detrended Anomalies
        pred_st_det = model.compute_model(X_train_ready, y_train_ready, X_test_ready, None, **params_models)

        # E. Reconstruction (Inverse Transform)
        # 1. Add Trend back (Using coeff learned from Train)
        # pred_st_det has the 'T' coordinate of the test year, so the math works
        pred_trend = apply_detrend_data(pred_st_det, y_coeffs, y_meta)
        pred_st = pred_st_det + pred_trend

        # 2. Destandardize (Using mean/std from Train)
        pred_raw = (pred_st * y_std) + y_mean
        
        hindcast_list.append(pred_raw)

    # 4. Assembly and Cleanup
    hindcast_det = xr.concat(hindcast_list, dim="T").sortby("T")
    
    # Apply mask and physical constraints
    hindcast_det = hindcast_det.transpose('T', 'Y', 'X') * mask
    hindcast_det = hindcast_det.clip(min=0)

    # 5. Probabilities
    # Probabilities are calculated globally on the rigorous hindcasts
    hindcast_prob = model.compute_prob(
        Predictant_safe,
        clim_year_start,
        clim_year_end,
        hindcast_det,
        **params_prob
    )
    hindcast_prob = (hindcast_prob * mask).clip(min=0, max=1)

    return hindcast_det, hindcast_prob

#### Other tentavives

if isinstance(model, WAS_CCA):
    # 1. Parameter Splitting
    all_params = {**model_params}
    params_prob = {
        k: v for k, v in all_params.items()
        if k not in model.compute_model.__code__.co_varnames
    }
    params_models = {
        k: v for k, v in all_params.items()
        if k not in params_prob
    }

    # 2. Prepare Raw Data & Global Mask
    # We work with deep copies to ensure no modification propagates
    Predictant_safe = Predictant.copy(deep=True)
    mask = xr.where(~np.isnan(Predictant_safe.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()
    
    print("Starting Rigorous Cross-Validation (No Data Leakage)...")
    hindcast_list = []

    # 3. Cross-Validation Loop
    for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
        
        # A. Split RAW Data
        X_train_raw = Predictor.isel(T=train_index)
        X_test_raw  = Predictor.isel(T=test_index)
        y_train_raw = Predictant_safe.isel(T=train_index)
        # y_test_raw is not needed for training, only for final verification

        # B. Process Predictor (X) inside the loop
        # 1. Detrend Training X
        X_train_det, X_coeffs, X_meta = detrended_data(X_train_raw, dim="T")
        X_train_ready = X_train_det.fillna(0.0)

        # 2. Process Test X using TRAINING statistics
        # We apply the trend learned from X_train to X_test
        X_test_trend = apply_detrend_data(X_test_raw, X_coeffs, X_meta)
        X_test_det = X_test_raw - X_test_trend
        X_test_ready = X_test_det.fillna(0.0)

        # C. Process Predictant (Y) inside the loop
        # 1. Standardize Training Y (Dynamic Climatology)
        # We calculate mean/std strictly on the training set
        y_mean = y_train_raw.mean(dim="T")
        y_std  = y_train_raw.std(dim="T")
        y_train_st = (y_train_raw - y_mean) / y_std

        # 2. Detrend Training Y
        y_train_det, y_coeffs, y_meta = detrended_data(y_train_st, dim="T")
        y_train_ready = y_train_det.fillna(0.0)

        # D. Compute Model
        # The model learns to predict: Standardized & Detrended Anomalies
        pred_st_det = model.compute_model(X_train_ready, y_train_ready, X_test_ready, None, **params_models)

        # E. Reconstruction (Inverse Transform)
        # 1. Add Trend back (Using coeff learned from Train)
        # pred_st_det has the 'T' coordinate of the test year, so the math works
        pred_trend = apply_detrend_data(pred_st_det, y_coeffs, y_meta)
        pred_st = pred_st_det + pred_trend

        # 2. Destandardize (Using mean/std from Train)
        pred_raw = (pred_st * y_std) + y_mean
        
        hindcast_list.append(pred_raw)

    # 4. Assembly and Cleanup
    hindcast_det = xr.concat(hindcast_list, dim="T").sortby("T")
    
    # Apply mask and physical constraints
    hindcast_det = hindcast_det.transpose('T', 'Y', 'X') * mask
    hindcast_det = hindcast_det.clip(min=0)
    hindcast_det['T'] = Predictant_safe['T']

    # 5. Probabilities
    # Probabilities are calculated globally on the rigorous hindcasts
    hindcast_prob = model.compute_prob(
        Predictant_safe,
        clim_year_start,
        clim_year_end,
        hindcast_det,
        **params_prob
    )
    hindcast_prob = (hindcast_prob * mask).clip(min=0, max=1)

    return hindcast_det, hindcast_prob


elif (isinstance(model, WAS_PCR) and any(isinstance(model.__dict__['reg_model'], i) for i in same_kind_model1)): 
            
            # 1. Parameter Splitting
            all_params = {**model_params}
            params_prob = {
                k: v for k, v in all_params.items() 
                if k not in model.__dict__['reg_model'].compute_model.__code__.co_varnames
            }
            params_models = {
                k: v for k, v in all_params.items() 
                if k not in params_prob
            } 

            # 2. Global Mask (Safe to do once)
            Predictant_safe = Predictant.copy(deep=True)
            mask = xr.where(~np.isnan(Predictant_safe.isel(T=0)), 1, np.nan).drop_vars(['T']).squeeze()

            print("Starting Rigorous PCR Cross-Validation (No Leakage)...")
            hindcast_list = []

            # 3. Cross-Validation Loop
            for i, (train_index, test_index) in enumerate(tqdm(self.custom_cv.split(Predictor['T'], self.nb_omit), total=n_splits), start=1):
                
                # --- A. Split Raw Data ---
                X_train_raw = Predictor.isel(T=train_index)
                X_test_raw  = Predictor.isel(T=test_index)
                y_train_raw = Predictant_safe.isel(T=train_index)
                
                # --- B. Process Predictor (X) ---
                # 1. Detrend Training X (Learn Trend from Train)
                X_train_det, X_coeffs, X_meta = detrended_data(X_train_raw, dim="T")
                
                # 2. Process Test X (Apply Training Trend to Test)
                # We calculate the trend value for the test year using the slope learned from Train
                X_test_trend = apply_detrend_data(X_test_raw, X_coeffs, X_meta)
                X_test_det = X_test_raw - X_test_trend

                # --- C. Process Predictant (Y) ---
                # 1. Standardize Training Y (Dynamic Climatology)
                # We calculate mean/std strictly on the training set to avoid leakage
                y_mean = y_train_raw.mean(dim="T")
                y_std  = y_train_raw.std(dim="T")
                y_train_st = (y_train_raw - y_mean) / y_std

                # --- D. Compute Model ---
                # The model trains on Detrended X and Standardized Y
                pred_st = model.compute_model(X_train_det, y_train_st, X_test_det, None, **params_models)
                
                # --- E. Reconstruction (Inverse Transform) ---
                # 1. Destandardize (Using mean/std learned from Train)
                # We convert the model's standardized prediction back to mm
                pred_raw = (pred_st * y_std) + y_mean
                
                hindcast_list.append(pred_raw)

            # 4. Assembly and Cleanup
            hindcast_det = xr.concat(hindcast_list, dim="T").sortby("T")
            
            # Apply Mask & Physical Constraints (Rainfall >= 0)
            hindcast_det = hindcast_det.transpose('T', 'Y', 'X') * mask
            hindcast_det = hindcast_det.clip(min=0)

            # 5. Probabilistic Calculation
            # Calculated globally on the rigorously reconstructed hindcasts
            hindcast_prob = model.compute_prob(
                Predictant_safe, 
                clim_year_start, 
                clim_year_end, 
                hindcast_det, 
                **params_prob
            )

            return hindcast_det, (hindcast_prob * mask).clip(min=0, max=1)