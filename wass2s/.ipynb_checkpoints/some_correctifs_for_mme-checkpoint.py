### 1 - Je dois revoir et tenir compte du leakage avant publication


### 2 - Aider l'apprentissage en intégrant dans MLP et ELM les Moyennes mean et std de "M" comme indiqué en bas 
import xarray as xr

# Assuming 'hindcast' is your existing DataArray (T, M, Y, X)

# 1. Calculate Mean and Std along 'M'
# We keep_attrs=True to preserve metadata
mean_da = hindcast.mean(dim='M', keep_attrs=True)
std_da = hindcast.std(dim='M', keep_attrs=True)

# 2. Expand dimensions to add 'M' back to these calculations
# We assign specific coordinates ('mean', 'std') so you can identify them later
mean_da = mean_da.expand_dims(M=['mean'])
std_da = std_da.expand_dims(M=['std'])

# 3. Concatenate the original data with the new stats
hindcast_expanded = xr.concat([hindcast, mean_da, std_da], dim='M')

# Result: hindcast_expanded has shape (T, M+2, Y, X)
print(hindcast_expanded)