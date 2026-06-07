

import rasterio
import matplotlib.pyplot as plt

# Abre tu archivo
with rasterio.open('data/raw/alphaearth_avocado_region_2025_64bands.tif') as src:
    evi_data = src.read(1)

# Visualízalo
plt.imshow(evi_data, cmap='RdYlGn') # RdYlGn es genial para índices de vegetación
plt.colorbar(label='Valor NDWI')
plt.title('Índice NDWI')
plt.show()