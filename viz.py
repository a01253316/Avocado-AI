

import rasterio
import matplotlib.pyplot as plt

# Abre tu archivo
with rasterio.open('data/processed/indices/H1/2024-03-07/NDWI.tif') as src:
    evi_data = src.read(1)

# Visualízalo
plt.imshow(evi_data, cmap='RdYlGn') # RdYlGn es genial para índices de vegetación
plt.colorbar(label='Valor NDWI')
plt.title('Índice NDWI')
plt.show()