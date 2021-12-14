# %%
import numpy as np
from matplotlib import pyplot as plt
import sys
sys.path.append('C:/Users/koki1/Google ドライブ/develop/PyTools/ForReserch/')
from AreaStats import AreaStats
from ImgConverter import ImgConverter
from statsmodels.tsa.seasonal import STL
import pymannkendall as mk

# %%
h,w = 1600, 1500
imc = ImgConverter()
for ceis in imc.CEIs_TEMP_ls + imc.CEIs_PRCP_ls:
    ceis_all_img = np.full((20*12, h, w), np.nan, dtype=np.float64)
    ndvi_all_img = np.full((20*12, h, w), np.nan, dtype=np.float64)

    out_img = np.full((h, w), np.nan, dtype=np.float64)

    c = 0

    # 画像の取得
    for year in np.arange(2001, 2020+1):
        for month in np.arange(1, 12+1):
            ceis_img = np.fromfile(
                f'E:/ResearchData_SUB/Level4/ETCCDI_RAW_005_Monthly/ETCCDI_{ceis}/ETCCDI_{ceis}.B{year}{str(month).zfill(2)}.float64_h1600w1500.raw',
                count=h*w, dtype=np.float64).reshape(h,w)
            ndvi_img = np.fromfile(
                f'D:/ResearchData/Level4/NDVI_Monthly/NDVI_Monthly.B{year}{str(month).zfill(2)}.float64_h1600w1500.raw',
                count=h*w, dtype=np.float64).reshape(h,w)

            ceis_all_img[c, :, :] = ceis_img
            ndvi_all_img[c, :, :] = ndvi_img
            c+=1
            print(f'{ceis} GetImages:{year}/{month}')
    


    for row in range(h):
        for column in range(w):
            target_ceis = ceis_all_img[:, row, column]
            target_ndvi = ndvi_all_img[:, row, column]

            if (np.nansum(np.isnan(target_ceis))!=0)|(np.nansum(np.isnan(target_ndvi)!=0)):
                corr = np.nan
            else:
                trend_ceis = STL(target_ceis, period=12).fit().trend
                trend_ndvi = STL(target_ndvi, period=12).fit().trend
                corr = np.corrcoef(trend_ceis, trend_ndvi)[0,1]

            out_img[row, column] = corr
            print(f'{ceis} | ROW:{row},COLUMN:{column} | {np.round(corr, 2)}')

    out_img.tofile(f'D:/ResearchData/Level6/NDVI_CEIs_Coef_RAW_005/Coef_NDVI_{ceis}.C20012020.float64_h{h}w{w}.raw')




    del ceis_all_img, ndvi_all_img
    break

# %%
np.corrcoef(trend_ceis, trend_ndvi)