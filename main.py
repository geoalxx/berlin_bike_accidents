import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore") # ignore warnings


def deriveSpeedLimit(row):
    default = 50

    if np.isnan(row['index_speed']) & np.isnan(row['index_road']):
        # no intersection with roads or speed limits found
        return None

    if np.isnan(row['index_speed']):
        # road without speed limit -> further check road prio
        if row['prio'] == 3:
            return default
        else:
            return None

    # check whether speed limit was already valid when accident occured
    if row['dat_t'] is not None:
        acc_date_str = f"{row['UJAHR']}-{row['UMONAT']}-15"
        acc_dt = datetime.strptime(acc_date_str, '%Y-%m-%d')
        if acc_dt < row['date']:
            # -> speed limit not yet valid (date)
            return default

    # check day
    if row['tag_t'] is not None:
        if (row['UWOCHENTAG'] < row['day_from'] or row['UWOCHENTAG'] > row['day_to']):
            # -> speed limit not yet valid (day)
            return default

    # check time
    if row['zeit_t'] is not None:
        if row['time_from'] < row['time_to']:
            # -> limit during day
            if (row['USTUNDE'] < row['time_from'] or row['USTUNDE'] >= row['time_to']):
                # -> speed limit not yet valid (time)
                return default
        else:
            # -> limit during night
            if (row['USTUNDE'] < row['time_from'] and row['USTUNDE'] >= row['time_to']):
                # -> speed limit not yet valid (time)
                return default

    # -> valid speed limit determined!
    return row['wert_ves']


# init
path = 'C:/Python/berlin_bike_accidents/'
dir_output = 'output/'
gpkg_name = 'map_data.gpkg'
gpkg_name_buf = 'map_data_buffered.gpkg'

"""
Prepare data for processing
    - load preprocessed data
    - delete/rename columns if possible
    - create buffers for roads/accidents and write to GeoPackage
    - clean-up data that isn't relevant for speed limit determination
    - split road dataset into roads with and without speed limit
"""

# read preprocessed files
print("--- Read and prepare data ---")
df_bike_acc_all = gpd.read_file(path + dir_output + gpkg_name, layer='bike_accidents')
df_roads = gpd.read_file(path + dir_output + gpkg_name, layer='fis_strassenabschnitte')
df_speed = gpd.read_file(path + dir_output + gpkg_name, layer='fis_tempolimit')
df_rva = gpd.read_file(path + dir_output + gpkg_name, layer='fis_rva')

# delete columns that aren't required for further processing
df_bike_acc_all.drop(['ULAND', 'UKREIS', 'UGEMEINDE', 'OBJECTID'], axis=1, inplace=True)
df_roads.drop(['strassenna', 'str_bez', 'strassenkl', 'strassen_1','strassen_2','verkehrsri'], axis=1, inplace=True)

# rename columns
df_speed.rename(columns={'elem_nr':'element_nr'}, inplace=True)

# create buffer for road segments and accidents to find a good match
df_bike_acc_all['geometry'] = df_bike_acc_all['geometry'].buffer(15, resolution=2)
df_roads['geometry'] = df_roads['geometry'].buffer(15, resolution=2, cap_style=2)
df_speed['geometry'] = df_speed['geometry'].buffer(15, resolution=2, cap_style=2)

#  and write to GeoPackage
# - buffered accidents (all)
# - buffered roads
# - buffered speed limits
df_bike_acc_all.to_file(path + dir_output + gpkg_name_buf, layer='buf_acc_all', driver='GPKG')
df_roads.to_file(path + dir_output + gpkg_name_buf, layer='buf_roads', driver='GPKG')
df_speed.to_file(path + dir_output + gpkg_name_buf, layer='buf_speed', driver='GPKG')
print("- GeoPackage updated")

# clean-up data:
# - speed: ignore road prio '0' (e.g. Autobahn) and records without speed limit
# - roads: ignore road prio '0' (e.g. Autobahn) and '1' (e.g. park, KGA, ...)
df_speed = pd.merge(df_speed, df_roads[['element_nr','prio']], how='left', on="element_nr")
df_speed = df_speed.loc[df_speed['prio'] != 0]
df_speed = df_speed.dropna(subset=['wert_ves'])
df_roads = df_roads.loc[df_roads['prio'] > 1]

# reduce roads by matching entries found in speed limits dataset (matching 'element_nr')
df_roads_wo_speed_idx = pd.concat([df_roads['element_nr'], df_speed['element_nr'], df_speed['element_nr']]).drop_duplicates(keep=False)
df_roads_wo_speed = df_roads.loc[df_roads['element_nr'].isin(df_roads_wo_speed_idx.values)]
print("- all data prepared")

"""
Spatial Joins & concatenation of results
    - accidents x roads (without speed limit)
    - accidents x roads (with speed limit)
    - accidents x roads (partial speed limit)
"""

print("\n--- Execute spatial joins ---")

# Spatial join (1): bike accidents for all roads without speed limit -> drop duplicates by considering road prio (highest ranked road 'wins')
df_acc_x_road_wo_speed = gpd.sjoin(df_bike_acc_all, df_roads_wo_speed, how='inner', predicate='intersects', rsuffix='road')
df_acc_x_road_wo_speed = df_acc_x_road_wo_speed.sort_values(['prio'], ascending=False).drop_duplicates(subset=['GUID'], keep='first')

# Spatial join (2): bike accidents for all roads with speed limit
df_acc_x_speed = gpd.sjoin(df_bike_acc_all, df_speed, how='inner', predicate='intersects', rsuffix='speed')

# concatenate all accidents that have been found so far (roads with or without speed limit)
df_concat = pd.concat([df_acc_x_road_wo_speed, df_acc_x_speed], axis=0)

# specialty: speed limits sometimes don't cover the entire length of the road, although they have matching IDs
# -> some roads have been ignored in spatial join (1) because it was assumed intersect was found in speed limit DF
# -> extract those missing entries and spatial join (3) again with roads
df_acc_rem_guid = pd.concat([df_bike_acc_all['GUID'], df_concat['GUID']]).drop_duplicates(keep=False)
df_acc_rem = df_bike_acc_all.loc[df_bike_acc_all['GUID'].isin(df_acc_rem_guid.values)]
df_acc_rem_x_road = gpd.sjoin(df_acc_rem, df_roads, how='left', predicate='intersects', rsuffix='road')

# append/concat remaining accidents
df_concat = pd.concat([df_concat, df_acc_rem_x_road], axis=0)
print("- all data joined")

"""
Derive speed limits by considering
    - road type
    - start of speed limit
    - validity of speed limit (day of week & time)
    - highest ranked speed limit in case of multiple matches
    -> write results to GeoPackage
"""

# all accidents intersecting roads (with or without speed limit) can be processed now to determine actual speed limit
print("\n--- Derive speed limits ---")
df_concat['speed_der'] = df_concat.apply(deriveSpeedLimit, axis=1)

# since accidents at intersections have multiple road matches
# -> drop duplicates (road with the highest speed limit 'wins')
# -> remove unnecessary columns
df_concat_u = df_concat.sort_values(['speed_der'],ascending=False).drop_duplicates(subset=['GUID'], keep='first')
df_concat_u.drop(['index_road', 'index_speed', 'zeit_t', 'tag_t','dat_t'], axis=1, inplace=True)

#  and write to GeoPackage
# - buffered accidents (incl. speed limit)
df_concat_u.to_file(path + dir_output + gpkg_name_buf, layer='buf_acc_speed', driver='GPKG')
print("- GeoPackage updated")

"""
Create crosstab for statistics and heatmap
"""

print("\n--- Create statistics ---")

# crosstab with absolute and percentage values
crosstab = pd.crosstab(df_concat_u['UKATEGORIE'],df_concat_u['speed_der']) #,margins=True)
crosstab_idx = pd.crosstab(df_concat_u['UKATEGORIE'],df_concat_u['speed_der'], normalize='columns').round(4)*100
print(crosstab)
print(crosstab_idx)

# heatmap
sns.heatmap(crosstab, annot=True, fmt="g", square=True, norm=LogNorm())
plt.show()

crosstab = pd.crosstab(df_concat_u['UKATEGORIE'],df_concat_u['IstPKW']) #,margins=True)

# """
# Temp: data validation
# """
# sel_guid = [13293, 15384, 19527, 19014, 13460, 8046, 657, 13952, 3617, 11973]
# df_sel = df_concat_u.loc[(df_concat_u['GUID'].isin(sel_guid))]

# ToDo: count join results (how many roads in area of accident)