import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")  # ignore warnings


def deriveSpeedLimit(row):
    default = 50

    if np.isnan(row['index_speed']) & np.isnan(row['index_road']):
        # no intersection with roads or speed limits found
        return None

    if np.isnan(row['index_speed']):
        # road without speed limit -> further check road rank
        if row['rank'] == 3:
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
        if row['UWOCHENTAG'] < row['day_from'] or row['UWOCHENTAG'] > row['day_to']:
            # -> speed limit not yet valid (day)
            return default

    # check time
    if row['zeit_t'] is not None:
        if row['time_from'] < row['time_to']:
            # -> limit during day
            if row['USTUNDE'] < row['time_from'] or row['USTUNDE'] >= row['time_to']:
                # -> speed limit not yet valid (time)
                return default
        else:
            # -> limit during night
            if row['USTUNDE'] < row['time_from'] and row['USTUNDE'] >= row['time_to']:
                # -> speed limit not yet valid (time)
                return default

    # -> valid speed limit determined!
    return row['wert_ves']


# read local config.ini file
rel_path = './'
config = configparser.ConfigParser()
config.read(rel_path + 'config.ini')

# get from config.ini
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
gpkg_src = rel_path + dir_output + config['FILE_SETTINGS']['GPKG_NAME']
gpkg_out = rel_path + dir_output + config['FILE_SETTINGS']['GPKG_NAME_BUF']

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
df_bike_acc = gpd.read_file(gpkg_src, layer='bike_accidents')
df_roads = gpd.read_file(gpkg_src, layer='fis_strassenabschnitte')
df_speed = gpd.read_file(gpkg_src, layer='fis_tempolimit')
df_rva = gpd.read_file(gpkg_src, layer='fis_rva')
print("- data read")

# delete columns that aren't required for further processing
df_bike_acc.drop(['ULAND', 'UKREIS', 'UGEMEINDE', 'OBJECTID'], axis=1, inplace=True)
df_bike_acc_buf = df_bike_acc.copy()
df_bike_acc_buf.drop(['UKATEGORIE', 'UART', 'UTYP1',
                      'IstRad', 'IstPKW', 'IstFuss', 'IstKrad', 'IstGkfz', 'IstSonstig'], axis=1, inplace=True)
df_roads.drop(['strassenna', 'str_bez', 'strassenkl', 'strassen_1', 'strassen_2', 'verkehrsri'], axis=1, inplace=True)
df_rva.drop(['sobj_kz', 'segm_segm', 'rva_typ', 'sorvt_typ', 'b_pflicht'], axis=1, inplace=True)

# rename columns
df_speed.rename(columns={'elem_nr': 'element_nr'}, inplace=True)

# create buffer for road/RVA segments and accidents to find a good match
# -> smaller buffer for RVA since each lane is captured separately
df_bike_acc_buf['geometry'] = df_bike_acc_buf['geometry'].buffer(10, resolution=2)
df_roads['geometry'] = df_roads['geometry'].buffer(15, resolution=2, cap_style=2)
df_speed['geometry'] = df_speed['geometry'].buffer(15, resolution=2, cap_style=2)
df_rva['geometry'] = df_rva['geometry'].buffer(5, resolution=2, cap_style=2)
print("- buffers created")

#  and write to GeoPackage
# - buffered roads
# - buffered speed limits
# - buffered RVA
df_roads.to_file(gpkg_out, layer='buf_roads', driver='GPKG')
df_speed.to_file(gpkg_out, layer='buf_speed', driver='GPKG')
df_rva.to_file(gpkg_out, layer='buf_rva', driver='GPKG')
print("- GeoPackage updated")

# clean-up data:
# - speed: ignore road rank '0' (e.g. Autobahn) and records without speed limit
# - roads: ignore road rank '0' (e.g. Autobahn) and '1' (e.g. park, KGA, ...)
df_speed = pd.merge(df_speed, df_roads[['element_nr', 'rank']], how='left', on="element_nr")
df_speed = df_speed.loc[df_speed['rank'] != 0]
df_speed = df_speed.dropna(subset=['wert_ves'])
df_roads = df_roads.loc[df_roads['rank'] > 1]

# reduce roads by matching entries found in speed limits dataset (matching 'element_nr')
df_roads_wo_speed_idx = pd.concat(
    [df_roads['element_nr'], df_speed['element_nr'], df_speed['element_nr']]).drop_duplicates(keep=False)
df_roads_wo_speed = df_roads.loc[df_roads['element_nr'].isin(df_roads_wo_speed_idx.values)]
print("- all data prepared")

"""
Spatial Joins & concatenation of results
    - accidents x roads (without speed limit)
    - accidents x roads (with speed limit)
    - accidents x roads (partial speed limit)
"""

print("\n--- Execute spatial joins ---")

# Spatial join (1): bike accidents for all roads without speed limit
# -> drop duplicates by considering road rank (highest ranked road 'wins')
df_acc_x_road_wo_speed = gpd.sjoin(df_bike_acc_buf, df_roads_wo_speed, how='inner', predicate='intersects',
                                   rsuffix='road')
df_acc_x_road_wo_speed = df_acc_x_road_wo_speed.sort_values(['rank'], ascending=False).drop_duplicates(subset=['GUID'],
                                                                                                       keep='first')

# Spatial join (2): bike accidents for all roads with speed limit
df_acc_x_speed = gpd.sjoin(df_bike_acc_buf, df_speed, how='inner', predicate='intersects', rsuffix='speed')

# concatenate all accidents that have been found so far (roads with or without speed limit)
df_concat = pd.concat([df_acc_x_road_wo_speed, df_acc_x_speed], axis=0)

# specialty: speed limits sometimes don't cover the entire length of the road, although they have matching IDs
# -> some roads have been ignored in spatial join (1) because it was assumed intersect was found in speed limit DF
# -> extract those missing entries and spatial join (3) again with roads
df_acc_rem_guid = pd.concat([df_bike_acc_buf['GUID'], df_concat['GUID']]).drop_duplicates(keep=False)
df_acc_rem = df_bike_acc_buf.loc[df_bike_acc_buf['GUID'].isin(df_acc_rem_guid.values)]
df_acc_rem_x_road = gpd.sjoin(df_acc_rem, df_roads, how='left', predicate='intersects', rsuffix='road')

# append/concat remaining accidents
df_concat = pd.concat([df_concat, df_acc_rem_x_road], axis=0)
print("- all data joined")

"""
Derive speed limits by considering
    - road type
    - start of speed limit
    - validity of speed limit (day of week & time)
    - highest ranked speed limit wins in case of multiple matches
    -> write results to GeoPackage
"""

# all accidents intersecting roads (with or without speed limit) can be processed now to determine actual speed limit
print("\n--- Derive speed limits ---")
df_concat['speed_der'] = df_concat.apply(deriveSpeedLimit, axis=1).astype('Int64')

# since accidents at intersections have multiple road matches
# -> drop duplicates (road with the highest speed limit 'wins')
# -> remove unnecessary columns
# -> count nan values (no speed limit derived)
df_concat_u = df_concat.sort_values(['speed_der'], ascending=False).drop_duplicates(subset=['GUID'], keep='first')
df_concat_u.drop(['index_road', 'index_speed', 'zeit_t', 'tag_t', 'dat_t', 'laenge'], axis=1, inplace=True)
nan_count_speed = df_concat_u['speed_der'].isna().sum()
print(f"- accidents without derived speed limit: {nan_count_speed}")

#  and write to GeoPackage
# - 10m buffered accidents (incl. speed limit)
df_concat_u.to_file(gpkg_out, layer='buf_acc_speed', driver='GPKG')
print("- GeoPackage updated")

"""
Bike lanes (RVA - Radverkehrsanlagen) 
    - data adjustment: dataset for RVA considers each lane separately, therefore RVA have smaller buffers
                       -> buffer of accident point will be reduced to better assign accident to a lane
    - spatial join all accidents with RVA
    - lowest ranked RVA wins in case of multiple matches
    -> write results to GeoPackage
"""

print("\n--- Determine RVA for accident ---")

# copy dataframe and reduce buffer of accidents
df_acc = df_concat_u.copy()
df_acc['geometry'] = df_acc['geometry'].buffer(-5, resolution=2, cap_style=2)

# spatial join with RVA
# -> drop duplicates (RVA with lowest rank/safety 'wins')
# -> remove unnecessary columns
# -> replace nan values to keep accidents where no RVA could be found
df_acc_x_rva = gpd.sjoin(df_acc, df_rva, how='left', predicate='intersects', lsuffix='road', rsuffix='rva')
df_acc_x_rva = df_acc_x_rva.sort_values(['rank_rva'], ascending=False).drop_duplicates(subset=['GUID'], keep='last')
df_acc_x_rva.drop(['index_rva'], axis=1, inplace=True)
df_acc_x_rva['rank_rva'].fillna(0, inplace=True)
df_acc_x_rva['rank_rva'] = df_acc_x_rva['rank_rva'].astype('Int64')

#  and write to GeoPackage
# - 5m buffered accidents (incl. RVA)
df_acc_x_rva.to_file(gpkg_out, layer='buf_acc_rva', driver='GPKG')
print("- GeoPackage updated")

"""
Add new information to original accident data (still POINT data)
    - Speed
    - RVA
    - speed limit indicator: 
        relevant for bike accident with car, motorbike, truck etc.,
        not relevant for bike only or pedestrians
"""

print("\n--- Extend original accident data ---")
df_bike_acc_out = pd.merge(df_bike_acc, df_concat_u[['GUID', 'speed_der']])
df_bike_acc_out = pd.merge(df_bike_acc_out, df_acc_x_rva[['GUID', 'rank_rva']])
df_bike_acc_out['speed_rel'] = np.where((df_bike_acc_out['IstPKW'] == 1) |
                                        (df_bike_acc_out['IstKrad'] == 1) |
                                        (df_bike_acc_out['IstGkfz'] == 1) |
                                        (df_bike_acc_out['IstSonstig'] == 1),
                                        True, False)
df_bike_acc_out.to_file(gpkg_src, layer='bike_accidents_ext', driver='GPKG')
print("- GeoPackage updated")

print("\n--- End Processing ---")
