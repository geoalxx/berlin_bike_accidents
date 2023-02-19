import pandas as pd
import geopandas as gpd
import warnings

warnings.filterwarnings("ignore")  # ignore warnings


def setCRS(input_gdf, target_crs):
    if input_gdf.crs is None:
        print("data has no CRS -> set new CRS")
        return input_gdf.set_crs(epsg=target_crs)
    elif input_gdf.crs == target_crs:
        print("no conversion needed conversion")
        return input_gdf
    else:
        print("data has CRS -> do conversion")
        return input_gdf.to_crs(epsg=target_crs)


def convertDayStrToNumber(str):
    if str == 'Sonntag':
        return 1
    elif str == 'Montag':
        return 2
    elif str == 'Dienstag':
        return 3
    elif str == 'Mittwoch':
        return 4
    elif str == 'Donnerstag':
        return 5
    elif str == 'Freitag':
        return 6
    elif str == 'Samstag':
        return 7


def preprocSpeedLimit(row):
    # extract time restriction
    if row['zeit_t'] is None:
        time_from = None
        time_to = None
    else:
        time_from = int(row['zeit_t'][:2])
        time_to = int(row['zeit_t'][8:10])

    # extract day restriction
    if row['tag_t'] is None:
        day_from = None
        day_to = None
    else:
        s = row['tag_t'].split()
        day_from = convertDayStrToNumber(s[0])
        day_to = convertDayStrToNumber(s[2])

    return time_from, time_to, day_from, day_to


def preprocRoadPrio(row):
    # exclude 'Autobahn' -> no bike access allowed
    if row['strassen_1'] == 'A' or row['strassen_2'] == 'AUBA':
        return 0

    # process based on StEP Class
    if row['strassenkl'] in ('I','II','III','IV'):
        return 3

    # process based on OKSTRA Class
    if row['strassen_1'] in ('B','L','S','K','G','E'):
        return 3

    # process based on RBS
    if row['strassen_2'] in ('BRB','STRA','PSTR','SIED','STKG','STPA','ZUFA','BRUE','PLAT'):
        return 3
    elif row['strassen_2'] in ('PARK', 'KGA', 'BRIP','GRAL','FUBR','FUWE','VWEG','INSE','WWEG','STFO'):
        return 1

    return 2


# init
path = 'C:/Python/berlin_bike_accidents/'
dir_input = 'data/'
dir_output = 'output/'
gpkg_name = 'map_data.gpkg'

# target CRS for Berlin (UTM zone 33N)
target_crs = '25833'

if 'G' in ('B','L','S','K','G','E'):
    print("in")
else:
    print("out")

"""
Preprocessing: 

    Accidents
    - load source files
    - extract bike accidents for Berlin
    - delete columns and convert data types
    - convert CRS if required
    - merge into one DF

    Roads, Bike lanes, speed limits
    - load sorce files
    - convert CRS if required

=> combine all data into one geopackage    
"""

# ToDo: convert Strings to ordinal values

print("--- Start Processing ---")

### read accident statistics data
years = [2018, 2019, 2020, 2021]

# read source files
files = []
for year in years:
    print(f'Loading accidents for year {year}')
    df = gpd.read_file(path + dir_input + f'Unfallorte{year}_LinRef.shp',
                       ignore_fields=['UREGBEZ',
                                      'UIDENTSTLA',
                                      'USTRZUSTAN',
                                      'STRZUSTAND',
                                      'ULICHTVERH',
                                      'LINREFX',
                                      'LINREFY',
                                      'XGCSWGS84',
                                      'YGCSWGS84'])
    files.append(df)

# process data frames:
# only keep bike accidents for Berlin
dfs = []
for df in files:
    df = df.loc[(df['IstRad'] == '1') & (df['ULAND'] == '11')]
    df = df.astype({"UJAHR": int,
                    "UMONAT": int,
                    "USTUNDE": int,
                    "UWOCHENTAG": int,
                    "UKATEGORIE": int,
                    "UART": int,
                    "UTYP1": int,
                    "IstRad": int,
                    "IstPKW": int,
                    "IstFuss": int,
                    "IstKrad": int,
                    "IstGkfz": int,
                    "IstSonstig": int
                    })
    dfs.append(setCRS(df, target_crs))

# add bike accidents as layer to GeoPackage
df_bike_acc_all = pd.concat(dfs, ignore_index=True)
df_bike_acc_all['GUID'] = df_bike_acc_all.index
df_bike_acc_all.to_file(path + dir_output + gpkg_name, layer='bike_accidents', driver='GPKG')

### ShapeFiles from FIS broker and add to GeoPackage

# road type data
file_name = 'fis_strassenabschnitte'
df_roads = gpd.read_file(path + dir_input + f'{file_name}.shp',
                         ignore_fields=['strassensc', 'bezirk', 'stadtteil', 'verkehrseb', 'beginnt_be', 'endet_bei_',
                                        'laenge', 'gueltig_vo', 'okstra_id'])
df_roads['prio'] = df_roads.apply(preprocRoadPrio, axis=1)
df_roads = setCRS(df_roads, target_crs)
df_roads.to_file(path + dir_output + gpkg_name, layer=file_name, driver='GPKG')

# speed limit data
file_name = 'fis_tempolimit'
df_speed = gpd.read_file(path + dir_input + f'{file_name}.shp', ignore_fields=['durch_t', 'dann_t'])
for idx, row in df_speed.iterrows():
    df_speed.loc[idx, 'time_from'], df_speed.loc[idx, 'time_to'], df_speed.loc[idx, 'day_from'], df_speed.loc[
        idx, 'day_to'] = preprocSpeedLimit(row)
df_speed['date'] = pd.to_datetime(df_speed['dat_t'], format='%d.%m.%Y')
df_speed = setCRS(df_speed, target_crs)
df_speed.to_file(path + dir_output + gpkg_name, layer=file_name, driver='GPKG')

# bicycle lane data
file_name = 'fis_rva'
df_rva = gpd.read_file(path + dir_input + f'{file_name}.shp',
                       ignore_fields=['segm_bez', 'stst_str', 'stor_name', 'ortstl', 'laenge'])
df_rva = setCRS(df_rva, target_crs)
df_rva.to_file(path + dir_output + gpkg_name, layer=file_name, driver='GPKG')

print("--- End Processing ---")
