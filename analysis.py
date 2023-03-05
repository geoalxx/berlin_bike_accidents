import configparser
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


import warnings


warnings.filterwarnings("ignore")  # ignore warnings

# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get from config.ini
path = config['FILE_SETTINGS']['PATH']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
gpkg_src = path + dir_output + config['FILE_SETTINGS']['GPKG_NAME']

# read preprocessed files
print("--- Read and prepare data ---")
df_bike_acc = gpd.read_file(gpkg_src, layer='bike_accidents_ext')
df_bike_acc['speed_der'] = df_bike_acc['speed_der'].astype('Int64')
df_bike_acc_speed = df_bike_acc.loc[df_bike_acc['speed_rel']]

"""
Create heatmap for absolute values
"""

print("\n--- Create statistics ---")
print("- heat maps with absolute values")

## crosstab with absolute values
crosstab_speed = pd.crosstab(df_bike_acc_speed['UKATEGORIE'],df_bike_acc_speed['speed_der'].fillna(0),margins=True)
crosstab_speed.rename(columns={0: "NA"},inplace=True)
crosstab_rva = pd.crosstab(df_bike_acc['UKATEGORIE'],df_bike_acc['rank_rva'],margins=True)

# print version includes margins (totals)
print(crosstab_speed)
print(crosstab_rva)

## combined heatmaps to show severity for RVA and speed limit
y_axis_labels = ['Todesfolge','Schwerverletzt','Leichtverletzt'] # labels for y-axis
x_axis_labels_rva = ['keine','Busspur','Radfahr-\nstreifen','Radweg'] # labels for x-axis

# start plotting...
f,(ax1,ax2, axcb) = plt.subplots(1,3, figsize=(12, 4), gridspec_kw={'width_ratios':[0.4,0.9,0.05]})
ax1.get_shared_y_axes().join(ax2)

# RVA heatmap (without margins)
g1 = sns.heatmap(crosstab_rva.iloc[0:3,0:4],cbar=False,ax=ax1,
                 cmap='Reds', annot=True, fmt="g", square=True,linewidths=2, norm=LogNorm())
g1.set_ylabel('')
g1.set_xlabel('')
g1.set_xticklabels(x_axis_labels_rva)
g1.set_yticklabels(y_axis_labels, rotation=0,fontsize=8)
g1.xaxis.set_tick_params(labelsize=8)
g1.title.set_text(f'Unfallschwere nach RVA\n(n={len(df_bike_acc)})\n')

# speed limit heatmap (without margins)
g2 = sns.heatmap(crosstab_speed.iloc[0:3,0:9],ax=ax2, cbar_ax=axcb, cmap='Reds',annot=True, fmt="g", square=True,linewidths=2, norm=LogNorm())
g2.set_ylabel('')
g2.set_xlabel('Tempolimit (km/h)',fontsize=10)
g2.set_yticks([])
g2.xaxis.set_tick_params(labelsize=8)
g2.title.set_text(f'Unfallschwere nach Tempolimit\n(n={len(df_bike_acc_speed)})\n')

# finish plotting...
plt.suptitle('Anzahl von Fahrradunfällen', fontsize=16)
plt.tight_layout(w_pad=4.0, pad=2)
plt.show()

"""
Create bar plot for percentage distribution
"""

## crosstab with percentage values
df_bike_acc_speed_sel = df_bike_acc_speed[df_bike_acc_speed['speed_der'].isin([30,50,60])]
crosstab_speed_sel_norm = pd.crosstab(df_bike_acc_speed_sel['UKATEGORIE'],df_bike_acc_speed_sel['speed_der'],normalize='index')
crosstab_rva_norm = pd.crosstab(df_bike_acc['UKATEGORIE'],df_bike_acc['rank_rva'],normalize='index')

print(crosstab_speed_sel_norm)
print(crosstab_rva_norm)

##

fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))


colors = ['#fdccb8', '#fc9a7b', '#f6563d', '#ce1a1e']

crosstab_rva_norm_rev = crosstab_rva_norm[crosstab_rva_norm.columns[::-1]]
labels_rva = reversed(x_axis_labels_rva)

ax = crosstab_rva_norm_rev.plot.barh(stacked=True, color=colors, ax=ax1,  width=0.75)

for bars in ax.containers:
    labels = [f'{x:.1%}' for x in bars.datavalues]
    ax.bar_label(bars, labels=labels,label_type='center',fontsize=8)

rows = len(crosstab_rva_norm_rev)
height = ax.patches[0].get_height()
stacks = len(ax.patches) // rows

for i in range(stacks):
    for j in range(0, rows - 1):
        h0 = np.sum([ax.patches[j + rows * k].get_width() for k in range(0, i + 1)])
        h1 = np.sum([ax.patches[j + 1 + rows * k].get_width() for k in range(0, i + 1)])
        ax.plot([h0, h1],[j + height / 2, j + 1 - height / 2],  color='grey', ls='--', zorder=1,linewidth=0.8)

ax.axes.get_yaxis().set_visible(False)
ax.axis('off')
ax.set_ylabel('')
ax.invert_yaxis()
ax.invert_xaxis()
#ax.yaxis.tick_right()
ax.set(yticks=[0,1,2], yticklabels=[])
for yloc, state in zip([0,1,2], y_axis_labels):
    ax.annotate(state, (0.5, yloc), xycoords=('figure fraction', 'data'),
                     ha='center', va='center')

ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])
ax.title.set_text(f'Unfallschwere nach RVA\n(n={len(df_bike_acc)})')

handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(reversed(handles),
                reversed([w.replace('-\n', '') for w in labels_rva]),
                bbox_to_anchor=(1.01,-0.15),
                loc='lower right',
                ncol = 4,
                frameon=False,
                title='Radverkehrsanlage',
                fontsize=8)
leg._legend_box.align = "right"


colors = ['#fdccb8', '#fb7757', '#ce1a1e']

ax = crosstab_speed_sel_norm.plot.barh(stacked=True, color=colors, ax=ax2,  width=0.75)

for bars in ax.containers:
    labels = [f'{x:.1%}' for x in bars.datavalues]
    ax.bar_label(bars, labels=labels, label_type='center',fontsize=8)

rows = len(crosstab_speed_sel_norm)
height = ax.patches[0].get_height()
stacks = len(ax.patches) // rows

for i in range(stacks):
    for j in range(0, rows - 1):
        h0 = np.sum([ax.patches[j + rows * k].get_width() for k in range(0, i + 1)])
        h1 = np.sum([ax.patches[j + 1 + rows * k].get_width() for k in range(0, i + 1)])
        ax.plot([h0, h1],[j + height / 2, j + 1 - height / 2],  color='grey', ls='--', zorder=1,linewidth=0.8)

ax.axis('off')
ax.axes.get_yaxis().set_visible(False)
ax.set_ylabel('')
ax.invert_yaxis()

ax.title.set_text(f'Unfallschwere nach Tempolimit\n(n={len(df_bike_acc_speed_sel)})')

ax.set_xticklabels([f'{x:.0%}' for x in ax.get_xticks()])

leg = ax.legend(bbox_to_anchor=(-0.01,-0.15),
                loc='lower left',
                ncol = 3,
                frameon=False,
                title='Tempolimit (km/h)',
                fontsize=8)
leg._legend_box.align = "left"


plt.suptitle('Prozentuale Verteilung von Fahrradunfällen', fontsize=16)
plt.tight_layout(w_pad=4, pad=1.5)
plt.show()
##

crosstab_speed_sel_norm = pd.crosstab(df_bike_acc_speed_sel['UKATEGORIE'],df_bike_acc_speed_sel['speed_der'],normalize='columns').sort_index(ascending=False)
crosstab_rva_norm = pd.crosstab(df_bike_acc['UKATEGORIE'],df_bike_acc['rank_rva'],normalize='columns').sort_index(ascending=False)

print(crosstab_speed_sel_norm)
print(crosstab_rva_norm)

##

#crosstab_speed_sel_norm.sort_index(inplace=True,ascending=False)
#crosstab_rva_norm.sort_index(inplace=True,ascending=False)

fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))


g1 = crosstab_rva_norm.T.plot.bar(stacked=True, ax=ax1, legend=False,  width=0.75)

for bars in g1.containers:
    labels = [f'{x:.1%}' for x in bars.datavalues]
    g1.bar_label(bars, labels=labels, label_type='center',fontsize=8)

rows = len(crosstab_rva_norm.T)
width = g1.patches[0].get_width()
stacks = len(g1.patches) // rows

for i in range(stacks):
    for j in range(0, rows - 1):
        h0 = np.sum([g1.patches[j + rows * k].get_height() for k in range(0, i + 1)])
        h1 = np.sum([g1.patches[j + 1 + rows * k].get_height() for k in range(0, i + 1)])
        g1.plot([j + width / 2, j + 1 - width / 2],[h0, h1],  color='grey', ls='--', zorder=1,linewidth=0.8)

#g1.axis('off')
g1.set_xlabel('')
g1.set_xticklabels(x_axis_labels_rva,rotation=0,fontsize=8)



g2 = crosstab_speed_sel_norm.T.plot.bar(stacked=True, ax=ax2, legend=False)

for bars in g2.containers:
    labels = [f'{x:.1%}' for x in bars.datavalues]
    g2.bar_label(bars, labels=labels, label_type='center',fontsize=8)

rows = len(crosstab_speed_sel_norm.T)
width = g2.patches[0].get_width()
stacks = len(g2.patches) // rows

for i in range(stacks):
    for j in range(0, rows - 1):
        h0 = np.sum([g2.patches[j + rows * k].get_height() for k in range(0, i + 1)])
        h1 = np.sum([g2.patches[j + 1 + rows * k].get_height() for k in range(0, i + 1)])
        g2.plot([j + width / 2, j + 1 - width / 2],[h0, h1],  color='C7', ls='--', zorder=1,linewidth=0.8)

g2.xaxis.set_tick_params(labelsize=8, rotation=0)
g2.set_xlabel('Tempolimit (km/h)')

handles, labels = g1.get_legend_handles_labels()
fig.legend([g1, g2], handles=reversed(handles), labels=y_axis_labels,
           loc="upper right")

plt.tight_layout(w_pad=4, pad=1.5)
plt.show()



##



df = crosstab_speed_sel_norm
df = df[df.columns[::-1]]

for bars, color in zip(ax.containers, colors):
    labels = [f'{x:.1%}' for x in bars.datavalues]
    print(labels)
    ax.bar_label(bars, labels=labels, color=color, label_type='center')

crosstab_speed_norm = pd.crosstab(df_bike_acc_speed['UKATEGORIE'],df_bike_acc_speed['speed_der'], normalize='columns').round(4)*100
crosstab_rva_norm = pd.crosstab(df_bike_acc['UKATEGORIE'],df_bike_acc['rank_rva'], normalize='columns').round(4)*100
#print(crosstab)
#print(crosstab_idx)

# heatmap


ax = sns.heatmap(crosstab, annot=True, fmt="g", square=True, norm=LogNorm(),cbar_kws={"shrink": 0.5}, yticklabels=y_axis_labels,linewidths=2)
plt.xlabel('Tempolimit [km/h]',fontsize=12)
plt.ylabel('')
#ax.set_title("Unfallschwere nach Tempolimit (n=333)", loc='left')
#plt.text(x=4.7, y=4.7, s='Sepal Length vs Width', fontsize=16, weight='bold')
#plt.text(x=4.7, y=4.6, s='The size of each point corresponds to sepal width', fontsize=8, alpha=0.75)
#ax.invert_yaxis()
#ax.invert_xaxis()

plt.text(x=0, y=-0.8, s="Unfallschwere nach Tempolimit", fontsize=18, ha="left")
plt.text(x=0, y=-0.5, s=f'(n={len(df_bike_acc_speed)})', fontsize=8, ha="left")

plt.tight_layout()
plt.show()




f,(ax1,ax2, axcb) = plt.subplots(1,3, figsize=(12, 4),
            gridspec_kw={'width_ratios':[0.5,1,0.075]})

ax1.get_shared_y_axes().join(ax2)
g1 = sns.heatmap(crosstab_rva_norm,cbar=False,ax=ax1, cmap='Reds', annot=True, fmt="g", square=True,linewidths=2)
g1.set_ylabel('')
g1.set_xlabel('')
g1.set_xticklabels(x_axis_labels_rva)
g1.set_yticklabels(y_axis_labels, rotation=0,fontsize=8)
g1.xaxis.set_tick_params(labelsize=8)
g1.title.set_text(f'Unfallschwere nach RVA\n(n={len(df_bike_acc)})\n')


g2 = sns.heatmap(crosstab_speed_norm,ax=ax2, cbar_ax=axcb, cmap='Reds',annot=True, fmt="g", square=True,linewidths=2)
g2.set_ylabel('')
g2.set_xlabel('Tempolimit (km/h)',fontsize=10)
g2.set_yticks([])
g2.xaxis.set_tick_params(labelsize=8)
g2.title.set_text(f'Unfallschwere nach Tempolimit\n(n={len(df_bike_acc_speed)})\n')

plt.suptitle('Anzahl von Fahrradunfällen', fontsize=16)

plt.tight_layout(w_pad=4.0, pad=2)
plt.show()







crosstab = pd.crosstab(df_concat_u['UKATEGORIE'],df_concat_u['IstPKW']) #,margins=True)



import plotly.graph_objects as go
from plotly.offline import plot


x = [1, 2, 3]
width = [0.5 for i in x]

data = [go.Bar(name=key, x=x, y=crosstab_idx[key], width=width) for key in crosstab_idx]
layout = go.Layout(barmode= 'stack')

fig = go.Figure(data=data, layout=layout)

for i in range(len(x)-1):
    for j, _ in enumerate(crosstab_idx):
        x1 = x[i]
        x2 = x[i+1]

        y1 = 0
        y2 = 0
        for key in list(crosstab_idx.keys())[:j+1]:
            y1 += crosstab_idx[key][i]
            y2 += crosstab_idx[key][i+1]

        fig.add_trace(go.Scatter(
            x=[x1+width[0]/2, x2-width[0]/2],
            y=[y1, y2],
            mode="lines",
            showlegend=False,
            line={'dash': 'dash', 'color': "#000000"}
        ))

plot(fig)




# https://pysal.org/esda/notebooks/adbscan_berlin_example.html

import contextily as cx
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
from libpysal.cg.alpha_shapes import alpha_shape_auto
from esda.adbscan import ADBSCAN, get_cluster_boundary, remap_lbls

db = df_bike_acc_all
ax = db.plot(markersize=0.1, color='orange')
cx.add_basemap(ax, crs=db.crs.to_string())
plt.show()

db["X"] = db.geometry.x
db["Y"] = db.geometry.y

db.shape[0] * 0.01

# Get clusters
adbs = ADBSCAN(100, 40, pct_exact=0.5, reps=10, keep_solus=True)
np.random.seed(1234)
adbs.fit(db)

ax = db.assign(lbls=adbs.votes["lbls"])\
       .plot(column="lbls",
             categorical=True,
             markersize=2.5,
             figsize=(12, 12)
            )
cx.add_basemap(ax, crs=db.crs.to_string())
plt.show()

polys = get_cluster_boundary(adbs.votes["lbls"], db, crs=db.crs)

ax = polys.plot(alpha=0.5, color="red")
cx.add_basemap(ax, crs=polys.crs.to_string())
plt.show()

gdf = gpd.GeoDataFrame(geometry=polys)

f, axs = plt.subplots(1, 4, figsize=(18, 6))
for i, ax in enumerate(axs):
    # Plot the boundary of the cluster found
    ax = polys[[i]].plot(ax=ax,
                         edgecolor="red",
                         facecolor="none"
                        )
    # Add basemap
    cx.add_basemap(ax,
                   crs=polys.crs.to_string(),
                   url=cx.providers.CartoDB.Voyager,
                   zoom=13
                  )
    # Extra to dim non-cluster areas
    (minX, maxX), (minY, maxY) = ax.get_xlim(), ax.get_ylim()
    bb = Polygon([(minX, minY),
                  (maxX, minY),
                  (maxX, maxY),
                  (minX, maxY),
                  (minX, minY)
                 ])
    gpd.GeoSeries([bb.difference(polys[i])],
                        crs=polys.crs
                       ).plot(ax=ax,
                              color='k',
                              alpha=0.5
                             )
    ax.set_axis_off()
    ax.set_title(f"Cluster {polys[[i]].index[0]}")
plt.show()


x = gpd.sjoin(db, gdf,how='inner', predicate='intersects')

res = x.groupby(['index_right'])['index_right'].count()

gdf.to_file(path + dir_output + gpkg_name_buf, layer='cluster', driver='GPKG')


import plotly.express as px

df = px.data.tips()

df_acc = df_acc_x_rva.loc[(df_acc_x_rva['UKATEGORIE'] < 3) & (df_acc_x_rva['speed_der'] >= 30) ]

fig = px.parallel_categories(df_acc, dimensions=['UART','rank_rva'])
fig.write_html('plots.html')

df_acc_x_rva = gpd.read_file(path + dir_output + gpkg_name_buf, layer='buf_acc_rva')
df = df_acc_x_rva[['UWOCHENTAG','UKATEGORIE','UART','UTYP1','speed_der','rank_rva']]
df = df.loc[(df['speed_der'] == 30) | (df['speed_der'] == 50)]
corr = df.corr(method="spearman")
ax = sns.heatmap(
    corr,
    annot=True,
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
plt.show()