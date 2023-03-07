import configparser
import ast
import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import warnings

warnings.filterwarnings("ignore")  # ignore warnings


def addBarLabels(ax):
    for bars in ax.containers:
        labels = [f'{x:.1%}' for x in bars.datavalues]
        ax.bar_label(bars, labels=labels, label_type='center', fontsize=8)


def addBarTotals(totals):
    for i, t in enumerate(totals):
        ax.text(i, 1.075, f'n={t}', ha='center', fontsize=8, color='grey')


def addVerticalConnectors(ax, rows):
    height = ax.patches[0].get_height()
    stacks = len(ax.patches) // rows

    for i in range(stacks):
        for j in range(0, rows - 1):
            h0 = np.sum([ax.patches[j + rows * k].get_width() for k in range(0, i + 1)])
            h1 = np.sum([ax.patches[j + 1 + rows * k].get_width() for k in range(0, i + 1)])
            ax.plot([h0, h1], [j + height / 2, j + 1 - height / 2], color='grey', ls='--', zorder=1, linewidth=0.8)


def addHorizontalConnectors(ax, rows):
    width = ax.patches[0].get_width()
    stacks = len(ax.patches) // rows

    for i in range(stacks):
        for j in range(0, rows - 1):
            h0 = np.sum([ax.patches[j + rows * k].get_height() for k in range(0, i + 1)])
            h1 = np.sum([ax.patches[j + 1 + rows * k].get_height() for k in range(0, i + 1)])
            ax.plot([j + width / 2, j + 1 - width / 2], [h0, h1], color='grey', ls='--', zorder=1, linewidth=0.8)


# read local config.ini file
config = configparser.ConfigParser()
config.read('config.ini')

# get from config.ini
path = config['FILE_SETTINGS']['PATH']
dir_output = config['FILE_SETTINGS']['DIR_OUTPUT']
gpkg_src = path + dir_output + config['FILE_SETTINGS']['GPKG_NAME']
colors3 = ast.literal_eval(config['CONFIG']['COLORS3'])
colors4 = ast.literal_eval(config['CONFIG']['COLORS4'])

# read preprocessed files
print("--- Read and prepare data ---")
df_bike_acc = gpd.read_file(gpkg_src, layer='bike_accidents_ext')
df_bike_acc['speed_der'] = df_bike_acc['speed_der'].astype('Int64')
df_bike_acc_speed = df_bike_acc.loc[df_bike_acc['speed_rel']]

"""
Create heatmap for absolute values
"""

print("\n--- Create plots ---")
print("- heat maps with absolute values\n")

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
fig,(ax1,ax2, axcb) = plt.subplots(1,3, figsize=(12, 4), gridspec_kw={'width_ratios':[0.4,0.9,0.05]})
ax1.get_shared_y_axes().join(ax2)

# RVA heatmap (without margins)
ax = sns.heatmap(crosstab_rva.iloc[0:3,0:4],cbar=False,ax=ax1,
                 cmap='Reds', annot=True, fmt="g", square=True,linewidths=2, norm=LogNorm())
ax.set_ylabel('')
ax.set_xlabel('')
ax.set_xticklabels(x_axis_labels_rva)
ax.set_yticklabels(y_axis_labels, rotation=0,fontsize=8)
ax.xaxis.set_tick_params(labelsize=8)
ax.title.set_text(f'Unfallschwere nach RVA\n(n={len(df_bike_acc)})\n')

# speed limit heatmap (without margins)
ax = sns.heatmap(crosstab_speed.iloc[0:3,0:9],ax=ax2, cbar_ax=axcb, cmap='Reds',annot=True, fmt="g", square=True,linewidths=2, norm=LogNorm())
ax.set_ylabel('')
ax.set_xlabel('Tempolimit (km/h)',fontsize=10)
ax.set_yticks([])
ax.xaxis.set_tick_params(labelsize=8)
ax.title.set_text(f'Unfallschwere nach Tempolimit\n(n={len(df_bike_acc_speed)})\n')

# finish plotting...
plt.suptitle('Anzahl von Fahrradunfällen', fontsize=16)
plt.tight_layout(w_pad=4.0, pad=2)
plt.show()

"""
Create bar plot for percentage distribution
    - based on accident severity
    - based on road conditions
"""

## crosstab with percentage values (filtered for speed limits 30, 50, 60)
# -> normalized by index
print("\n- stacked horizontal bars for severity\n")

df_bike_acc_speed_sel = df_bike_acc_speed[df_bike_acc_speed['speed_der'].isin([30,50,60])]
crosstab_speed_sel = pd.crosstab(df_bike_acc_speed_sel['UKATEGORIE'],df_bike_acc_speed_sel['speed_der'],margins=True)
crosstab_speed_sel_norm = pd.crosstab(df_bike_acc_speed_sel['UKATEGORIE'],df_bike_acc_speed_sel['speed_der'],normalize='index')
crosstab_rva_norm = pd.crosstab(df_bike_acc['UKATEGORIE'],df_bike_acc['rank_rva'],normalize='index')

print(crosstab_speed_sel_norm)
print(crosstab_rva_norm)

## combined horizontal stacked bar plots based on severity

# RVA plot from right to left -> reverse data
crosstab_rva_norm_rev = crosstab_rva_norm[crosstab_rva_norm.columns[::-1]]
labels_rva = reversed(x_axis_labels_rva)

# get totals for each stacked bar
totals_rva = crosstab_rva['All'][0:3]
totals_speed = crosstab_speed_sel['All'][0:3]

# start plotting...
fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12, 4))

# RVA horizontal bars
ax = crosstab_rva_norm_rev.plot.barh(stacked=True, color=colors4, ax=ax1,  width=0.75)
addBarLabels(ax)
addVerticalConnectors(ax, len(crosstab_rva_norm_rev))
ax.axis('off')
ax.invert_yaxis()
ax.invert_xaxis()

# add legend for RVA
handles, labels = ax.get_legend_handles_labels()
leg = ax.legend(reversed(handles), reversed([w.replace('-\n', '') for w in labels_rva]),
                bbox_to_anchor=(1.01,-0.15), loc='lower right',
                ncol = 4, frameon=False, title='Radverkehrsanlage', fontsize=8)
leg._legend_box.align = "right"

# add y labels and totals in between the bars
for yloc, state in enumerate(y_axis_labels):
    ax.annotate(state, (0.5, yloc), xycoords=('figure fraction', 'data'),
                fontsize=8, ha='center', va='center')
    ax.annotate(f'n={totals_rva[yloc+1]}', (0.44, yloc+.25), xycoords=('figure fraction', 'data'),
                fontsize=8, color='grey', ha='left', va='center')
    ax.annotate(f'n={totals_speed[yloc + 1]}', (0.56, yloc + .25), xycoords=('figure fraction', 'data'),
                fontsize=8, color='grey', ha='right', va='center')

# speed limit horizontal bars
ax = crosstab_speed_sel_norm.plot.barh(stacked=True, color=colors3, ax=ax2,  width=0.75)
addBarLabels(ax)
addVerticalConnectors(ax, len(crosstab_speed_sel_norm))
ax.axis('off')
ax.invert_yaxis()

# add legend for speed limit
leg = ax.legend(bbox_to_anchor=(-0.01,-0.15), loc='lower left',
                ncol=3, frameon=False, title='Tempolimit (km/h)', fontsize=8)
leg._legend_box.align = "left"

# finish plotting...
plt.suptitle('Prozentuale Verteilung von Fahrradunfällen nach Unfallschwere', fontsize=16)
plt.tight_layout(w_pad=4, pad=1)
plt.show()

## crosstab with percentage values (filtered for speed limits 30, 50, 60)
# -> normalized by columns
print("\n- stacked bars for road condition\n")

crosstab_speed_sel_norm = pd.crosstab(df_bike_acc_speed_sel['UKATEGORIE'],df_bike_acc_speed_sel['speed_der'],normalize='columns').sort_index(ascending=False)
crosstab_rva_norm = pd.crosstab(df_bike_acc['UKATEGORIE'],df_bike_acc['rank_rva'],normalize='columns').sort_index(ascending=False)

print(crosstab_speed_sel_norm)
print(crosstab_rva_norm)

## combined horizontal stacked bar plots based on severity

# start plotting...
fig, (ax1,ax2) = plt.subplots(ncols=2, sharey=True, figsize=(12, 4), gridspec_kw={'width_ratios':[0.8,0.6]})

# stacked bars for RVA
ax = crosstab_rva_norm.T.plot.bar(stacked=True, ax=ax1, legend=False, width=0.75, color=colors3)
addBarLabels(ax)
addHorizontalConnectors(ax, len(crosstab_rva_norm.T))
addBarTotals(crosstab_rva.loc['All'][0:4])
ax.set_xlabel('')
ax.set_xticklabels(x_axis_labels_rva, rotation=0, fontsize=8)
ax.set(frame_on=False)
ax.yaxis.set_visible(False)

# stacked bars for speed limit
ax = crosstab_speed_sel_norm.T.plot.bar(stacked=True, ax=ax2, legend=False, width=0.75, color=colors3)
addBarLabels(ax)
addHorizontalConnectors(ax, len(crosstab_speed_sel_norm.T))
addBarTotals(crosstab_speed_sel.loc['All'][0:3])
ax.xaxis.set_tick_params(labelsize=8, rotation=0)
ax.set_xlabel('Tempolimit (km/h)')
ax.set(frame_on=False)
ax.yaxis.set_visible(False)

# create joint legend
handles, labels = ax.get_legend_handles_labels()
leg = fig.legend([ax1, ax2], handles=reversed(handles), labels=y_axis_labels,
           loc=1, bbox_to_anchor=(1,0.8),
           frameon=False, title='Unfallschwere', fontsize=8)
leg._legend_box.align = "left"

# finish plotting...
plt.suptitle('Prozentuale Verteilung von Fahrradunfällen nach Straßenbedingung', fontsize=16)
plt.tight_layout(w_pad=1, pad=1.5)
fig.subplots_adjust(right=0.9)
plt.show()
