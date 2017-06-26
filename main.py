#!/usr/bin/env python3.4
# -*- coding: utf-8 -*-

import urllib.request
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

print('Python version ' + sys.version)
print('Pandas version ' + pd.__version__)

def download_file(name):
    file_extension = name.split('.')[-1]
    url = 'http://exoplanet.eu/catalog/' + file_extension + '/'
    destination = name
    urllib.request.urlretrieve(url, destination)

def download_file_in_folder(name):
    try:
        file = open(name, 'r')
        file.close()
    except IOError as e:
        print(u'Не удалось открыть файл с базой данных по экзопланетам\nЗагрузка из сети.\nЗагрузка началась.')
        download_file(name)
        print('Загрузка завершена.')
    else:
        flag = 'n'
        #flag = input(u'Обновить базу данных по экзопланетам из сети? (y/n): ')
        if (flag.lower() not in {'n', 'not', 'ne', 'nee', 'no', 'not', 'noo'}):
            shutil.copy(name, name + '.old')
            download_file(name)
            print('Обновление выполнено.')


def plot_hist_rasl(x,y,xlabel = 'x', ylabel = 'y',title = 'graph',color = 'g'):
    plt.figure(figsize=(13, 13))
    plt.subplot(211)
    plt.plot(x, y, '.', color = color)
    print('plot "{}" len(x) = {}, len(y) = {}'.format(title,len(x),len(y)))
    plt.ylabel(ylabel, color = 'w')
    plt.title('plot and histagram of ' + title,color = 'w')
    plt.grid(True, color = 'w')

    H, xedges, yedges = np.histogram2d(x, y, bins=(100,30))
    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)

    # Mask zeros
    Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

    # Plot 2D histogram using pcolor
    plt.subplot(212)
    plt.pcolormesh(xedges, yedges, Hmasked)
    plt.xlabel(xlabel,color = 'w')
    plt.ylabel(ylabel,color = 'w')
    plt.title(title,color = 'w')
    plt.grid(True,color = 'w')

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    ax = plt.gca()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="5%", pad=0.05)
    cbar = plt.colorbar(cax=cax, orientation='horizontal')
    cbar.ax.set_xlabel('')

    plt.savefig('plt3.png')
    plt.show()
    #H, xedges, yedges = np.histogram2d(x, y, bins=(50, 50))
    # print(H, len(H))
    #H = H.T  # Let each row list bins with common y range.
    #print(H,len(H))


def plot_rasl(x,y,xlabel = 'x', ylabel = 'y',title = 'graph',color = 'g'):
    fig, ax = plt.subplots()
    ax.plot(x, y, '.', color = color)
    print('plot "{}" len(x) = {}, len(y) = {}'.format(title,len(x),len(y)))
    plt.xlabel(xlabel, color = 'w')
    plt.ylabel(ylabel, color = 'w')
    plt.title(title,color = 'w')
    plt.grid(True,color = 'w')
    plt.savefig('plt2.png')
    plt.show()


def hist_rasl(x,y,xlabel = 'x', ylabel = 'y', title1 = 'graph1', title2 = 'graph2', color = 'g'):
    plt.figure(figsize=(13, 13))

    plt.subplot(211)
    plt.hist(x, bins=50, facecolor=color,)
    #plt.xlabel(xlabel)
    plt.xlabel(title1,color = 'w')
    plt.title('Histogram of ' + title1,color = 'w')
    plt.grid(True,color = 'w')

    plt.subplot(212)
    plt.hist(y, bins=50, facecolor=color)
    plt.xlabel(title2,color = 'w')
    plt.title('Histogram of ' + title2,color = 'w')

    plt.grid(True,color = 'w')
    plt.savefig('plt1.png')
    plt.show()

    #plot_hist_rasl(x,y,xlabel = title1,ylabel = title2, title= title1 + ' & ' + title2)
    #hist2d_plot(x,y)



def hexbin_rasl(x,y,xlabel = 'x', ylabel = 'y',title = 'graph',color = 'g'):
    fig, ax = plt.subplots()
    #ax.plot(x, y, '.', color = color)
    hb = ax.hexbin(x, y, gridsize=20)
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts',color = 'w')

    plt.xlabel(xlabel,color = 'w')
    plt.ylabel(ylabel,color = 'w')
    plt.title(title,color = 'w')
    plt.grid(True,color = 'w')

    plt.savefig('plt4.png')
    plt.show()


def plotm(x,y):
    plot_rasl(x, y)
    #hexbin_rasl(x, y)


def histplanet(x, splot = 111, ylabel = 'y',xlabel = '',color = 'g',title1 = 'planet',mmax = 1, mmin = -1,bins = 'auto'):

    plt.subplot(splot)
    #plt.grid(True)
    histt = plt.hist(x, bins=bins, facecolor=color)
    plt.ylabel(ylabel,color = 'w')
    plt.xlabel(xlabel,color = 'w')
    plt.grid(True,color = 'w')
    plt.axis([mmin, mmax, min(histt[0]),max(histt[0])],'w')
    if splot == 711:
        plt.title('Histogram of star_metallicity')
    if splot == 717:
        plt.xlabel('star_metallicity')



def plotplsnet(x,y):
    plt.figure(figsize=(16, 9))
    #print(max(x), max
    par_xy = [(xi, yi) for xi, yi in zip(x, y) if (not np.isnan(xi)) and (not np.isnan(yi))]

    [histplanet([star_m for star_m,star_pn in par_xy if (star_pn == i)],
                int(max(y) * 100 + 10 + i),
                ylabel = i,
                mmax = max(x),
                mmin = min(x),
                bins = 50
                ) for i in np.arange(1, max(y) + 1)]

    plt.savefig('plt5.png')
    plt.show()

    #hexbin_rasl(x, y)



def korvp(x,y,name1 = 'ololo1',name2 = 'ololo2'):
    par_xy = [(xi,yi)for xi,yi in zip(x,y) if (not np.isnan(xi))and(not np.isnan(yi))]

    print('rasmotrim {} elem of {}.\nst_{} , sr_{} '.format(len(par_xy), len(x), name1,name2))
    hist_rasl([xi for xi,yi in par_xy],[yi for xi,yi in par_xy],title1=name1,title2=name2)
    x_sr = sum([xi for xi, yi in par_xy]) / len(par_xy)
    y_sr = sum([yi for xi, yi in par_xy]) / len(par_xy)
    print('{} and {}'.format(name1, name2))
    print('rasmotrim {} elem of {}.\nst_{} = {}, sr_{} = {}'.format(len(par_xy), len(x), name1,x_sr,name2,y_sr))
    chisl = sum([(xi - x_sr)*(yi - y_sr) for xi,yi in par_xy])
    znam1 = np.power(sum([np.power(xi - x_sr, 2) for xi,yi in par_xy]),0.5)
    znam2 = np.power(sum([np.power(yi - y_sr, 2) for xi,yi in par_xy]),0.5)
    lkk = chisl / znam1 / znam2
    print('lin_koef_korr = {}\nchislitel = {}, znamenatel = {}'.format(lkk,chisl,znam1*znam2))
    '''
    print(sum([(xi - x_sr)*(yi - y_sr) for xi,yi in zip(x,y) if (not np.isnan(xi))and(not np.isnan(yi))])/
          (np.power(sum([np.power((xi - x.sum() / len(x)),2) for xi,yi in zip(x,y) if (not np.isnan(xi))and(not np.isnan(yi))]),0.5)
          * np.power(sum([np.power((yi - y.sum() / len(y)),2) for xi,yi in zip(x,y) if (not np.isnan(xi))and(not np.isnan(yi))]),0.5)),
          sum([(xi - x.sum() / len(x)) * (yi - y.sum() / len(y)) for xi, yi in zip(x, y) if
               (not np.isnan(xi)) and (not np.isnan(yi))]),
          (np.power(sum([np.power((xi - x.sum() / len(x)), 2) for xi,yi in zip(x,y) if (not np.isnan(xi))and(not np.isnan(yi))]), 0.5)
           * np.power(sum([np.power((yi - y.sum() / len(y)), 2) for xi,yi in zip(x,y) if (not np.isnan(xi))and(not np.isnan(yi))]), 0.5)),
            sep='\n')
            '''
    print()

    return 0

from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman', 'Arial', 'Tahoma'
rcParams['font.fantasy'] = 'Times New Roman'

# Изменение параметров рисования (смена чёрного по белому на белое по чёрному)
facecolor = 'k'

rcParams['figure.edgecolor'] = facecolor
rcParams['figure.facecolor'] = facecolor
rcParams['axes.facecolor'] = facecolor
rcParams['axes.edgecolor'] = 'k'
rcParams['grid.color'] = 'w'
rcParams['xtick.color'] = 'w'
rcParams['ytick.color'] = 'w'
rcParams['axes.labelcolor'] = 'w'

rcParams["savefig.facecolor"]='b'
rcParams['savefig.edgecolor']='w'


rcParams['savefig.edgecolor'] = 'k'
rcParams['savefig.facecolor'] = 'k'
rcParams['grid.color'] = 'w'
rcParams['xtick.color'] = 'w'
rcParams['ytick.color'] = 'w'
rcParams['axes.labelcolor'] = 'w'


name = 'exoplanet.eu_catalog.csv'
download_file_in_folder(name)
fixed_df = pd.read_csv(name, sep=',',parse_dates=False, index_col=False)#, parse_dates=['updated'], index_col='updated')

#print(fixed_df[['star_name', '# name','star_metallicity', 'eccentricity', 'inclination']][:5])
print(fixed_df)
srez = fixed_df[['star_name', '# name', 'orbital_period',
                 'eccentricity', 'eccentricity_error_min', 'eccentricity_error_max',
                 'inclination', 'inclination_error_min', 'inclination_error_max',
                 'star_metallicity', 'star_metallicity_error_min', 'star_metallicity_error_max']]
star_srez = fixed_df[['star_name', 'star_metallicity', 'star_metallicity_error_min', 'star_metallicity_error_max']]

z = [[star_srez.values[i].tolist()[0],star_srez.values[i].tolist()[1],star_srez.values[i].tolist()[2],star_srez.values[i].tolist()[3]] for i in np.arange(len(star_srez)-1) if ((star_srez.values[i].tolist()[1]!=star_srez.values[i+1].tolist()[1])or i == len(star_srez))]
#for i in z:
#    print(i)
print(len(z))
#plotm(fixed_df.star_metallicity, fixed_df.eccentricity)
#plotm(fixed_df.star_metallicity, fixed_df.inclination)
korvp(fixed_df.star_metallicity, fixed_df.eccentricity, 'star_metallicity', 'eccentricity')
#korvp(fixed_df.star_metallicity, fixed_df.inclination, 'star_metallicity', 'inclination')

x = [i for i in srez['star_name'].value_counts().to_frame().star_name]
print(len(x))
y = []
for i in srez['star_name'].value_counts().index.tolist():
    f = 0
    for j in srez.values:
        if (f == 0) and (j[0] == i):
            f = 1
            y.append(j[9])
korvp(y,x,'star_metallicity','number of planets')

plotplsnet(y, x)#
