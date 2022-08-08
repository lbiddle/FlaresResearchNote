import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import scipy
from scipy.optimize import minimize
import emcee
import corner
import pandas as pd
from os.path import exists
import seaborn as sns



def weighted_violinplot():

    # generate empty dataframe
    df = pd.DataFrame(data=None,columns=['Fuel','MW'])

    # generate fake data for each fuel type and append to df
    for myfuel in ['Biomass','Coal','Hydro','Natural Gas','Oil','Solar','Wind','Other']:
        df = df.append(
                       pd.DataFrame({'Fuel': myfuel,
                            # To make it easy to see that the violinplot of dfw (below)
                            # makes sense, here we'll just use a simple range list from
                            # 0 to 10
                            'MW': np.array(range(11))
                           }),
                       ignore_index=True
                       )

    # I have to recast the data type here to avoid an error when using violinplot below
    df.MW = df.MW.astype(float)

    # create another empty dataframe
    dfw = pd.DataFrame(data=None,columns=['Fuel','MW'])
    # since dfw will be huge, specify data types (in particular, use "category" for Fuel to limit dfw size)
    dfw = dfw.astype(dtype={'Fuel':'category', 'MW':'float'})

    # Define the MW size by which to normalize all of the units
    # Careful: too big -> loss of fidelity in data for small plants
    #          too small -> dfw will need to store an enormous amount of data
    norm = 0.1 # this is in MW, so 0.1 MW = 100 kW

    # Define a var to represent (for each row) how many basic units
    # of size = norm there are in each row
    mynum = 0

    # loop through rows of df
    for index, row in df.iterrows():

        # calculate and store the number of norm MW there are within the MW of each plant
        mynum = int(round(row['MW']/norm))

        # insert mynum rows into dfw, each with Fuel = row['Fuel'] and MW = row['MW']
        dfw = dfw.append(
                       pd.DataFrame({'Fuel': row['Fuel'],
                                     'MW': np.array([row['MW']]*mynum,dtype='float')
                                     }),
                                     ignore_index=True
                        )


    # Set up figure and axes
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey='row')

    # make violinplot
    sns.violinplot(x = 'Fuel',
                   y = 'MW',
                   data=df,
                   inner=None,
                   scale='area',
                   cut=0,
                   linewidth=0.5,
                   ax = ax1
                  )

    # make violinplot
    sns.violinplot(x = 'Fuel',
                   y = 'MW',
                   data=dfw,
                   inner=None,
                   scale='area',
                   cut=0,
                   linewidth=0.5,
                   ax = ax2
                  )

    # loop through the set of tick labels for both axes
    # set tick label size and rotation
    for item in (ax1.get_xticklabels() + ax2.get_xticklabels()):
        item.set_fontsize(8)
        item.set_rotation(30)
        item.set_horizontalalignment('right')

    plt.show()
# weighted_violinplot()



def get_fwhm(x_in, y_in):
    x_interp = np.linspace(np.min(x_in), np.max(x_in), 100000)
    y_interp = np.interp(x_interp, x_in, y_in)

    half = np.max(y_in) / 2.0
    signs = np.sign(np.add(y_interp, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    where_zero_crossings = np.where(zero_crossings)[0]

    try:
        x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
    except:
        print('issue with fwhm determination interp get_fwhm: x1')
        # plt.plot(x_in, y_in)
        # plt.show()
        # import pdb; pdb.set_trace()
        x1 = float('nan')

    try:
        x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])
    except:
        print('issue with fwhm determination interp get_fwhm: x2')
        # plt.plot(x_in, y_in)
        # plt.show()
        # import pdb; pdb.set_trace()
        x2 = float('nan')

    return x2 - x1
def get_fwhm2(x_in, y_in, y_err, do_interp=True):
    if do_interp == True:
        x_interp = np.linspace(np.min(x_in), np.max(x_in), 100000)
        y_interp = np.interp(x_interp, x_in, y_in)

        half = np.max(y_in) / 2.0
        signs = np.sign(np.add(y_interp, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        where_zero_crossings = np.where(zero_crossings)[0]

        x1 = np.mean(x_interp[where_zero_crossings[0]:where_zero_crossings[0] + 1])
        x2 = np.mean(x_interp[where_zero_crossings[1]:where_zero_crossings[1] + 1])

        xdiff = x2 - x1

        average_uncertainty = np.mean(y_err)

        upper = half + average_uncertainty
        signs_upper = np.sign(np.add(y_interp, -upper))
        zero_crossings_upper = (signs_upper[0:-2] != signs_upper[1:-1])
        where_zero_crossings_upper = np.where(zero_crossings_upper)[0]

        x1_upper = np.mean(x_interp[where_zero_crossings_upper[0]:where_zero_crossings_upper[0] + 1])
        x2_upper = np.mean(x_interp[where_zero_crossings_upper[1]:where_zero_crossings_upper[1] + 1])

        xdiff_upper = x2_upper - x1_upper

        lower = half - average_uncertainty
        signs_lower = np.sign(np.add(y_interp, -lower))
        zero_crossings_lower = (signs_lower[0:-2] != signs_lower[1:-1])
        where_zero_crossings_lower = np.where(zero_crossings_lower)[0]

        x1_lower = np.mean(x_interp[where_zero_crossings_lower[0]:where_zero_crossings_lower[0] + 1])
        x2_lower = np.mean(x_interp[where_zero_crossings_lower[1]:where_zero_crossings_lower[1] + 1])

        xdiff_lower = x2_lower - x1_lower

        xdiff_plus = xdiff_lower - xdiff
        xdiff_minus = xdiff - xdiff_upper

        # print(xdiff*24*60, xdiff_plus*24*60, xdiff_minus*24*60)

        # import pdb; pdb.set_trace()


    else:
        half = np.max(y_in) / 2.0
        signs = np.sign(np.add(y_in, -half))
        zero_crossings = (signs[0:-2] != signs[1:-1])
        where_zero_crossings = np.where(zero_crossings)[0]

        x1 = np.mean(x_in[where_zero_crossings[0]:where_zero_crossings[0] + 1])
        x2 = np.mean(x_in[where_zero_crossings[1]:where_zero_crossings[1] + 1])

        xdiff = x2 - x1

        average_uncertainty = np.mean(y_err)

        upper = half + average_uncertainty
        signs_upper = np.sign(np.add(y_in, -upper))
        zero_crossings_upper = (signs_upper[0:-2] != signs_upper[1:-1])
        where_zero_crossings_upper = np.where(zero_crossings_upper)[0]

        x1_upper = np.mean(x_in[where_zero_crossings_upper[0]:where_zero_crossings_upper[0] + 1])
        x2_upper = np.mean(x_in[where_zero_crossings_upper[1]:where_zero_crossings_upper[1] + 1])

        xdiff_upper = x2_upper - x1_upper

        lower = half - average_uncertainty
        signs_lower = np.sign(np.add(y_in, -lower))
        zero_crossings_lower = (signs_lower[0:-2] != signs_lower[1:-1])
        where_zero_crossings_lower = np.where(zero_crossings_lower)[0]

        x1_lower = np.mean(x_in[where_zero_crossings_lower[0]:where_zero_crossings_lower[0] + 1])
        x2_lower = np.mean(x_in[where_zero_crossings_lower[1]:where_zero_crossings_lower[1] + 1])

        xdiff_lower = x2_lower - x1_lower

        xdiff_plus = xdiff_lower - xdiff
        xdiff_minus = xdiff - xdiff_upper

    return xdiff, xdiff_plus, xdiff_minus

def flare_profile_tests(amp=1., tpeak=1., fwhm=(10.*u.min).to(u.d).value):

    B_tests = [-0.251299705922117, -0.251299705922117, -0.251299705922117]
    C_tests = [0.22675974948468916, 0.22675974948468916, 0.22675974948468916]
    D_tests = [0.15551880775110513, 0.15551880775110513, 0.15551880775110513]
    E_tests = [1.2150539528490194, 1.2150539528490194, 1.2150539528490194]
    F_tests = [0.12695865022878844, 0.30, 0.50]

    B_tests = [-0.251299705922117, -0.251299705922117, -0.251299705922117]
    C_tests = [0.22675974948468916, 0.22675974948468916, 0.22675974948468916]
    D_tests = [0.15551880775110513, 0.05, 0.50]
    E_tests = [1.2150539528490194, 1.2150539528490194, 1.2150539528490194]
    F_tests = [0.12695865022878844, 0.12695865022878844, 0.12695865022878844]

    B_tests = [-0.251299705922117, -0.251299705922117]
    C_tests = [0.22675974948468916, 0.22675974948468916]
    D_tests = [0.15551880775110513, 0.5] # , 0.05, 0.50]
    E_tests = [1.2150539528490194, 1.2150539528490194] # , 1.2150539528490194, 1.2150539528490194]
    F_tests = [0.12695865022878844, 0.12695865022878844] # , 0.12695865022878844, 0.12695865022878844]

    colors = ['black', 'orange']

    font_size = 'medium'

    fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
    ax = fig.add_subplot(111)
    for test in range(len(D_tests)):

        A,B,C,D,E,F = [0.9687734504375167,B_tests[test],C_tests[test],
                       D_tests[test],E_tests[test],F_tests[test]]
        print(D + E + F)

        t_fine = np.linspace(0.5, 1.5, 100000)
        x_fine = (t_fine - tpeak) / fwhm
        y_fine = ((1. / 2) * np.sqrt(np.pi) * A * C * F * np.exp(-D * x_fine + ((B / C) + (D * C / 2)) ** 2) *
             scipy.special.erfc(((B - x_fine) / C) + (C * D / 2))) \
            + ((1. / 2) * np.sqrt(np.pi) * A * C * (1 - F) * np.exp(-E * x_fine + ((B / C) + (E * C / 2)) ** 2) *
               scipy.special.erfc(((B - x_fine) / C) + (C * E / 2)))

        if test == 0:
            # plot_label = 'Original= ' + 'D: ' + str(D) + ' E: ' + str(E) + ' F: ' + str(F)
            plot_label = "Original--> D: {D:.2f}  E: {E:.2f}  F: {F:.2f}"
        else:
            # plot_label = 'D: ' + str(D) + ' E: ' + str(E) + ' F: ' + str(F)
            plot_label = "D: {D:.2f}  E: {E:.2f}  F: {F:.2f}"

        ax.plot(x_fine, y_fine*amp, color=colors[test], lw=2, label=plot_label.format(D=D, E=E, F=F))

        line_loc1 = C + E

        ax.plot([line_loc1, line_loc1], [0, 2], color=colors[test], lw=0.75)
        ax.plot([-line_loc1, -line_loc1], [0, 2], color=colors[test], lw=0.75)
        ax.plot([0, 0], [0, 2], color=colors[test], lw=0.75)
        # ax.plot([-1, -1], [0, 2], color="#000000", lw=0.75)
        ax.plot([0, line_loc1], [0.5*A, 0.5*A], color=colors[test], lw=0.75)
        ax.plot([-line_loc1, 0], [0.5*A, 0.5*A], color=colors[test], lw=0.75)
    ax.set_ylabel('y', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('x', fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    ax.set_xlim(-2, 5) # 30)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
    plt.tight_layout()
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()
# flare_profile_tests()

def flare_profile(t, noise, exptime, amp, tpeak, fwhm):

    A,B,C,D,E,F = [0.9687734504375167,-0.251299705922117,0.22675974948468916,
                   0.15551880775110513,1.2150539528490194,0.12695865022878844]
    x = (t - tpeak) / fwhm


    y = ((1. / 2) * np.sqrt(np.pi) * A * C * F * np.exp(-D * x + ((B / C) + (D * C / 2)) ** 2) *
         scipy.special.erfc(((B - x) / C) + (C * D / 2))) \
        + ((1. / 2) * np.sqrt(np.pi) * A * C * (1 - F) * np.exp(-E * x + ((B / C) + (E * C / 2)) ** 2) *
           scipy.special.erfc(((B - x) / C) + (C * E / 2)))

    y_scatter = []
    y_err = []
    for point in range(len(y)):
        spread = noise # (noise / np.sqrt(exptime * 2)) / np.sqrt(abs(y[point]) + 1.)
        y_scatter.append(np.random.normal(y[point], spread))
        y_uncertainty = spread  # (noise**2/np.sqrt(exptime*2)) / np.sqrt(y[point] + 1.)
        y_err.append(y_uncertainty)

    y_scatter = np.array(y_scatter)
    y_err = np.array(y_err)

    t_fine = np.linspace(min(t), max(t), 100000)
    x_fine = (t_fine - tpeak) / fwhm
    y_fine = ((1. / 2) * np.sqrt(np.pi) * A * C * F * np.exp(-D * x_fine + ((B / C) + (D * C / 2)) ** 2) *
         scipy.special.erfc(((B - x_fine) / C) + (C * D / 2))) \
        + ((1. / 2) * np.sqrt(np.pi) * A * C * (1 - F) * np.exp(-E * x_fine + ((B / C) + (E * C / 2)) ** 2) *
           scipy.special.erfc(((B - x_fine) / C) + (C * E / 2)))

    return t_fine, y_fine * amp, y_scatter * amp, y_err
def flare_equation(t, amp, tpeak, fwhm):

    A, B, C, D, E, F = [0.9687734504375167, -0.251299705922117, 0.22675974948468916,
                        0.15551880775110513, 1.2150539528490194, 0.12695865022878844]
    x = (t - tpeak) / fwhm

    y = ((1. / 2) * np.sqrt(np.pi) * A * C * F * np.exp(-D * x + ((B / C) + (D * C / 2)) ** 2) *
         scipy.special.erfc(((B - x) / C) + (C * D / 2))) \
        + ((1. / 2) * np.sqrt(np.pi) * A * C * (1 - F) * np.exp(-E * x + ((B / C) + (E * C / 2)) ** 2) *
           scipy.special.erfc(((B - x) / C) + (C * E / 2)))

    # y = ((1. / 2) * np.sqrt(np.pi) * A * C * F * np.exp(-D * t + ((B / C) + (D * C / 2)) ** 2) *
    #      scipy.special.erfc(((B - t) / C) + (C * D / 2))) \
    #     + ((1. / 2) * np.sqrt(np.pi) * A * C * (1 - F) * np.exp(-E * t + ((B / C) + (E * C / 2)) ** 2) *
    #        scipy.special.erfc(((B - t) / C) + (C * E / 2)))

    return y * amp

def D14_profile(t, noise, ampl, tpeak, fwhm):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    y = np.piecewise(t, [(t <= tpeak) * (t - tpeak) / fwhm > -1., (t > tpeak)],
                         [lambda x: (_fr[0] +  # 0th order
                                     _fr[1] * ((x - tpeak) / fwhm) +  # 1st order
                                     _fr[2] * ((x - tpeak) / fwhm) ** 2. +  # 2nd order
                                     _fr[3] * ((x - tpeak) / fwhm) ** 3. +  # 3rd order
                                     _fr[4] * ((x - tpeak) / fwhm) ** 4.),  # 4th order
                          lambda x: (_fd[0] * np.exp(((x - tpeak) / fwhm) * _fd[1]) +
                                     _fd[2] * np.exp(((x - tpeak) / fwhm) * _fd[3]))]
                         ) * np.abs(ampl)  # amplitude

    y_scatter = []
    y_err = []
    for point in range(len(y)):
        y_scatter.append(np.random.normal(y[point], noise))
        y_err.append(noise)

    y_scatter = np.array(y_scatter)
    y_err = np.array(y_err)

    return y, y_scatter, y_err
def D14_equation(t, ampl, tpeak, fwhm):
    '''
    The Analytic Flare Model evaluated for a single-peak (classical).
    Reference Davenport et al. (2014) http://arxiv.org/abs/1411.3723
    Use this function for fitting classical flares with most curve_fit
    tools.
    Note: this model assumes the flux before the flare is zero centered
    Parameters
    ----------
    t : 1-d array
        The time array to evaluate the flare over
    tpeak : float
        The time of the flare peak
    fwhm : float
        The "Full Width at Half Maximum", timescale of the flare
    ampl : float
        The amplitude of the flare
    Returns
    -------
    flare : 1-d array
        The flux of the flare model evaluated at each time
    '''
    _fr = [1.00000, 1.94053, -0.175084, -2.24588, -1.12498]
    _fd = [0.689008, -1.60053, 0.302963, -0.278318]

    y = np.piecewise(t, [(t <= tpeak) * (t - tpeak) / fwhm > -1., (t > tpeak)],
                         [lambda x: (_fr[0] +  # 0th order
                                     _fr[1] * ((x - tpeak) / fwhm) +  # 1st order
                                     _fr[2] * ((x - tpeak) / fwhm) ** 2. +  # 2nd order
                                     _fr[3] * ((x - tpeak) / fwhm) ** 3. +  # 3rd order
                                     _fr[4] * ((x - tpeak) / fwhm) ** 4.),  # 4th order
                          lambda x: (_fd[0] * np.exp(((x - tpeak) / fwhm) * _fd[1]) +
                                     _fd[2] * np.exp(((x - tpeak) / fwhm) * _fd[3]))]
                         ) * np.abs(ampl)  # amplitude

    return y

def plot_flare(x, y, x_scatter, y_scatter, yerr, xlim, flare_id):

    font_size = 'medium'

    fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color='#000000', lw=4, alpha=0.2, zorder=0)
    ax.scatter(x_scatter, y_scatter, color='#000000', s=np.pi * (3) ** 2, alpha=1, zorder=0)  # , label=label_y[v], zorder=zrdr):
    ax.errorbar(x_scatter, y_scatter, yerr=yerr, fmt='None', ecolor='#000000', elinewidth=2, capsize=2, capthick=2, zorder=0)  # , alpha=1, zorder=zrdr)
    if len(flare_id) > 0:
        ax.scatter(x_scatter[flare_id], y_scatter[flare_id], color='red', s=np.pi * (3) ** 2, alpha=1, zorder=1)

    ax.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (d)', fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    # ax.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
    ax.set_xlim(xlim[0], xlim[1])
    plt.tight_layout()
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()

    import pdb; pdb.set_trace()

def get_start_stop(x, amp, tpeak, fwhm, sampling_rate):
    y = flare_equation(x, amp, tpeak, fwhm)

    where_flare_firsthalf = np.where(y >= 0.005)[0]
    flare_start = x[where_flare_firsthalf[0]]

    rise_from_designated_start = tpeak - flare_start
    # where_start = np.where(x >= flare_start - 12 * rise_from_designated_start)[0]
    # start = x[where_start[0]]

    up_to_start = x[x <= flare_start]
    start = up_to_start[int(-8*sampling_rate*60)]

    where_flare_secondhalf = np.where(x > tpeak + 7 * rise_from_designated_start)[0]
    flare_stop = x[where_flare_secondhalf[0]]
    flare_stop = 1.1

    # where_flare_secondhalf = np.where(x > tpeak + 40 * rise_from_designated_start)[0]
    # stop = x[where_flare_secondhalf[0]]

    beyond_end = x[x >= flare_stop]
    stop = beyond_end[int(8*sampling_rate*60)]


    # # plot the flare

    # y_flare = y[(x >= flare_start) & (x <= flare_stop)]
    # x_flare = x[(x >= flare_start) & (x <= flare_stop)]
    # plt.scatter(x, y, color='#000000', s=np.pi*2**2)
    # plt.scatter(x_flare, y_flare, color='pink', s=np.pi * 2 ** 2) # #ff0066
    # plt.xlim(start-(30.*u.min).to(u.d).value, stop+(30.*u.min).to(u.d).value)
    # plt.show()
    # import pdb; pdb.set_trace()

    return start, stop, flare_start, flare_stop
def identify_flare(x, y, yerr, fl_start, fl_stop, x_truth, y_truth, do_pause):
    y_input = np.copy(y)
    yerr_input = np.copy(yerr)
    x_input = np.copy(x)
    check_range = [0.60, 1.50]
    y_elements = np.arange(0,len(y_input),1)
    y_elements = y_elements[(x >= check_range[0]) & (x <= check_range[1])]
    y = y[(x >= check_range[0]) & (x <= check_range[1])] + 1
    yerr = yerr[(x >= check_range[0]) & (x <= check_range[1])]
    x = x[(x >= check_range[0]) & (x <= check_range[1])]

    N1 = 3
    N2 = 1
    N3 = 2
    track_elements = []
    base_y1 = y[x < 0.99]
    base_y2 = y[x > 1.15]
    base_y = np.concatenate((base_y1,base_y2))
    baseline_mean = np.mean(base_y)
    baseline_std = np.std(base_y)

    start_count = False
    for point_i in range(len(y)-N3):
        segment_satisfy = []

        segment_y = np.array(y[point_i:point_i+N3])
        segment_yerr = np.array(yerr[point_i:point_i+N3])
        segment_elements = np.array(y_elements[point_i:point_i+N3])  # np.arange(point_i, point_i+N3, 1)
        condition_a = segment_y - baseline_mean
        condition_b = (segment_y - baseline_mean) / baseline_std
        condition_c = (segment_y - baseline_mean + segment_yerr) / baseline_std

        for seg_i in range(N3):
            if (condition_a[seg_i] > 0) and (condition_b[seg_i] >= N1) and (condition_c[seg_i] > N2):
                segment_satisfy.append(True)
            else:
                segment_satisfy.append(False)

        # if segment_y[0] == max(y):
        #     print(condition_a)
        #     print(condition_b)
        #     print(condition_c)
        #
        # print(segment_satisfy)
        # print(' ')

        if False in segment_satisfy:
            start_count = False
            continue
        else:
            if start_count == True:
                if segment_elements[-1] - track_elements[-1] == 1:  # Don't let it add elements after the streak breaks
                    track_elements.append(segment_elements[-1])
            if start_count == False:
                for elmt in segment_elements:
                    track_elements.append(elmt)
                start_count = True

    if len(track_elements) > 0:
        for pre_flare_i in range(2):
            if track_elements[0] > 0:
                track_elements = list(np.concatenate(([track_elements[0] - 1],track_elements)))
        for post_flare_i in range(5):
            if track_elements[-1] + 1 <= y_elements[-1]:
                track_elements.append(track_elements[-1] + 1)


    track_elements = np.array(track_elements)[np.array(track_elements) <= y_elements[-1]]

    if do_pause == True:
        # print('baseline_mean:  ' + str(baseline_mean))
        # print('baseline_std:  ' + str(baseline_std))
        plt.plot(x_truth, y_truth, color='#000000', lw=1, zorder=0)
        plt.plot([min(x), max(x)], [0, 0], color='#000000', alpha=0.3, zorder=0)
        # plt.plot(x, y-1)
        plt.errorbar(x, y - 1, yerr=yerr, zorder=0)
        if len(track_elements) > 0:
            plt.scatter(x[track_elements], y[track_elements] - 1, color='red', zorder=1)
        plt.plot([fl_start, fl_start], [min(y-1)-yerr[0], 1.15*max([max(y_truth),max(y-1)])], '--', color='#000000', zorder=0)
        plt.plot([fl_stop, fl_stop], [min(y-1)-yerr[0], 1.15*max([max(y_truth),max(y-1)])], '--', color='#000000', zorder=0)
        plt.xlim(min(x), max(x))
        plt.ylim(min(y-1)-yerr[0], 1.15*max([max(y_truth),max(y-1)]))
        plt.show()
        import pdb; pdb.set_trace()

    print('TRACK ELEMENTS: ')
    print(track_elements)

    return y_input, yerr_input, x_input, track_elements

def downsample(x, cadence_min):
    '''
        Function to convert a finely sampled array to specified sampling rate

        Parameters
        ----------
        x : 1-d array
            The time array quantized in units of seconds
        cadence_min : float
            The desired output sampling rate

        Returns
        -------
        x_downsample : 1-d array
            The time array downsampled to desired sampling rate
    '''

    cadence_interval = (cadence_min) * 60  # to put in terms of seconds because finest sampling done with 1 sec cadence
    start_point = int(np.random.uniform(0,cadence_interval,1)[0]) # 0
    x_downsample = x[start_point::int(cadence_interval)]
    x_downsample = x_downsample.flatten()

    return x_downsample

def get_tpeak_frac(x, tpeak):
    crossing = np.sign(tpeak - x)
    zero_crossing = (crossing[0:-2] != crossing[1:-1])
    where_zero_crossing = np.where(zero_crossing)[0]
    prior_crossing = x[where_zero_crossing[0]]
    post_crossing = x[where_zero_crossing[0] + 1]
    tpeak_frac = (tpeak - prior_crossing) / (post_crossing - prior_crossing)

    return tpeak_frac

def calc_eqdur_truth(start, end, amp, tpeak, fwhm):

    x = np.linspace(start, end, int(((end - start) * u.d).to(u.s).value)*100)
    y = flare_equation(x, amp, tpeak, fwhm)

    eqdur = np.trapz(y, x=x)
    eqdur = (eqdur*u.d).to(u.s).value
    return eqdur
def calc_eqdur_fit(start, end, results, results_upper, results_lower):

    x = np.linspace(start, end, int(((end - start) * u.d).to(u.s).value) * 100)

    y = flare_equation(x, results[0], results[1], results[2])
    eqdur = np.trapz(y, x=x)
    eqdur = (eqdur * u.d).to(u.s).value

    y_upper = flare_equation(x, results[0]+results_upper[0], results[1]+results_upper[1], results[2]+results_upper[2])
    eqdur_upper = np.trapz(y_upper, x=x)
    eqdur_upper = (eqdur_upper * u.d).to(u.s).value

    y_lower = flare_equation(x, results[0]-results_lower[0], results[1]-results_lower[1], results[2]-results_lower[2])
    eqdur_lower = np.trapz(y_lower, x=x)
    eqdur_lower = (eqdur_lower * u.d).to(u.s).value

    return eqdur, eqdur_upper, eqdur_lower
def calc_eqdur_obs(x, y, yerr):

    eqdur = np.trapz(y, x=x)
    eqdur = (eqdur * u.d).to(u.s).value

    y_upper = y + yerr
    eqdur_upper = np.trapz(y_upper, x=x)
    eqdur_upper = (eqdur_upper * u.d).to(u.s).value

    y_lower = y - yerr
    eqdur_lower = np.trapz(y_lower, x=x)
    eqdur_lower = (eqdur_lower * u.d).to(u.s).value

    return eqdur, eqdur_upper, eqdur_lower
def calc_eqdur_obs2(x, y, yerr, SNR, duration):

    eqdur = np.trapz(y, x=x)
    eqdur = (eqdur * u.d).to(u.s).value

    duration = (duration * u.d).to(u.s).value

    eqdur_err = np.sqrt(2.) * ((duration - eqdur) / SNR)
    eqdur_upper = eqdur + eqdur_err
    eqdur_lower = eqdur - eqdur_err

    # y_upper = y + yerr
    # eqdur_upper = np.trapz(y_upper, x=x)
    # eqdur_upper = (eqdur_upper * u.d).to(u.s).value
    #
    # y_lower = y - yerr
    # eqdur_lower = np.trapz(y_lower, x=x)
    # eqdur_lower = (eqdur_lower * u.d).to(u.s).value

    return eqdur, eqdur_upper, eqdur_lower

def calc_rise_decay_cadence_resolution(flare_start, flare_stop, tpeak, sample_rate):

    rise_resolution = (tpeak - flare_start) / ((sample_rate*u.min).to(u.d).value)
    decay_resolution = (flare_stop - tpeak) / ((sample_rate*u.min).to(u.d).value)
    total_resolution = (flare_stop - flare_start) / ((sample_rate*u.min).to(u.d).value)

    return rise_resolution, decay_resolution, total_resolution
def calc_rise_decay_point_resolution(x, flare_start, flare_stop, tpeak):

    x_in_rise = np.where((x >= flare_start) & (x <= tpeak))[0]
    rise_resolution = len(x_in_rise)

    x_in_decay = np.where((x >= tpeak) & (x <= flare_stop))[0]
    decay_resolution = len(x_in_decay)

    x_in_duration = np.where((x >= flare_start) & (x <= flare_stop))[0]
    total_resolution = len(x_in_duration)

    return rise_resolution, decay_resolution, total_resolution

def save_results(lc_parameters, truth, fit, fit_upper, fit_lower, obs, obs_upper, obs_lower,
                 rise_cadence_resolution_true, decay_cadence_resolution_true, total_cadence_resolution_true,
                 rise_cadence_resolution_obs, decay_cadence_resolution_obs, total_cadence_resolution_obs,
                 rise_point_resolution_true, decay_point_resolution_true, total_point_resolution_true,
                 rise_point_resolution_obs, decay_point_resolution_obs, total_point_resolution_obs,
                 flare_start, flare_end, obs_start, obs_end, fit_tpf, true_tpf, fit_profile, Test=False):

    the_data = {'Sampling Rate (min)': [lc_parameters[1]],
                'SNR': [lc_parameters[0]],
                'Impulsive Index': [lc_parameters[2]],
                'Flare Start Time': [flare_start],
                'Flare End Time': [flare_end],
                'Obs Start Time': [obs_start],
                'Obs End Time': [obs_end],
                'Rise Cadence Resolution True': [rise_cadence_resolution_true],
                'Decay Cadence Resolution True': [decay_cadence_resolution_true],
                'Total Cadence Resolution True': [total_cadence_resolution_true],
                'Rise Cadence Resolution Obs': [rise_cadence_resolution_obs],
                'Decay Cadence Resolution Obs': [decay_cadence_resolution_obs],
                'Total Cadence Resolution Obs': [total_cadence_resolution_obs],
                'Rise Point Resolution True': [rise_point_resolution_true],
                'Decay Point Resolution True': [decay_point_resolution_true],
                'Total Point Resolution True': [total_point_resolution_true],
                'Rise Point Resolution Obs': [rise_point_resolution_obs],
                'Decay Point Resolution Obs': [decay_point_resolution_obs],
                'Total Point Resolution Obs': [total_point_resolution_obs],
                'tpeak frac truth': [true_tpf],
                'amp truth': [truth[0]],
                'tpeak truth': [truth[1]],
                'fwhm truth': [truth[2]],
                'eqdur truth': [truth[3]],
                'tpeak frac fit': [fit_tpf],
                'amp fit': [fit[0]],
                'tpeak fit': [fit[1]],
                'fwhm fit': [fit[2]],
                'eqdur fit': [fit[3]],
                'amp fit upper': [fit_upper[0]],
                'tpeak fit upper': [fit_upper[1]],
                'fwhm fit upper': [fit_upper[2]],
                'eqdur fit upper': [fit_upper[3]],
                'amp fit lower': [fit_lower[0]],
                'tpeak fit lower': [fit_lower[1]],
                'fwhm fit lower': [fit_lower[2]],
                'eqdur fit lower': [fit_lower[3]],
                'amp obs': [obs[0]],
                'tpeak obs': [obs[1]],
                'fwhm obs': [obs[2]],
                'eqdur obs': [obs[3]],
                'eqdur obs flare id': [obs[4]],
                'amp obs upper': [obs_upper[0]],
                'tpeak obs upper': [obs_upper[1]],
                'fwhm obs upper': [obs_upper[2]],
                'eqdur obs upper': [obs_upper[3]],
                'eqdur obs flare id upper': [obs_upper[4]],
                'amp obs lower': [obs_lower[0]],
                'tpeak obs lower': [obs_lower[1]],
                'fwhm obs lower': [obs_lower[2]],
                'eqdur obs lower': [obs_lower[3]],
                'eqdur obs flare id lower': [obs_lower[4]],
                }
    if Test == False:
        if fit_profile == "Convolution":
            file_name = '/Users/lbiddle/PycharmProjects/FlaresResearchNote/Fit_Data_Convolution_turmeric.csv'
        if fit_profile == "D14":
            file_name = '/Users/lbiddle/PycharmProjects/FlaresResearchNote/Fit_Data_D14_turmeric.csv'
    if Test == True:
        if fit_profile == "Convolution":
            file_name = '/Users/lbiddle/PycharmProjects/FlaresResearchNote/Files/Fit_Data_Convolution_turmeric_test.csv'
        if fit_profile == "D14":
            file_name = '/Users/lbiddle/PycharmProjects/FlaresResearchNote/Files/Fit_Data_D14_turmeric_test.csv'

    if exists(file_name):

        old_fit_data = pd.read_csv(file_name)

        cols = old_fit_data.columns
        for col_i, col in enumerate(cols):
            current_col = old_fit_data[col].values
            for val_i, val in enumerate(current_col):
                the_data[col].append(val)

    # for key in the_data.keys():
    #     print(key + ':  ' + str(len(the_data[key])))
    #
    # import pdb; pdb.set_trace()

    df_fit_data = pd.DataFrame(the_data)
    df_fit_data.to_csv(file_name, index=False)



def log_likelihood(theta, t, y, yerr):

    amp, tpeak, fwhm, log_f = theta
    # A, B, C, D, E, F = theta

    # model = 0.5 * np.sqrt(np.pi) * A * C * (
    #         F * (np.exp(-D * t + (B / C + D * C / 2) ** 2)) * scipy.special.erfc((B - t) / C + (D * C / 2)) +
    #         (1. - F) * (np.exp(-E * t + (B / C + E * C / 2) ** 2)) * scipy.special.erfc((B - t) / C + (E * C / 2))
    # )

    model = flare_equation(t, amp, tpeak, fwhm)

    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))
def log_likelihood_D14(theta, t, y, yerr):

    amp, tpeak, fwhm, log_f = theta
    # A, B, C, D, E, F = theta

    # model = 0.5 * np.sqrt(np.pi) * A * C * (
    #         F * (np.exp(-D * t + (B / C + D * C / 2) ** 2)) * scipy.special.erfc((B - t) / C + (D * C / 2)) +
    #         (1. - F) * (np.exp(-E * t + (B / C + E * C / 2) ** 2)) * scipy.special.erfc((B - t) / C + (E * C / 2))
    # )

    model = D14_equation(t, amp, tpeak, fwhm)

    sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
    return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))


def log_prior(theta, x, y):
    amp, tpeak, fwhm, log_f = theta
    min_fwhm = 0.001*(1.0*u.min).to(u.d).value
    if (np.isfinite(amp) == True) and (np.isfinite(tpeak) == True) and (np.isfinite(fwhm) == True) and (np.isfinite(log_f) == True):
        # if (0 < amp < 100) and (min(x) < tpeak < max(x)) and (0 < fwhm < (max(x) - min(x))) and (-50. < log_f < 1.):
        if (np.mean(y) < amp) and (min(x) < tpeak < max(x)) and (0.25*np.diff(x)[0] < fwhm < (max(x) - min(x))) and (-100. < log_f < 1.):
            return 0.0
    else:
        import pdb; pdb.set_trace()
    return -np.inf
# def log_prior(theta, x):
#     # A, B, C, D, E, F, tpeak, fwhm, log_f = theta
#     amp, tpeak, fwhm, log_f = theta
#     if (np.isfinite(amp) == True) and (np.isfinite(tpeak) == True) and (np.isfinite(fwhm) == True) and (np.isfinite(log_f) == True):
#         if (amp > 0) and (tpeak > 0) and (fwhm > 0.5*np.diff(x)[0]) and (log_f < 1.0):
#             return 0.0
#     return -np.inf
def log_probability(theta, x, y, yerr):
    lp = log_prior(theta, x, y)
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(log_likelihood(theta, x, y, yerr)):
        return -np.inf
    if np.isnan(log_likelihood(theta, x, y, yerr)):
        print(' ')
        print('log likelihood returned NaN')
        print(' ')
        import pdb; pdb.set_trace()

    return lp + log_likelihood(theta, x, y, yerr)
def log_probability_D14(theta, x, y, yerr):
    lp = log_prior(theta, x, y)
    if not np.isfinite(lp):
        return -np.inf
    if not np.isfinite(log_likelihood_D14(theta, x, y, yerr)):
        return -np.inf
    if np.isnan(log_likelihood_D14(theta, x, y, yerr)):
        print(' ')
        print('log likelihood D14 returned NaN')
        print(' ')
        import pdb; pdb.set_trace()

    return lp + log_likelihood_D14(theta, x, y, yerr)


np.random.seed(444)

sampling_rate = 1.
x_fine_initial = np.linspace(0., 15., int((15.*u.d).to(u.s).value))

test_SNRs = [4., 8., 16., 32., 64., 128., 256.]
# test_sampling_rates = [60., 55., 45., 40., 35., 30., 25., 20., 15., 12., 10., 8., 5., 2., 1., 1./2.]
test_sampling_rates = list(np.random.uniform(1./2., 50.0, 100000))
# test_sampling_rates = [11., 5., 2., 1., 1./2.]
# test_impulsivities = [5760., 2880., 1440., 720., 360., 180., 90., 45., 32., 24., 12.]
test_impulsivities = [180.]
# test_impulsivities = test_impulsivities[::-1]
# test_sampling_rates = test_sampling_rates[::-1]
# test_SNRs = test_SNRs[::-1]

amp_true = 1.
tpeak_true = 1.0
f_true = 0.05

num_walkers = 32
nsteps = 5000

do_plots = False
for bep in range(1):
    for rate_i,rate in enumerate(test_sampling_rates):
        for SNR in test_SNRs:
            for imp in test_impulsivities:

                print('\n\n################################################')
                print('#      ' + "{:06d}".format(rate_i + 1))
                print('RATE:  ' + str(rate))
                print('SNR:   ' + str(SNR))
                print('################################################')
                print(' ')


                # fwhm_true = ((amp_true/imp) * u.min).to(u.d).value
                fwhm_true = amp_true/imp
                truth = np.array([amp_true, tpeak_true, fwhm_true, np.log(f_true)])

                lc_pars = [SNR, rate, imp]

                start_time, end_time, flare_start_time, flare_stop_time = get_start_stop(x=x_fine_initial, amp=amp_true, tpeak=tpeak_true, fwhm=fwhm_true, sampling_rate=rate)

                # if flare_stop_time - flare_start_time >= 3*(rate*u.min).to(u.d).value:

                x_fine = np.linspace(start_time, end_time, int(((end_time - start_time) * u.d).to(u.s).value))


                x_data = downsample(x=x_fine, cadence_min=rate)
                err = 0.9687734504375167/SNR  # 1./SNR

                x_truth, y_truth, y_data, yerr_data = flare_profile(t=x_data, noise=err, exptime=rate, amp=amp_true, tpeak=tpeak_true, fwhm=fwhm_true)


                # if (rate_i+1 == 10) and (SNR >= 128.):
                #     y_data_test, yerr_data_test, x_data_test, flare_id = identify_flare(x=x_data, y=y_data, yerr=yerr_data, fl_start=flare_start_time,
                #                               fl_stop=flare_stop_time, x_truth=x_truth, y_truth=y_truth, do_pause=True)
                # else:
                y_data_test, yerr_data_test, x_data_test, flare_id = identify_flare(x=x_data, y=y_data, yerr=yerr_data,
                                                                                    fl_start=flare_start_time,
                                                                                    fl_stop=flare_stop_time,
                                                                                    x_truth=x_truth, y_truth=y_truth,
                                                                                    do_pause=False)
                print('FLARE ID:  ')
                print(flare_id)
                print(' ')

                if len(flare_id) > 0:
                    try:
                        obs_start_time = min(x_data[flare_id])
                    except:
                        print('issue with obs start time')
                        import pdb; pdb.set_trace()
                    obs_end_time = max(x_data[flare_id])
                else:
                    obs_start_time = float('nan')
                    obs_end_time = float('nan')

                rise_cadence_resolution_true, decay_cadence_resolution_true, total_cadence_resolution_true = calc_rise_decay_cadence_resolution(flare_start=flare_start_time, flare_stop=flare_stop_time, tpeak=tpeak_true, sample_rate=rate)
                if len(flare_id) > 0:
                    rise_cadence_resolution_obs, decay_cadence_resolution_obs, total_cadence_resolution_obs = calc_rise_decay_cadence_resolution(flare_start=obs_start_time, flare_stop=obs_end_time, tpeak=x_data[flare_id][y_data[flare_id] == max(y_data[flare_id])][0], sample_rate=rate)
                else:
                    rise_cadence_resolution_obs = float('nan')
                    decay_cadence_resolution_obs = float('nan')
                    total_cadence_resolution_obs = float('nan')

                rise_point_resolution_true, decay_point_resolution_true, total_point_resolution_true = calc_rise_decay_point_resolution(x=x_data, flare_start=flare_start_time, flare_stop=flare_stop_time, tpeak=tpeak_true)
                if len(flare_id) > 0:
                    rise_point_resolution_obs, decay_point_resolution_obs, total_point_resolution_obs = calc_rise_decay_point_resolution(x=x_data[flare_id], flare_start=obs_start_time, flare_stop=obs_end_time, tpeak=x_data[flare_id][y_data[flare_id] == max(y_data[flare_id])][0])
                else:
                    rise_point_resolution_obs = float('nan')
                    decay_point_resolution_obs = float('nan')
                    total_point_resolution_obs = float('nan')

                tpeak_frac_truth = get_tpeak_frac(x=x_data, tpeak=tpeak_true)

                eqdur_truth = calc_eqdur_truth(start=flare_start_time, end=flare_stop_time, amp=amp_true, tpeak=tpeak_true, fwhm=fwhm_true)
                true_pars = [amp_true, tpeak_true, fwhm_true, eqdur_truth]

                if do_plots == True:
                    do_test_plot = True
                    if do_test_plot == True:
                        plot_flare(x=x_truth, y=y_truth, x_scatter=x_data, y_scatter=y_data, yerr=yerr_data, xlim=[start_time, end_time], flare_id=flare_id)

                if len(x_data) > 5:
                    x_data_max_elmt = np.where(x_data == x_data[y_data == max(y_data)])[0][0]
                    post_peak_data = x_data[x_data > x_data[x_data_max_elmt]]

                    # if len(post_peak_data) > 3:
                    #     proceed_to_test = True
                    # else:
                    #     proceed_to_test = False

                    if len(flare_id) > 0:
                        proceed_to_test = True
                    else:
                        proceed_to_test = False

                    if proceed_to_test == True:

                        y_data_max_values = y_data[x_data_max_elmt:x_data_max_elmt + 3]

                        baseline_data1 = y_data[x_data < flare_start_time]
                        baseline_data2 = y_data[x_data > flare_stop_time]
                        baseline_data = np.concatenate((np.array(baseline_data1), np.array(baseline_data2)))

                        # baseline_data = y_data[x_data < flare_start_time]

                        if (y_data_max_values[0] > 3 * np.std(baseline_data)) and (y_data_max_values[1] > 3 * np.std(baseline_data)) and (y_data_max_values[2] > 3 * np.std(baseline_data)):
                            proceed = True
                        else:
                            proceed = False
                    else:
                        proceed = False

                    # import pdb; pdb.set_trace()

                    if proceed == True:  #  and (y_data_max_values[2] > 3*np.std(y_data)):

                        dur = flare_stop_time - flare_start_time
                        obs_dur = max(x_data[flare_id]) - min(x_data[flare_id])
                        eqdur_obs, eqdur_obs_upper, eqdur_obs_lower = calc_eqdur_obs2(x=x_data, y=y_data, yerr=yerr_data, SNR=SNR, duration=dur)
                        eqdur_obs_flare_id, eqdur_obs_flare_id_upper, eqdur_obs_flare_id_lower = calc_eqdur_obs2(x=x_data[flare_id], y=y_data[flare_id], yerr=yerr_data[flare_id], SNR=SNR, duration=obs_dur)
                        # eqdur_obs1, eqdur_obs_upper1, eqdur_obs_lower1 = calc_eqdur_obs(x=x_data, y=y_data, yerr=yerr_data)
                        # import pdb; pdb.set_trace()
                        fwhm_issue = False
                        try:
                            fwhm_obs, fwhm_obs_upper, fwhm_obs_lower = get_fwhm2(x_data[flare_id], y_data[flare_id], yerr_data[flare_id])
                        except:
                            fwhm_obs = float('nan')
                            fwhm_obs_upper = float('nan')
                            fwhm_obs_lower = float('nan')
                            fwhm_issue = True
                            # import pdb; pdb.set_trace()

                        obs_pars = [max(y_data), x_data[y_data == max(y_data)][0], fwhm_obs, eqdur_obs, eqdur_obs_flare_id]
                        obs_pars_upper = [max(y_data) + yerr_data[y_data == max(y_data)][0], x_data[y_data == max(y_data)][0], fwhm_obs_upper, abs(eqdur_obs_upper - eqdur_obs), abs(eqdur_obs_flare_id_upper - eqdur_obs_flare_id)]
                        obs_pars_lower = [max(y_data) - yerr_data[y_data == max(y_data)][0], x_data[y_data == max(y_data)][0], fwhm_obs_lower, abs(eqdur_obs_lower - eqdur_obs), abs(eqdur_obs_flare_id_lower - eqdur_obs_flare_id)]


                        # ---------------------

                        # Calculate Maximum Likelihood Estimation

                        if fwhm_issue == False:
                            guess_fwhm = get_fwhm2(x_data, y_data, yerr_data)[0]
                        if fwhm_issue == True:
                            guess_fwhm = 0.5*(end_time - start_time)

                        nll = lambda *args: -log_likelihood(*args)
                        initial = np.array([max(y_data), np.random.normal(x_data[y_data == max(y_data)][0], (rate * u.min).to(u.d).value, 1)[0], guess_fwhm, 0.01])
                        soln = minimize(nll, initial, args=(x_data, y_data, yerr_data))
                        amp_ml, tpeak_ml, fwhm_ml, log_f_ml = soln.x

                        nll_D14 = lambda *args: -log_likelihood_D14(*args)
                        initial = np.array([max(y_data), np.random.normal(x_data[y_data == max(y_data)][0], (rate * u.min).to(u.d).value, 1)[0], guess_fwhm, 0.01])
                        soln_D14 = minimize(nll_D14, initial, args=(x_data, y_data, yerr_data))
                        amp_ml_D14, tpeak_ml_D14, fwhm_ml_D14, log_f_ml_D14 = soln_D14.x

                        # print(' ')
                        # print("Maximum likelihood estimates:")
                        # print("amplitude = {0:.3f}".format(amp_ml))
                        # print("tpeak = {0:.3f}".format(tpeak_ml))
                        # print("fwhm (d) = {0:.6f}".format(fwhm_ml))
                        # print("fwhm (min) = {0:.5f}".format((fwhm_ml*u.d).to(u.min).value))
                        # print("f = {0:.5f}".format(log_f_ml))
                        # print(' ')
                        # print("D14 Maximum likelihood estimates:")
                        # print("amplitude_D14 = {0:.3f}".format(amp_ml_D14))
                        # print("tpeak_D14 = {0:.3f}".format(tpeak_ml_D14))
                        # print("fwhm_D14 (d) = {0:.6f}".format(fwhm_ml_D14))
                        # print("fwhm D14 (min) = {0:.5f}".format((fwhm_ml_D14 * u.d).to(u.min).value))
                        # print("f_D14 = {0:.5f}".format(log_f_ml_D14))
                        # print(' ')
                        # print("Data derived estimates:")
                        # print("amplitude_data = {0:.3f}".format(initial[0]))
                        # print("tpeak_data = {0:.3f}".format(initial[1]))
                        # print("fwhm_data (d) = {0:.6f}".format(initial[2]))
                        # print("fwhm_data (min) = {0:.5f}".format((initial[2] * u.d).to(u.min).value))
                        # print("f_data = {0:.5f}".format(initial[3]))
                        #
                        # print(' ')
                        # print('SNR: ' + str(SNR))
                        # print('Sampling Rate: ' + str(rate))
                        # print('Imp Index: ' + str(imp))
                        # print('FWHM (d): ' + str(fwhm_true))
                        # print('FWHM (min): ' + str((fwhm_true * u.d).to(u.min).value))
                        # print(' ')

                        fit = flare_equation(x_fine, amp_ml, tpeak_ml, fwhm_ml)
                        fit_D14 = D14_equation(x_fine, amp_ml_D14, tpeak_ml_D14, fwhm_ml_D14)

                        if do_plots == True:

                            print(rate)

                            y_range = max([max(y_data + yerr_data), max(fit), max(fit_D14)]) - min([min(y_data - yerr_data), min(fit), min(fit_D14)])

                            ymin = -abs(1.10 * min([min(y_data - yerr_data), min(fit), min(fit_D14)])) - 0.10*y_range
                            ymax = 1.05 * y_range
                            font_size = 'medium'
                            plt.close()
                            plt.plot(x_truth, y_truth, color="#000000", alpha=0.3, lw=3, label="Truth")
                            # plt.scatter(x_data, y_data, color='blue', s=np.pi*2**2, label="Truth With Scatter")
                            plt.errorbar(x_data, y_data, yerr=yerr_data, fmt=".k", capsize=0, label="Simulation", zorder=0)
                            plt.plot(x_fine, fit, color="red", label="Max L. Continuous")
                            plt.plot(x_fine, fit_D14, color="blue", label="Max L. Piecewise")
                            plt.vlines([flare_start_time, flare_stop_time], ymin=ymin, ymax=ymax, color="#000000", lw=0.75)
                            plt.legend(fontsize=font_size)
                            plt.xlim(start_time, end_time)
                            plt.ylim(ymin, ymax)
                            title_str = 'Maximum Likelihood Estimates For Both Templates\nSampling Rate: {0:.2f} min  SNR: {1:.1f}'
                            plt.title(title_str.format(rate, SNR), fontsize='medium', style='normal', family='sans-serif')
                            plt.ylabel('y', fontsize=font_size, style='normal', family='sans-serif')
                            plt.xlabel('t', fontsize=font_size, style='normal', family='sans-serif')
                            plt.legend(loc='upper right', fontsize='small', framealpha=1.0, fancybox=False, frameon=True)
                            plt.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
                            plt.show()

                        # ---------------------

                        # Do MCMC

                        initial[-1] = np.log(initial[-1])

                        which_input = initial  # soln.x
                        which_input_d14 = initial  # soln_D14.x


                        for decision in [False, True]:

                            do_fit_D14 = decision

                            guess_D14 = False
                            if do_fit_D14 == True:
                                guess_D14 = True
                                pos = which_input_d14 + 1e-4 * np.random.randn(num_walkers, len(truth))
                                for pos_i in pos:
                                    # pos_i[0] = abs(soln_D14.x[0] + 1e0 * np.random.randn(1, 1)[0][0])
                                    pos_i[0] = abs(max(y_data) + 1e0 * np.random.randn(1, 1)[0][0])
                                    pos_i[1] = np.random.normal(x_data[y_data == max(y_data)][0], (rate * u.min).to(u.d).value, 1)[0]  # soln_D14.x[1] + 1e-6 * np.random.randn(1, 1)[0][0]
                                    pos_i[2] = abs(soln_D14.x[2] + 1e-3 * np.random.randn(1, 1)[0][0])
                                    too_long = 0
                                    while (pos_i[2] >= end_time - start_time) and (too_long < 20):
                                        pos_i[2] = abs(which_input_d14[2] + 1e0 * np.random.randn(1, 1)[0][0])
                                        too_long += 1
                                    if too_long >= 20:
                                        pos_i[2] = 0.3*(end_time - start_time)
                                    if pos_i[0] >= 20:
                                        pos_i[0] = abs(max(y_data) + 1e0 * np.random.randn(1, 1)[0][0])
                                        print(' ')
                                        print('soln_D14.x[0] is beeg.')
                                        print(' ')
                                        # import pdb; pdb.set_trace()

                                nwalkers, ndim = pos.shape

                                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability_D14, args=(x_data, y_data, yerr_data))
                                try:
                                    # np.random.seed(42)
                                    sampler.run_mcmc(pos, nsteps, progress=True)
                                except:
                                    print(' ')
                                    print('Issue with D14 sampler.run_mcmc()')
                                    print(' ')
                                    import pdb; pdb.set_trace()
                                # sampler.run_mcmc(pos, nsteps, progress=True)
                            else:
                                pos = which_input + 1e-4 * np.random.randn(num_walkers, len(truth))
                                for pos_i in pos:
                                    # pos_i[0] = abs(soln.x[0] + 1e0 * np.random.randn(1, 1)[0][0])
                                    pos_i[0] = abs(max(y_data) + 1e0 * np.random.randn(1, 1)[0][0])
                                    pos_i[1] = np.random.normal(x_data[y_data == max(y_data)][0], (rate * u.min).to(u.d).value, 1)[0]
                                    # pos_i[1] = soln.x[1] + 1e-6 * np.random.randn(1, 1)[0][0]
                                    pos_i[2] = abs(which_input[2] + 1e-3 * np.random.randn(1, 1)[0][0])
                                    too_long = 0
                                    while (pos_i[2] >= end_time - start_time) and (too_long < 20):
                                        pos_i[2] = abs(soln.x[2] + 1e0 * np.random.randn(1, 1)[0][0])
                                        too_long += 1
                                    if too_long >= 20:
                                        pos_i[2] = 0.3 * (end_time - start_time)
                                    if pos_i[0] >= 20:
                                        pos_i[0] = abs(max(y_data) + 1e0 * np.random.randn(1, 1)[0][0])
                                        print(' ')
                                        print('soln.x[0] is beeg.')
                                        print(' ')
                                        # import pdb; pdb.set_trace()

                                nwalkers, ndim = pos.shape

                                sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(x_data, y_data, yerr_data))
                                try:
                                    # np.random.seed(42)
                                    sampler.run_mcmc(pos, nsteps, progress=True)
                                except:
                                    # font_size = 'medium'
                                    # plt.plot(x_truth, y_truth, color="#000000", alpha=0.3, lw=3, label="Truth")
                                    # # plt.scatter(x_data, y_data, color='blue', s=np.pi*2**2, label="Truth With Scatter")
                                    # plt.errorbar(x_data, y_data, yerr=yerr_data, fmt=".k", capsize=0, label="Scatter", zorder=0)
                                    # plt.plot(x_fine, fit, color="red", label="Maximum Likelihood")
                                    # plt.plot(x_fine, fit_D14, color="blue", label="Maximum Likelihood D14")
                                    # plt.legend(fontsize=font_size)
                                    # plt.xlim(start_time, end_time)
                                    # plt.ylabel('y', fontsize=font_size, style='normal', family='sans-serif')
                                    # plt.xlabel('t', fontsize=font_size, style='normal', family='sans-serif')
                                    # plt.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
                                    # plt.show()

                                    # soln.x[0] *= 0.5
                                    # soln.x[1] -= (rate*u.min).to(u.d).value
                                    # soln.x[-1] = np.log(0.5)

                                    print(' ')
                                    print('Issue with sampler.run_mcmc()')
                                    print(' ')
                                    import pdb; pdb.set_trace()


                            labels = ["Amplitude", "tpeak", "FWHM", "log(f)"]

                            if do_plots == True:
                                font_size = 'medium'
                                fig, axes = plt.subplots(len(truth), figsize=(7.5, 6.5), sharex=True)
                                samples = sampler.get_chain()
                                for i in range(ndim):
                                    ax = axes[i]
                                    ax.plot(samples[:, :, i], "k", alpha=0.3)
                                    ax.set_xlim(0, len(samples))
                                    ax.set_ylabel(labels[i], fontsize=font_size, style='normal', family='sans-serif')
                                    ax.yaxis.set_label_coords(-0.1, 0.5)
                                    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
                                axes[-1].set_xlabel("Step Number", fontsize=font_size, style='normal', family='sans-serif')
                                plt.tight_layout()
                                plt.show()

                                # tau = sampler.get_autocorr_time()
                                #
                                # burnin = int(2 * np.max(tau))
                                # thin = int(0.5 * np.min(tau))
                            thin = 60
                            burnin = int(0.15 * nsteps)

                            flat_samples = sampler.get_chain(discard=burnin, flat=True, thin=thin)
                            # print(flat_samples.shape)

                            transposed_samples = np.transpose(flat_samples)
                            flat_samples2 = np.transpose(transposed_samples[0:-1])  # samples without the stupid fractional shift thing

                            if do_plots == True:
                                fig = corner.corner(flat_samples2, labels=labels[0:-1], truths=[amp_true, tpeak_true, fwhm_true])
                                plt.show()


                            fit_results = []
                            fit_results_upper = []
                            fit_results_lower = []

                            fit_results = []
                            fit_results_upper = []
                            fit_results_lower = []

                            for i in range(flat_samples2.shape[1]):
                                # mcmc = np.percentile(flat_samples2[:, i], [16, 50, 84])
                                mcmc = np.percentile(flat_samples2[:, i], [16, 50, 84])
                                qdiff = np.diff(mcmc)
                                txt = labels[i] + " = " + str(mcmc[1]) + " +" + str(qdiff[1]) + " -" + str(qdiff[0])
                                print(txt)

                                fit_results.append(mcmc[1])
                                fit_results_upper.append(qdiff[1])
                                fit_results_lower.append(qdiff[0])


                            percentiles = np.array([fleeb for fleeb in range(85-16)]) + 16
                            sample_amps = np.percentile(flat_samples2[:, 0], percentiles)
                            sample_tpeaks = np.percentile(flat_samples2[:, 1], percentiles)
                            sample_fwhms = np.percentile(flat_samples2[:, 2], percentiles)
                            if do_plots == True:
                                # inds = np.array([mlem for mlem in range(int(0.10 * len(flat_samples2)))]) + int(0.45*len(flat_samples2))
                                # for ind in inds:
                                for ind in range(len(sample_amps)):
                                    sample = [sample_amps[ind], sample_tpeaks[ind], sample_fwhms[ind]]
                                    if do_fit_D14 == True:
                                        sample_fit = D14_equation(x_fine, sample[0], sample[1], sample[2])
                                    else:
                                        sample_fit = flare_equation(x_fine, sample[0], sample[1], sample[2])
                                    plt.plot(x_fine, sample_fit, "C1", alpha=0.1, zorder=1)
                                plt.errorbar(x_data, y_data, yerr=yerr_data, fmt=".k", capsize=0, zorder=0)
                                plt.plot(x_truth, y_truth, "#000000", label="Truth", zorder=0)
                                if do_fit_D14 == True:
                                    plt.plot(x_fine, fit_D14, color="blue", label="Maximum likelihood")
                                    fit_median = D14_equation(x_fine, fit_results[0], fit_results[1], fit_results[2])
                                    fit_upper = D14_equation(x_fine, fit_results[0] + fit_results_upper[0],
                                                             fit_results[1] + fit_results_upper[1],
                                                             fit_results[2] + fit_results_upper[2])
                                    fit_lower = D14_equation(x_fine, fit_results[0] - fit_results_lower[0],
                                                             fit_results[1] - fit_results_lower[1],
                                                             fit_results[2] - fit_results_lower[2])
                                    plt.fill_between(x_fine, fit_lower, fit_upper, color="green", alpha=0.2, zorder=0.5)
                                    plt.plot(x_fine, fit_median, color="green", label="Median Fit", zorder=0.5)

                                    plt.title('Fit with Piecewise Flare Profile')
                                else:
                                    plt.plot(x_fine, fit, color="red", label="Maximum Likelihood")
                                    fit_median = flare_equation(x_fine, fit_results[0], fit_results[1], fit_results[2])
                                    fit_upper = flare_equation(x_fine, fit_results[0] + fit_results_upper[0],
                                                             fit_results[1] + fit_results_upper[1],
                                                             fit_results[2] + fit_results_upper[2])
                                    fit_lower = flare_equation(x_fine, fit_results[0] - fit_results_lower[0],
                                                             fit_results[1] - fit_results_lower[1],
                                                             fit_results[2] - fit_results_lower[2])
                                    plt.fill_between(x_fine, fit_lower, fit_upper, color="green", alpha=0.2)
                                    plt.plot(x_fine, fit_median, color="green", label="Median Fit")


                                    plt.title('Fit with Continuous Flare Profile')
                                plt.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
                                plt.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
                                plt.xlim(start_time, end_time)
                                # plt.ylim(-0.05,1.05)
                                plt.xlabel("Time (d)", fontsize=font_size, style='normal', family='sans-serif')
                                plt.ylabel("Flux", fontsize=font_size, style='normal', family='sans-serif')
                                plt.show()

                            # import pdb; pdb.set_trace()


                            eqdur_fit, eqdur_fit_upper, eqdur_fit_lower = calc_eqdur_fit(start=start_time, end=end_time, results=fit_results, results_upper=fit_results_upper, results_lower=fit_results_lower)

                            fit_results.append(eqdur_fit)
                            fit_results_upper.append(abs(eqdur_fit_upper - eqdur_fit))
                            fit_results_lower.append(abs(eqdur_fit_lower - eqdur_fit))

                            # import pdb; pdb.set_trace()

                            tpeak_frac_fit = get_tpeak_frac(x=x_data, tpeak=fit_results[1])

                            # import pdb; pdb.set_trace()

                            if do_fit_D14 == True:
                                profile = "D14"
                            else:
                                profile = "Convolution"

                            if do_plots == True:
                                save_test_dat = True
                            else:
                                save_test_dat = False

                            save_results(lc_parameters=lc_pars, truth=true_pars, fit=fit_results, fit_upper=fit_results_upper,
                                         fit_lower=fit_results_lower, obs=obs_pars, obs_upper=obs_pars_upper, obs_lower=obs_pars_lower,
                                         rise_cadence_resolution_true=rise_cadence_resolution_true,
                                         decay_cadence_resolution_true=decay_cadence_resolution_true,
                                         total_cadence_resolution_true=total_cadence_resolution_true,
                                         rise_cadence_resolution_obs=rise_cadence_resolution_obs,
                                         decay_cadence_resolution_obs=decay_cadence_resolution_obs,
                                         total_cadence_resolution_obs=total_cadence_resolution_obs,
                                         rise_point_resolution_true=rise_point_resolution_true,
                                         decay_point_resolution_true=decay_point_resolution_true,
                                         total_point_resolution_true=total_point_resolution_true,
                                         rise_point_resolution_obs=rise_point_resolution_obs,
                                         decay_point_resolution_obs=decay_point_resolution_obs,
                                         total_point_resolution_obs=total_point_resolution_obs,
                                         flare_start=flare_start_time, flare_end=flare_stop_time,
                                         obs_start=obs_start_time, obs_end=obs_end_time, fit_tpf=tpeak_frac_fit,
                                         true_tpf=tpeak_frac_truth, fit_profile=profile, Test=save_test_dat)
                    else:

                        fit_results = [float('nan'), float('nan'), float('nan'), float('nan')]
                        fit_results_upper = [float('nan'), float('nan'), float('nan'), float('nan')]
                        fit_results_lower = [float('nan'), float('nan'), float('nan'), float('nan')]
                        obs_pars = [float('nan'), x_data[y_data == max(y_data)][0], float('nan'), float('nan'), float('nan')]
                        obs_pars_upper = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
                        obs_pars_lower = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
                        tpeak_frac_fit = float('nan')

                        if do_plots == True:
                            save_test_dat = True
                        else:
                            save_test_dat = False

                        save_results(lc_parameters=lc_pars, truth=true_pars, fit=fit_results,
                                     fit_upper=fit_results_upper,
                                     fit_lower=fit_results_lower, obs=obs_pars, obs_upper=obs_pars_upper,
                                     obs_lower=obs_pars_lower,
                                     rise_cadence_resolution_true=rise_cadence_resolution_true,
                                     decay_cadence_resolution_true=decay_cadence_resolution_true,
                                     total_cadence_resolution_true=total_cadence_resolution_true,
                                     rise_cadence_resolution_obs=rise_cadence_resolution_obs,
                                     decay_cadence_resolution_obs=decay_cadence_resolution_obs,
                                     total_cadence_resolution_obs=total_cadence_resolution_obs,
                                     rise_point_resolution_true=rise_point_resolution_true,
                                     decay_point_resolution_true=decay_point_resolution_true,
                                     total_point_resolution_true=total_point_resolution_true,
                                     rise_point_resolution_obs=rise_point_resolution_obs,
                                     decay_point_resolution_obs=decay_point_resolution_obs,
                                     total_point_resolution_obs=total_point_resolution_obs,
                                     flare_start=flare_start_time, flare_end=flare_stop_time,
                                     obs_start=obs_start_time, obs_end=obs_end_time, fit_tpf=tpeak_frac_fit,
                                     true_tpf=tpeak_frac_truth, fit_profile="Convolution", Test=save_test_dat)
                        save_results(lc_parameters=lc_pars, truth=true_pars, fit=fit_results,
                                     fit_upper=fit_results_upper,
                                     fit_lower=fit_results_lower, obs=obs_pars, obs_upper=obs_pars_upper,
                                     obs_lower=obs_pars_lower,
                                     rise_cadence_resolution_true=rise_cadence_resolution_true,
                                     decay_cadence_resolution_true=decay_cadence_resolution_true,
                                     total_cadence_resolution_true=total_cadence_resolution_true,
                                     rise_cadence_resolution_obs=rise_cadence_resolution_obs,
                                     decay_cadence_resolution_obs=decay_cadence_resolution_obs,
                                     total_cadence_resolution_obs=total_cadence_resolution_obs,
                                     rise_point_resolution_true=rise_point_resolution_true,
                                     decay_point_resolution_true=decay_point_resolution_true,
                                     total_point_resolution_true=total_point_resolution_true,
                                     rise_point_resolution_obs=rise_point_resolution_obs,
                                     decay_point_resolution_obs=decay_point_resolution_obs,
                                     total_point_resolution_obs=total_point_resolution_obs,
                                     flare_start=flare_start_time, flare_end=flare_stop_time,
                                     obs_start=obs_start_time, obs_end=obs_end_time, fit_tpf=tpeak_frac_fit,
                                     true_tpf=tpeak_frac_truth, fit_profile="D14", Test=save_test_dat)

                        print('------------------')
                        print('No Flare Identified')
                        print('------------------')
                        print(' ')

                else:

                    fit_results = [float('nan'), float('nan'), float('nan'), float('nan')]
                    fit_results_upper = [float('nan'), float('nan'), float('nan'), float('nan')]
                    fit_results_lower = [float('nan'), float('nan'), float('nan'), float('nan')]
                    obs_pars = [float('nan'), x_data[y_data == max(y_data)][0], float('nan'), float('nan'), float('nan')]
                    obs_pars_upper = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
                    obs_pars_lower = [float('nan'), float('nan'), float('nan'), float('nan'), float('nan')]
                    tpeak_frac_fit = float('nan')

                    if do_plots == True:
                        save_test_dat = True
                    else:
                        save_test_dat = False

                    save_results(lc_parameters=lc_pars, truth=true_pars, fit=fit_results,
                                 fit_upper=fit_results_upper,
                                 fit_lower=fit_results_lower, obs=obs_pars, obs_upper=obs_pars_upper,
                                 obs_lower=obs_pars_lower,
                                 rise_cadence_resolution_true=rise_cadence_resolution_true,
                                 decay_cadence_resolution_true=decay_cadence_resolution_true,
                                 total_cadence_resolution_true=total_cadence_resolution_true,
                                 rise_cadence_resolution_obs=rise_cadence_resolution_obs,
                                 decay_cadence_resolution_obs=decay_cadence_resolution_obs,
                                 total_cadence_resolution_obs=total_cadence_resolution_obs,
                                 rise_point_resolution_true=rise_point_resolution_true,
                                 decay_point_resolution_true=decay_point_resolution_true,
                                 total_point_resolution_true=total_point_resolution_true,
                                 rise_point_resolution_obs=rise_point_resolution_obs,
                                 decay_point_resolution_obs=decay_point_resolution_obs,
                                 total_point_resolution_obs=total_point_resolution_obs,
                                 flare_start=flare_start_time, flare_end=flare_stop_time,
                                 obs_start=obs_start_time, obs_end=obs_end_time, fit_tpf=tpeak_frac_fit,
                                 true_tpf=tpeak_frac_truth, fit_profile="Convolution", Test=save_test_dat)
                    save_results(lc_parameters=lc_pars, truth=true_pars, fit=fit_results,
                                 fit_upper=fit_results_upper,
                                 fit_lower=fit_results_lower, obs=obs_pars, obs_upper=obs_pars_upper,
                                 obs_lower=obs_pars_lower,
                                 rise_cadence_resolution_true=rise_cadence_resolution_true,
                                 decay_cadence_resolution_true=decay_cadence_resolution_true,
                                 total_cadence_resolution_true=total_cadence_resolution_true,
                                 rise_cadence_resolution_obs=rise_cadence_resolution_obs,
                                 decay_cadence_resolution_obs=decay_cadence_resolution_obs,
                                 total_cadence_resolution_obs=total_cadence_resolution_obs,
                                 rise_point_resolution_true=rise_point_resolution_true,
                                 decay_point_resolution_true=decay_point_resolution_true,
                                 total_point_resolution_true=total_point_resolution_true,
                                 rise_point_resolution_obs=rise_point_resolution_obs,
                                 decay_point_resolution_obs=decay_point_resolution_obs,
                                 total_point_resolution_obs=total_point_resolution_obs,
                                 flare_start=flare_start_time, flare_end=flare_stop_time,
                                 obs_start=obs_start_time, obs_end=obs_end_time, fit_tpf=tpeak_frac_fit,
                                 true_tpf=tpeak_frac_truth, fit_profile="D14", Test=save_test_dat)

                    print('------------------')
                    print('Too Small Data Train')
                    print('------------------')
                    print(' ')

                if do_plots == True:
                    import pdb; pdb.set_trace()
