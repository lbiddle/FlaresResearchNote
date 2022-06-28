import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import pandas as pd
# from mycolorpy import colorlist as mcp
import astropy.units as u



def plot_flare(x, y, y_scatter, yerr, xlim):

    font_size = 'medium'

    fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
    ax = fig.add_subplot(111)

    ax.plot(x, y, color='#000000', lw=4, alpha=0.2)
    ax.scatter(x, y_scatter, color='#000000', s=np.pi * (3) ** 2, alpha=1)  # , label=label_y[v], zorder=zrdr):
    ax.errorbar(x, y_scatter, yerr=yerr, fmt='None', ecolor='#000000', elinewidth=2, capsize=2, capthick=2)  # , alpha=1, zorder=zrdr)

    ax.set_ylabel('Flux', fontsize=font_size, style='normal', family='sans-serif')
    ax.set_xlabel('Time (d)', fontsize=font_size, style='normal', family='sans-serif')
    ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
    # ax.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
    ax.set_xlim(xlim[0], xlim[1])
    plt.tight_layout()
    # plt.savefig(save_as, dpi=300)
    # plt.close()
    plt.show()

def unique(input_list):

    # initialize a null list
    unique_list = []

    # traverse for all elements
    for x in input_list:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)

    return unique_list

save_location = '/Users/lbiddle/PycharmProjects/FlaresResearchNote/Test/'

fit_data_convolution = pd.read_csv('/Users/lbiddle/PycharmProjects/FlaresResearchNote/Fit_Data_Convolution.csv')
fit_data_convolution = fit_data_convolution[fit_data_convolution['Sampling Rate (min)'] <= 40.]
# print(fit_data_convolution.columns)
fit_data_D14 = pd.read_csv('/Users/lbiddle/PycharmProjects/FlaresResearchNote/Fit_Data_D14.csv')
fit_data_D14 = fit_data_D14[fit_data_D14['Sampling Rate (min)'] <= 40.]
# print(fit_data_D14.columns)

do_which = 'D14'  # 'convolution'  # 'D14'

do_test_sampling_rates = True
if do_test_sampling_rates == True:

    sampling_rates = unique(input_list=fit_data_D14['Sampling Rate (min)'].values)
    SNRs = unique(input_list=fit_data_D14['SNR'].values)
    sampling_rates.sort()
    SNRs.sort()
    # SNRs = np.concatenate(([0],SNRs,[256*2]))
    # bounds = list(np.concatenate(([0], SNRs, [256*2])))
    bounds = list(np.concatenate((SNRs, [256 * 2])))
    Nsegments = len(SNRs)

    # colormap = plt.cm.jet
    colormap = cm.get_cmap('jet', Nsegments+1)
    normalize = matplotlib.colors.BoundaryNorm(bounds, colormap.N)
    # normalize = matplotlib.colors.Normalize(vmin=np.nanmin(SNRs), vmax=np.nanmax(SNRs))

    for sampling_rate in sampling_rates:
        plt.close()

        fit_data_c = fit_data_convolution[fit_data_convolution['Sampling Rate (min)'] == sampling_rate]
        rise_cadence_resolution_max_c = max([max(fit_data_convolution['Rise Cadence Resolution True'].values), max(fit_data_convolution['Rise Cadence Resolution Obs'].values)])
        rise_point_resolution_max_c = max([max(fit_data_convolution['Rise Point Resolution True'].values), max(fit_data_convolution['Rise Point Resolution Obs'].values)])


        fit_data_p = fit_data_D14[fit_data_D14['Sampling Rate (min)'] == sampling_rate]
        rise_cadence_resolution_max_p = max([max(fit_data_D14['Rise Cadence Resolution True'].values), max(fit_data_D14['Rise Cadence Resolution Obs'].values)])
        rise_point_resolution_max_p = max([max(fit_data_D14['Rise Point Resolution True'].values), max(fit_data_D14['Rise Point Resolution Obs'].values)])

        print('Sampling Rate: ' + str(sampling_rate))
        print(len(fit_data_c))


        eqdur_truth_c = np.array(fit_data_c['eqdur truth'].values)
        eqdur_truth_p = np.array(fit_data_p['eqdur truth'].values)

        eqdur_fit_c = np.array(fit_data_c['eqdur fit'].values)
        eqdur_fit_upper_c = np.array(fit_data_c['eqdur fit upper'].values)
        eqdur_fit_lower_c = np.array(fit_data_c['eqdur fit lower'].values)
        # eqdur_fit_upper_c = abs(np.array(fit_data_c['eqdur fit upper'].values) - eqdur_fit_c)
        # eqdur_fit_lower_c = abs(np.array(fit_data_c['eqdur fit lower'].values) - eqdur_fit_c)
        eqdur_fit_p = np.array(fit_data_p['eqdur fit'].values)
        eqdur_fit_upper_p = np.array(fit_data_p['eqdur fit upper'].values)
        eqdur_fit_lower_p = np.array(fit_data_p['eqdur fit lower'].values)
        # eqdur_fit_upper_p = abs(np.array(fit_data_p['eqdur fit upper'].values) - eqdur_fit_p)
        # eqdur_fit_lower_p = abs(np.array(fit_data_p['eqdur fit lower'].values) - eqdur_fit_p)

        eqdur_obs = np.array(fit_data_c['eqdur obs'].values)
        eqdur_obs_upper = np.array(fit_data_c['eqdur obs upper'].values)
        eqdur_obs_lower = np.array(fit_data_c['eqdur obs lower'].values)
        # eqdur_obs_upper = abs(np.array(fit_data_c['eqdur obs upper'].values) - eqdur_obs)
        # eqdur_obs_lower = abs(np.array(fit_data_c['eqdur obs lower'].values) - eqdur_obs)

        if len(np.where(np.isfinite(eqdur_obs) == True)[0]) == 0:
            continue

        eqdur_fit_truth_c = eqdur_fit_c/eqdur_truth_c
        eqdur_fit_truth_upper_c = np.sqrt((1./eqdur_truth_c)**2 * (eqdur_fit_upper_c)**2)
        eqdur_fit_truth_lower_c = np.sqrt((1./eqdur_truth_c)**2 * (eqdur_fit_lower_c)**2)

        eqdur_fit_truth_p = eqdur_fit_p / eqdur_truth_p
        eqdur_fit_truth_upper_p = np.sqrt((1. / eqdur_truth_p) ** 2 * (eqdur_fit_upper_p) ** 2)
        eqdur_fit_truth_lower_p = np.sqrt((1. / eqdur_truth_p) ** 2 * (eqdur_fit_lower_p) ** 2)

        try:
            eqdur_fit_truth_upper_p = np.sqrt((1. / eqdur_truth_p) ** 2 * (eqdur_fit_upper_p) ** 2)
        except:
            import pdb; pdb.set_trace()
        eqdur_fit_truth_lower_p = np.sqrt((1. / eqdur_truth_p) ** 2 * (eqdur_fit_lower_p) ** 2)

        eqdur_obs_truth = eqdur_obs / eqdur_truth_c
        eqdur_obs_truth_upper = np.sqrt((1. / eqdur_truth_c) ** 2 * (eqdur_obs_upper) ** 2)
        eqdur_obs_truth_lower = np.sqrt((1. / eqdur_truth_c) ** 2 * (eqdur_obs_lower) ** 2)

        # fit_data_c['eqdur obs/truth'] = eqdur_obs_truth
        # fit_data_c['eqdur obs/truth upper'] = eqdur_obs_truth_upper
        # fit_data_c['eqdur obs/truth lower'] = eqdur_obs_truth_lower
        # fit_data_p['eqdur obs/truth'] = eqdur_obs_truth
        # fit_data_p['eqdur obs/truth upper'] = eqdur_obs_truth_upper
        # fit_data_p['eqdur obs/truth lower'] = eqdur_obs_truth_lower

        rate_c = np.array(fit_data_c['Sampling Rate (min)'].values)
        rate_c_d = (np.array(fit_data_c['Sampling Rate (min)'].values)*u.min).to(u.d).value
        SNR_c = np.array(fit_data_c['SNR'].values)
        rate_p = np.array(fit_data_p['Sampling Rate (min)'].values)
        rate_p_d = (np.array(fit_data_p['Sampling Rate (min)'].values) * u.min).to(u.d).value
        SNR_p = np.array(fit_data_p['SNR'].values)

        duration_true = np.array(fit_data_c['Flare End Time'].values) - np.array(fit_data_c['Flare Start Time'].values)
        duration_true_min = (duration_true*u.d).to(u.min).value

        fwhm_true = np.array(fit_data_c['fwhm truth'].values)
        fwhm_true_min = (fwhm_true * u.d).to(u.min).value
        fwhm_fit = np.array(fit_data_c['fwhm fit'].values)
        fwhm_obs = np.array(fit_data_c['fwhm obs'].values)

        tpeak_obs = np.array(fit_data_c['tpeak obs'].values)
        tpeak_true = np.array(fit_data_c['tpeak truth'].values)
        tpeak_obs_true_diff = ((tpeak_obs - tpeak_true)*u.d).to(u.min).value / rate_c

        tpeak_fit_c = np.array(fit_data_c['tpeak fit'].values)
        tpeak_fit_p = np.array(fit_data_p['tpeak fit'].values)
        tpeak_fit_diff_c = abs(tpeak_fit_c - tpeak_true)
        tpeak_fit_diff_p = abs(tpeak_fit_p - tpeak_true)
        tpeak_fit_diff_cadence_ratio_c = tpeak_fit_diff_c / rate_c_d
        tpeak_fit_diff_cadence_ratio_p = tpeak_fit_diff_p / rate_p_d

        tpeak_fit_diff_fwhm_c = tpeak_fit_diff_c / fwhm_true
        tpeak_fit_diff_fwhm_p = tpeak_fit_diff_p / fwhm_true
        tpeak_fit_diff_duration_c = tpeak_fit_diff_c / duration_true
        tpeak_fit_diff_duration_p = tpeak_fit_diff_p / duration_true

        rise_cadence_resolution_true = np.array(fit_data_c['Rise Cadence Resolution True'].values)
        decay_cadence_resolution_true = np.array(fit_data_c['Decay Cadence Resolution True'].values)
        total_cadence_resolution_true = np.array(fit_data_c['Total Cadence Resolution True'].values)
        rise_cadence_resolution_obs = np.array(fit_data_c['Rise Cadence Resolution Obs'].values)
        decay_cadence_resolution_obs = np.array(fit_data_c['Decay Cadence Resolution Obs'].values)
        total_cadence_resolution_obs = np.array(fit_data_c['Total Cadence Resolution Obs'].values)
        rise_point_resolution_true = np.array(fit_data_c['Rise Point Resolution True'].values)
        decay_point_resolution_true = np.array(fit_data_c['Decay Point Resolution True'].values)
        total_point_resolution_true = np.array(fit_data_c['Total Point Resolution True'].values)
        rise_point_resolution_obs = np.array(fit_data_c['Rise Point Resolution Obs'].values)
        decay_point_resolution_obs = np.array(fit_data_c['Decay Point Resolution Obs'].values)
        total_point_resolution_obs = np.array(fit_data_c['Total Point Resolution Obs'].values)

        tpeak_frac_truth = np.array(fit_data_c['tpeak frac truth'].values)
        tpeak_frac_fit_c = np.array(fit_data_c['tpeak frac fit'].values)
        tpeak_frac_fit_p = np.array(fit_data_p['tpeak frac fit'].values)

        tpeak_frac_fit_truth_c = np.array(tpeak_frac_fit_c) / np.array(tpeak_frac_truth)
        tpeak_frac_fit_truth_p = np.array(tpeak_frac_fit_p) / np.array(tpeak_frac_truth)

        # if len(tpeak_fit_diff_fwhm_c[tpeak_fit_diff_fwhm_c > 1.0]) >= 1:
        #     print('\nrate_c: ' + str(rate_c[0]))
        #     print('fit_diff / cadence:  ' + str(tpeak_fit_diff_cadence_ratio_c[tpeak_fit_diff_fwhm_c > 1.0]))
        #     print('fit_diff / fwhm:  ' + str(tpeak_fit_diff_fwhm_c[tpeak_fit_diff_fwhm_c > 1.0]))
        #     print('fit_diff / duration:  ' + str(tpeak_fit_diff_duration_c[tpeak_fit_diff_fwhm_c > 1.0]))
        #     print('fwhm_true / rate_c_d: ' + str(fwhm_true[0]/rate_c_d[0]))
        #     import pdb; pdb.set_trace()
        # if len(tpeak_fit_diff_fwhm_p[tpeak_fit_diff_fwhm_p > 1.0]) >= 1:
        #     print('\nrate_p: ' + str(rate_p[0]))
        #     print('fit_diff / cadence:  ' + str(tpeak_fit_diff_cadence_ratio_p[tpeak_fit_diff_fwhm_p > 1.0]))
        #     print('fit_diff / fwhm:  ' + str(tpeak_fit_diff_fwhm_p[tpeak_fit_diff_fwhm_p > 1.0]))
        #     print('fit_diff / duration:  ' + str(tpeak_fit_diff_duration_p[tpeak_fit_diff_fwhm_p > 1.0]))
        #     print('fwhm_true / rate_p_d: ' + str(fwhm_true[0] / rate_p_d[0]))
        #     import pdb; pdb.set_trace()

        max_y = 1.1*max([np.nanmax(eqdur_fit_c+eqdur_fit_upper_c), np.nanmax(eqdur_fit_p+eqdur_fit_upper_p),
                     np.nanmax(eqdur_obs+eqdur_obs_upper)])
        x_1to1 = np.linspace(0, max_y, 10)
        y_1to1 = x_1to1

        font_size = 'large'
        fit_color_c = 'red' # '#990000'  # red
        fit_color_p = 'blue'  # red
        obs_color = 'orange'  # blue

        hist_color_truth = 'grey'
        hist_color_fit_c = 'red'
        hist_color_fit_p = 'blue'
        hist_color_obs = 'orange'
        vert_color = 'cyan'

        # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
        # ax = fig.add_subplot(111)
        # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
        # ax.scatter(eqdur_truth, eqdur_fit, color='black', s=np.pi * (3) ** 2, alpha=1)
        # ax.errorbar(eqdur_truth, eqdur_fit, yerr=[abs(eqdur_fit_lower), abs(eqdur_fit_upper)],
        #             fmt='None', ecolor='black', elinewidth=2, capsize=2, capthick=2)
        #
        # ax.set_title(str(sampling_rate) + ' min', fontsize=font_size, style='normal', family='sans-serif')
        # ax.set_ylabel('ED fit', fontsize=font_size, style='normal', family='sans-serif')
        # ax.set_xlabel('ED truth', fontsize=font_size, style='normal', family='sans-serif')
        # ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        # # ax.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
        # ax.set_ylim(1e1, max_y)
        # # ax.set_xlim(1e1, max_y)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.tight_layout()
        # # plt.savefig(save_as, dpi=300)
        # # plt.close()
        # plt.show()
        # plt.close()
        #
        # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
        # ax = fig.add_subplot(111)
        # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
        # ax.scatter(eqdur_truth, eqdur_obs, color='black', s=np.pi * (3) ** 2, alpha=1)
        # ax.errorbar(eqdur_truth, eqdur_obs, yerr=[abs(eqdur_obs_lower), abs(eqdur_obs_upper)],
        #             fmt='None', ecolor='black', elinewidth=2, capsize=2, capthick=2)
        #
        # ax.set_title(str(sampling_rate) + ' min', fontsize=font_size, style='normal', family='sans-serif')
        # ax.set_ylabel('ED obs', fontsize=font_size, style='normal', family='sans-serif')
        # ax.set_xlabel('ED truth', fontsize=font_size, style='normal', family='sans-serif')
        # ax.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
        # # ax.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
        # ax.set_ylim(1e1, max_y)
        # # ax.set_xlim(1e1, max_y)
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        # plt.tight_layout()
        # # plt.savefig(save_as, dpi=300)
        # # plt.close()
        # plt.show()
        # plt.close()


        do_tpeak_frac_vs_ED_ratio = False
        if do_tpeak_frac_vs_ED_ratio == True:
            where_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == False)[0]
            where_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == False)[0]
            where_legit_obs = np.where(np.isnan(eqdur_obs_truth) == False)[0]

            ED_ratio_found_c = eqdur_fit_truth_c[where_legit_c]
            ED_ratio_found_p = eqdur_fit_truth_p[where_legit_p]
            ED_ratio_found_obs = eqdur_obs_truth[where_legit_obs]

            tpeak_frac_fit_found_c = tpeak_frac_fit_c[where_legit_c]
            tpeak_frac_fit_found_p = tpeak_frac_fit_p[where_legit_p]
            tpeak_frac_truth_found_c = tpeak_frac_truth[where_legit_c]
            tpeak_frac_truth_found_p = tpeak_frac_truth[where_legit_p]

            tpeak_frac_ratio_found_c = tpeak_frac_fit_found_c / tpeak_frac_truth_found_c
            tpeak_frac_ratio_found_p = tpeak_frac_fit_found_p / tpeak_frac_truth_found_p

            where_not_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == True)[0]
            where_not_legit_obs = np.where(np.isnan(eqdur_obs_truth) == True)[0]

            ED_fit_not_found_c = eqdur_truth_c[where_not_legit_c]
            ED_fit_not_found_p = eqdur_truth_p[where_not_legit_p]
            ED_fit_not_found_obs = eqdur_obs_truth[where_not_legit_obs]

            tpeak_frac_fit_not_found_c = tpeak_frac_truth[where_not_legit_c]
            tpeak_frac_fit_not_found_p = tpeak_frac_truth[where_not_legit_p]

            xmin = min(tpeak_frac_truth_found_c)
            xmax = max(tpeak_frac_truth_found_c)

            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(311)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            ax1.scatter(tpeak_frac_truth_found_c, ED_ratio_found_c, color=fit_color_c, s=np.pi * (3) ** 2, alpha=1, label='Continuous')
            # ax1.errorbar(rise_point_resolution_true, ED_ratio_found_c, yerr=[abs(eqdur_fit_truth_lower_c), abs(eqdur_fit_truth_upper_c)],
            #             fmt='None', ecolor=fit_color_c, elinewidth=2, capsize=2, capthick=2)
            ax1.plot([xmin,xmax],[1,1], ls='--', lw=1, color='#000000')

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('ED Fit/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax1.set_xlim(xmin,xmax)


            ax2 = fig.add_subplot(312)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            ax2.scatter(tpeak_frac_truth_found_p, ED_ratio_found_p, color=fit_color_p, s=np.pi * (3) ** 2, alpha=1, label='Piecewise')
            # ax2.errorbar(rise_point_resolution_true, ED_ratio_found_p, yerr=[abs(eqdur_fit_truth_lower_p), abs(eqdur_fit_truth_upper_p)],
            #             fmt='None', ecolor=fit_color_p, elinewidth=2, capsize=2, capthick=2)
            ax2.plot([xmin, xmax], [1, 1], ls='--', lw=1, color='#000000')

            ax2.set_ylabel('ED Fit/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax2.set_xlim(xmin, xmax)
            # ax.set_yscale('log')
            # ax.set_xscale('log')


            ax3 = fig.add_subplot(313)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            ax3.scatter(tpeak_frac_truth_found_p, ED_ratio_found_obs, color=obs_color, s=np.pi * (3) ** 2, alpha=1, label='Observed Points')
            # ax2.errorbar(rise_point_resolution_true, ED_ratio_found_p, yerr=[abs(eqdur_fit_truth_lower_p), abs(eqdur_fit_truth_upper_p)],
            #             fmt='None', ecolor=fit_color_p, elinewidth=2, capsize=2, capthick=2)
            ax3.plot([xmin, xmax], [1, 1], ls='--', lw=1, color='#000000')

            ax3.set_ylabel('ED Obs/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax3.set_xlabel('Time Fraction of True Peak Location Between Points', fontsize=font_size, style='normal', family='sans-serif')
            ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax3.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax3.set_xlim(xmin, xmax)
            # ax.set_yscale('log')
            # ax.set_xscale('log')

            plt.tight_layout()
            plt.savefig(save_location + 'tpeak_Frac_vs_ED_Ratio/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()
            # plt.show()

        do_fit_tpeak_frac_ratio_vs_fit_ED_ratio = False
        if do_fit_tpeak_frac_ratio_vs_fit_ED_ratio == True:
            where_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == False)[0]
            where_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == False)[0]
            where_legit_obs = np.where(np.isnan(eqdur_obs_truth) == False)[0]

            ED_ratio_found_c = eqdur_fit_truth_c[where_legit_c]
            ED_ratio_found_p = eqdur_fit_truth_p[where_legit_p]
            ED_ratio_found_obs = eqdur_obs_truth[where_legit_obs]

            tpeak_frac_fit_found_c = tpeak_frac_fit_c[where_legit_c]
            tpeak_frac_fit_found_p = tpeak_frac_fit_p[where_legit_p]
            tpeak_frac_truth_found_c = tpeak_frac_truth[where_legit_c]
            tpeak_frac_truth_found_p = tpeak_frac_truth[where_legit_p]

            tpeak_frac_ratio_found_c = tpeak_frac_fit_found_c / tpeak_frac_truth_found_c
            tpeak_frac_ratio_found_p = tpeak_frac_fit_found_p / tpeak_frac_truth_found_p

            where_not_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == True)[0]
            where_not_legit_obs = np.where(np.isnan(eqdur_obs_truth) == True)[0]

            ED_fit_not_found_c = eqdur_truth_c[where_not_legit_c]
            ED_fit_not_found_p = eqdur_truth_p[where_not_legit_p]
            ED_fit_not_found_obs = eqdur_obs_truth[where_not_legit_obs]

            tpeak_frac_fit_not_found_c = tpeak_frac_truth[where_not_legit_c]
            tpeak_frac_fit_not_found_p = tpeak_frac_truth[where_not_legit_p]


            xmin = min([min(tpeak_frac_ratio_found_c), min(tpeak_frac_ratio_found_p)])
            xmax = max([max(tpeak_frac_ratio_found_c), max(tpeak_frac_ratio_found_p)])

            xmin = 0
            xmax = 3


            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(211)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            ax1.scatter(tpeak_frac_ratio_found_c, ED_ratio_found_c, color=fit_color_c, s=np.pi * (3) ** 2, alpha=1, label='Continuous')
            # ax1.errorbar(rise_point_resolution_true, ED_ratio_found_c, yerr=[abs(eqdur_fit_truth_lower_c), abs(eqdur_fit_truth_upper_c)],
            #             fmt='None', ecolor=fit_color_c, elinewidth=2, capsize=2, capthick=2)
            ax1.plot([xmin, xmax], [1, 1], ls='--', lw=1, color='#000000')

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('ED Fit/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax1.set_xlim(xmin,xmax)


            ax2 = fig.add_subplot(212)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            ax2.scatter(tpeak_frac_ratio_found_p, ED_ratio_found_p, color=fit_color_p, s=np.pi * (3) ** 2, alpha=1, label='Piecewise')
            # ax2.errorbar(rise_point_resolution_true, ED_ratio_found_p, yerr=[abs(eqdur_fit_truth_lower_p), abs(eqdur_fit_truth_upper_p)],
            #             fmt='None', ecolor=fit_color_p, elinewidth=2, capsize=2, capthick=2)
            ax2.plot([xmin, xmax], [1, 1], ls='--', lw=1, color='#000000')

            ax2.set_ylabel('ED Fit/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax2.set_xlabel('Ratio of Fit/True Time Fraction\nof Peak Location Between Points', fontsize=font_size, style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax2.set_xlim(xmin, xmax)
            plt.tight_layout()
            plt.savefig(save_location + 'Fit_tpeak_Frac_Ratio_vs_Fit_ED_Ratio/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()
            # plt.show()

        do_tpeak_fit_diff_cadence_ratio_vs_ED_ratio = True
        if do_tpeak_fit_diff_cadence_ratio_vs_ED_ratio == True:
            where_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == False)[0]
            where_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == False)[0]
            where_legit_obs = np.where(np.isnan(eqdur_obs_truth) == False)[0]

            ED_ratio_found_c = eqdur_fit_truth_c[where_legit_c]
            ED_ratio_found_p = eqdur_fit_truth_p[where_legit_p]
            ED_ratio_found_obs = eqdur_obs_truth[where_legit_obs]

            tpeak_frac_fit_found_c = tpeak_frac_fit_c[where_legit_c]
            tpeak_frac_fit_found_p = tpeak_frac_fit_p[where_legit_p]
            tpeak_frac_truth_found_c = tpeak_frac_truth[where_legit_c]
            tpeak_frac_truth_found_p = tpeak_frac_truth[where_legit_p]

            tpeak_frac_ratio_found_c = tpeak_frac_fit_found_c / tpeak_frac_truth_found_c
            tpeak_frac_ratio_found_p = tpeak_frac_fit_found_p / tpeak_frac_truth_found_p

            where_not_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == True)[0]
            where_not_legit_obs = np.where(np.isnan(eqdur_obs_truth) == True)[0]

            xmin = np.nanmin([np.nanmin(tpeak_fit_diff_cadence_ratio_c), np.nanmin(tpeak_fit_diff_cadence_ratio_p)])
            xmax = 1.025*np.nanmax([np.nanmax(tpeak_fit_diff_cadence_ratio_c), np.nanmax(tpeak_fit_diff_cadence_ratio_p)])

            # xmin = 0
            # xmax = 2

            fig = plt.figure(figsize=(6, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(211)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            scatterplot1 = ax1.scatter(tpeak_fit_diff_cadence_ratio_c, eqdur_fit_truth_c, c=SNR_c, s=np.pi*(3)**2,
                                       alpha=1, cmap=colormap, norm=normalize, vmin=np.nanmin(SNR_c),
                                       vmax=np.nanmax(SNR_c),label='Continuous')
            # ax1.errorbar(tpeak_fit_diff_cadence_ratio_c, eqdur_fit_truth_c,
            #              yerr=[abs(eqdur_fit_truth_lower_c), abs(eqdur_fit_truth_upper_c)],
            #              fmt='None', ec=SNR_c, elinewidth=2, capsize=2, capthick=2, map=colormap, norm=normalize,
            #              vmin=np.nanmin(SNR_c), vmax=np.nanmax(SNR_c))
            ax1.plot([xmin, xmax], [1, 1], ls='--', lw=1, color='#000000')
            ax1.text(0.16, 0.915, 'Continuous', horizontalalignment='center', verticalalignment='center',
                     transform=ax1.transAxes, fontsize='large', style='normal', family='sans-serif',
                     bbox=dict(facecolor='none', edgecolor='#000000', pad=3.0))

            cb1 = plt.colorbar(scatterplot1, label='SNR', cmap=colormap, norm=normalize, ticks=bounds[0:-1], boundaries=bounds) # spacing='proportional' , cm.ScalarMappable(norm=normalize, cmap=colormap), shrink=0.80)  # , aspect=1) #,cmap=mycolormap, vmin=np.min(color_intervals.values),vmax=np.max(color_intervals.values))  # , ax=ax) #, cmap=cmap)
            cb1.set_label('SNR', labelpad=4, fontsize=font_size, style='normal', family='sans-serif')
            # plt.colorbar().ax.set_ylabel('Mean Absolute Error', rotation=270, fontsize=15, labelpad=15)

            # cb1 = plt.colorbar.ColorbarBase(ax1, cmap=colormap, norm=normalize, spacing='proportional', ticks=bounds, boundaries=bounds)

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(
                title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate),
                fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('ED Fit/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            # ax1.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax1.set_xlim(xmin, xmax)
            ax1.set_xscale('log')
            # ax1.set_xticks([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1.0, 2.0])
            # ax1.set_xticklabels([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.5, 1.0, 2.0])

            ax2 = fig.add_subplot(212)
            # ax.plot(x_1to1, y_1to1, '--', c='black', alpha=0.3)
            scatterplot2 = ax2.scatter(tpeak_fit_diff_cadence_ratio_p, eqdur_fit_truth_p, c=SNR_p,
                                       s=np.pi * (3) ** 2, alpha=1, cmap=colormap, norm=normalize,
                                       vmin=np.nanmin(SNR_p), vmax=np.nanmax(SNR_p), label='Piecewise')
            # ax2.errorbar(tpeak_fit_diff_cadence_ratio_p, eqdur_fit_truth_p,
            #              yerr=[abs(eqdur_fit_truth_lower_p), abs(eqdur_fit_truth_upper_p)],
            #              fmt='None', ec=SNR_p, elinewidth=2, capsize=2, capthick=2, map=colormap, norm=normalize,
            #              vmin=np.nanmin(SNR_p), vmax=np.nanmax(SNR_p))
            ax2.plot([xmin, xmax], [1, 1], ls='--', lw=1, color='#000000')
            ax2.text(0.16, 0.915, 'Piecewise', horizontalalignment='center', verticalalignment='center',
                     transform=ax2.transAxes, fontsize='large', style='normal', family='sans-serif',
                     bbox=dict(facecolor='none', edgecolor='#000000', pad=3.0))

            cb2 = plt.colorbar(scatterplot2, ax=ax2, label='SNR', cmap=colormap, norm=normalize, ticks=bounds[0:-1], boundaries=bounds)  # spacing='proportional' , cm.ScalarMappable(norm=normalize, cmap=colormap), shrink=0.80)  # , aspect=1) #,cmap=mycolormap, vmin=np.min(color_intervals.values),vmax=np.max(color_intervals.values))  # , ax=ax) #, cmap=cmap)
            cb2.set_label('SNR', labelpad=4, fontsize=font_size, style='normal', family='sans-serif')

            ax2.set_ylabel('ED Fit/Truth', fontsize=font_size, style='normal', family='sans-serif')
            ax2.set_xlabel('(Fit - True tpeak) / Cadence', fontsize=font_size,
                           style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            # ax2.legend(loc='upper right', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            # ax.set_ylim(0, 3)
            ax2.set_xlim(xmin, xmax)
            ax2.set_xscale('log')

            plt.tight_layout()
            plt.savefig(save_location + 'Fit_tpeak-True_tpeak_Cadence_Ratio_vs_ED_Ratio/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()
            # plt.show()

            # import pdb; pdb.set_trace()

        # --------
        do_true_tpeak_frac_distribution_found = False
        if do_true_tpeak_frac_distribution_found == True:


            where_legit_c = np.where(np.isnan(tpeak_frac_fit_c) == False)[0]
            where_legit_p = np.where(np.isnan(tpeak_frac_fit_p) == False)[0]

            where_not_legit_c = np.where(np.isnan(tpeak_frac_fit_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(tpeak_frac_fit_p) == True)[0]

            tpeak_frac_fit_found_c = tpeak_frac_fit_c[where_legit_c]
            tpeak_frac_fit_found_p = tpeak_frac_fit_p[where_legit_p]

            tpeak_frac_fit_not_found_c = tpeak_frac_truth[where_not_legit_c]
            tpeak_frac_fit_not_found_p = tpeak_frac_truth[where_not_legit_p]


            bin_width = 0.1
            edges = np.arange(0,1+bin_width,bin_width)

            hist_truth, hist_edges = np.histogram(tpeak_frac_truth, bins=edges)
            hist_fit_c, hist_edges = np.histogram(tpeak_frac_fit_c, bins=edges)
            hist_fit_p, hist_edges = np.histogram(tpeak_frac_fit_p, bins=edges)
            #frac_hist_obs = hist_obs / sum(hist_obs)
            #self.sum_hist_obs.append(sum(hist_obs))
            # edge_width = np.diff(histedges)[0]

            hist_truth = hist_truth / sum(hist_truth)
            hist_fit_c = hist_fit_c / sum(hist_fit_c)
            hist_fit_p = hist_fit_p / sum(hist_fit_p)

            # max_y_hist = 1.2*max([max(hist_truth),max(hist_fit_c),max(hist_fit_p)])
            max_y_hist = 1.05

            plt.close()
            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(211)

            ax1.bar(edges[0:-1], hist_truth, bin_width, align='edge', color=hist_color_truth, edgecolor='#000000',
                   lw=2, alpha=0.4, label='Truth')
            # ax1.bar(edges[0:-1], hist_truth, bin_width, align='edge', color='None', edgecolor='#000000',
            #        lw=2, alpha=1)

            ax1.bar(edges[0:-1], hist_fit_c, bin_width, align='edge', color=hist_color_fit_c, edgecolor='#000000',
                   lw=2, alpha=0.4,label='Continuous')
            # ax1.bar(edges[0:-1], hist_fit_c, bin_width, align='edge', color='None', edgecolor='#000000',
            #        lw=2, alpha=1)

            title_str = 'Sampling Rate: {2:.2f} min\nRise Resolution: {0:.2f}  Decay Resolution: {1:0.2f}'
            ax1.set_title(title_str.format(rise_cadence_resolution_true[0],decay_cadence_resolution_true[0],sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax1.set_xlabel('Fractional Location of Flare Peak Between Measurements', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax1.set_ylim(0, max_y_hist)
            ax1.set_xlim(0, 1)
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            # plt.tight_layout()
            # plt.savefig(save_location + 'tpeak_Frac_Found/' + 'hist_c-' + str(sampling_rate) + '_' + str(SNR) + '.pdf', dpi=300)
            # plt.close()
            # plt.show()

            #--------------------------
            # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
            ax2 = fig.add_subplot(212)

            ax2.bar(edges[0:-1], hist_truth, bin_width, align='edge', color=hist_color_truth, edgecolor='#000000',
                   lw=2, alpha=0.4, label='Truth')
            # ax2.bar(edges[0:-1], hist_truth, bin_width, align='edge', color='None', edgecolor='#000000',
            #        lw=2, alpha=1)

            ax2.bar(edges[0:-1], hist_fit_p, bin_width, align='edge', color=hist_color_fit_p, edgecolor='#000000',
                   lw=2, alpha=0.4, label='Piecewise')
            # ax2.bar(edges[0:-1], hist_fit_p, bin_width, align='edge', color='None', edgecolor='#000000',
            #        lw=2, alpha=1)

            # title_str = 'Rise Resolution: {0:.2f}  Decay Resolution: {1:0.2f}\nSampling Rate: {2:.2f} min}'
            # ax.set_title(title_str.format(rise_cadence_resolution_true[0],decay_cadence_resolution_true[0],sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax2.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax2.set_xlabel('Fractional Location of Flare Peak Between Measurements', fontsize=font_size,
                          style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax2.set_ylim(0, max_y_hist)
            ax2.set_xlim(0, 1)
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            plt.tight_layout()
            plt.savefig(save_location + 'True_tpeak_Frac_Distribution_Found/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()
            # plt.show()

        do_true_tpeak_frac_distribution_not_found = False
        if do_true_tpeak_frac_distribution_not_found == True:


            where_legit_c = np.where(np.isnan(tpeak_frac_fit_c) == False)[0]
            where_legit_p = np.where(np.isnan(tpeak_frac_fit_p) == False)[0]

            where_not_legit_c = np.where(np.isnan(tpeak_frac_fit_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(tpeak_frac_fit_p) == True)[0]

            tpeak_frac_fit_found_c = tpeak_frac_fit_c[where_legit_c]
            tpeak_frac_fit_found_p = tpeak_frac_fit_p[where_legit_p]

            tpeak_frac_fit_not_found_c = tpeak_frac_truth[where_not_legit_c]
            tpeak_frac_fit_not_found_p = tpeak_frac_truth[where_not_legit_p]



            bin_width = 0.1
            edges = np.arange(0,1+bin_width,bin_width)

            hist_truth, hist_edges = np.histogram(tpeak_frac_truth, bins=edges)
            hist_fit_found_c, hist_edges = np.histogram(tpeak_frac_fit_found_c, bins=edges)
            hist_fit_found_p, hist_edges = np.histogram(tpeak_frac_fit_found_p, bins=edges)

            hist_not_found_c, hist_edges = np.histogram(tpeak_frac_fit_not_found_c, bins=edges)
            hist_not_found_p, hist_edges = np.histogram(tpeak_frac_fit_not_found_p, bins=edges)

            bar_frac_not_found_c = hist_not_found_c / hist_truth
            bar_frac_not_found_p = hist_not_found_p / hist_truth

            bar_frac_not_found_c[np.isnan(bar_frac_not_found_c) == True] = 0
            bar_frac_not_found_p[np.isnan(bar_frac_not_found_p) == True] = 0

            # max_y_hist = 1.2*max([max(hist_truth),max(hist_fit_found_c),max(hist_fit_found_p)])
            max_y_hist = 1.05

            plt.close()
            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(211)

            ax1.bar(edges[0:-1], bar_frac_not_found_c, bin_width, align='edge', color=hist_color_fit_c,
                   edgecolor='#000000', lw=2, alpha=1, label='Continuous')
            ax1.bar(edges[0:-1], bar_frac_not_found_c, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('Fraction of Test Flares Unable to Fit', fontsize=font_size, style='normal', family='sans-serif')
            ax1.set_xlabel('Fractional Location of True Flare Peak Between Measurements', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax1.set_ylim(0, max_y_hist)
            ax1.set_xlim(0, 1)
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            # plt.tight_layout()
            # plt.savefig(save_location + 'tpeak_Frac_Not_Found/' + 'bar_c-' + str(sampling_rate) + '.pdf', dpi=300)
            # plt.close()
            # plt.show()

            # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
            ax2 = fig.add_subplot(212)

            ax2.bar(edges[0:-1], bar_frac_not_found_p, bin_width, align='edge', color=hist_color_fit_p,
                   edgecolor='#000000', lw=2, alpha=1, label='Piecewise')
            ax2.bar(edges[0:-1], bar_frac_not_found_p, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)

            # title_str = 'Rise Resolution: {0:.2f}  Decay Resolution: {1:0.2f}\nSampling Rate: {2:.2f} min'
            # ax.set_title(title_str.format(rise_cadence_resolution_true[0],decay_cadence_resolution_true[0],sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax2.set_ylabel('Fraction of Test Flares Unable to Fit', fontsize=font_size, style='normal',
                          family='sans-serif')
            ax2.set_xlabel('Fractional Location of True Flare Peak Between Measurements', fontsize=font_size,
                          style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax2.set_ylim(0, max_y_hist)
            ax2.set_xlim(0, 1)
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            plt.tight_layout()
            plt.savefig(save_location + 'True_tpeak_Frac_Distribution_Not_Found/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()

        do_tpeak_frac_ratio_distribution = False
        if do_tpeak_frac_ratio_distribution == True:


            where_legit_c = np.where(np.isnan(tpeak_frac_fit_c) == False)[0]
            where_legit_p = np.where(np.isnan(tpeak_frac_fit_p) == False)[0]

            where_not_legit_c = np.where(np.isnan(tpeak_frac_fit_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(tpeak_frac_fit_p) == True)[0]

            tpeak_frac_fit_found_c = tpeak_frac_fit_c[where_legit_c]
            tpeak_frac_fit_found_p = tpeak_frac_fit_p[where_legit_p]
            tpeak_frac_truth_found_c = tpeak_frac_truth[where_legit_c]
            tpeak_frac_truth_found_p = tpeak_frac_truth[where_legit_p]

            tpeak_frac_fit_not_found_c = tpeak_frac_truth[where_not_legit_c]
            tpeak_frac_fit_not_found_p = tpeak_frac_truth[where_not_legit_p]

            tpeak_frac_ratio_found_c = tpeak_frac_fit_found_c / tpeak_frac_truth_found_c
            tpeak_frac_ratio_found_p = tpeak_frac_fit_found_p / tpeak_frac_truth_found_p


            max_range = max([max(tpeak_frac_ratio_found_c),max(tpeak_frac_ratio_found_p)])
            bin_width = 0.1
            edges = np.arange(0,max_range+bin_width,bin_width)

            hist_truth, hist_edges = np.histogram(tpeak_frac_truth, bins=edges)
            hist_fit_found_c, hist_edges = np.histogram(tpeak_frac_fit_found_c, bins=edges)
            hist_fit_found_p, hist_edges = np.histogram(tpeak_frac_fit_found_p, bins=edges)

            hist_frac_ratio_found_c, hist_edges = np.histogram(tpeak_frac_ratio_found_c, bins=edges)
            hist_frac_ratio_found_p, hist_edges = np.histogram(tpeak_frac_ratio_found_p, bins=edges)

            # hist_not_found_c, hist_edges = np.histogram(tpeak_frac_fit_not_found_c, bins=edges)
            # hist_not_found_p, hist_edges = np.histogram(tpeak_frac_fit_not_found_p, bins=edges)
            #
            # bar_frac_not_found_c = hist_not_found_c / hist_truth
            # bar_frac_not_found_p = hist_not_found_p / hist_truth
            #
            # bar_frac_not_found_c[np.isnan(bar_frac_not_found_c) == True] = 0
            # bar_frac_not_found_p[np.isnan(bar_frac_not_found_p) == True] = 0

            max_y_hist = 1.2*max([max(hist_frac_ratio_found_c),max(hist_frac_ratio_found_p)])
            # max_y_hist = 1.05

            plt.close()
            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(211)

            ax1.bar(edges[0:-1], hist_frac_ratio_found_c, bin_width, align='edge', color=hist_color_fit_c,
                   edgecolor='#000000', lw=2, alpha=1, label='Continuous')
            ax1.bar(edges[0:-1], hist_frac_ratio_found_c, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax1.set_xlabel('Fractional Location of Fit/True Flare Peak Between Measurements', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax1.set_ylim(0, max_y_hist)
            ax1.set_xlim(0, max(edges))
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            # plt.tight_layout()
            # plt.savefig(save_location + 'tpeak_Frac_Ratio/' + 'hist_ratio_c-' + str(sampling_rate) + '.pdf', dpi=300)
            # plt.close()
            # plt.show()

            # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
            ax2 = fig.add_subplot(212)

            ax2.bar(edges[0:-1], hist_frac_ratio_found_p, bin_width, align='edge', color=hist_color_fit_p,
                   edgecolor='#000000', lw=2, alpha=1, label='Piecewise')
            ax2.bar(edges[0:-1], hist_frac_ratio_found_p, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)

            ax2.set_ylabel('Counts', fontsize=font_size, style='normal',
                          family='sans-serif')
            ax2.set_xlabel('Fractional Location of Fit/True Flare Peak Between Measurements', fontsize=font_size,
                          style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax2.set_ylim(0, max_y_hist)
            ax2.set_xlim(0, max(edges))
            # ax.set_yscale('log')
            # ax.set_xscale('log')
            plt.tight_layout()
            plt.savefig(save_location + 'tpeak_Frac_Ratio_Distribution/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()

        do_ED_ratio_distribution = True
        if do_ED_ratio_distribution == True:
            where_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == False)[0]
            where_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == False)[0]
            where_legit_obs = np.where(np.isnan(eqdur_obs_truth) == False)[0]

            where_not_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == True)[0]
            where_not_legit_obs = np.where(np.isnan(eqdur_obs_truth) == True)[0]

            ED_ratio_found_c = eqdur_fit_truth_c[where_legit_c]
            ED_ratio_found_p = eqdur_fit_truth_p[where_legit_p]
            ED_ratio_found_obs = eqdur_obs_truth[where_legit_obs]

            ED_fit_not_found_c = eqdur_truth_c[where_not_legit_c]
            ED_fit_not_found_p = eqdur_truth_p[where_not_legit_p]
            ED_fit_not_found_obs = eqdur_obs_truth[where_not_legit_obs]

            max_range = max([max(ED_ratio_found_c), max(ED_ratio_found_p), max(ED_ratio_found_obs)])
            max_range = 2.75
            bin_width = 0.1
            edges = np.arange(0, max_range + bin_width, bin_width)

            hist_truth, hist_edges = np.histogram(tpeak_frac_truth, bins=edges)
            hist_ED_ratio_found_c, hist_edges = np.histogram(ED_ratio_found_c, bins=edges)
            hist_ED_ratio_found_p, hist_edges = np.histogram(ED_ratio_found_p, bins=edges)
            hist_ED_ratio_found_obs, hist_edges = np.histogram(ED_ratio_found_obs, bins=edges)

            max_y_hist = 1.2 * max([max(hist_ED_ratio_found_c), max(hist_ED_ratio_found_p), max(hist_ED_ratio_found_obs)])
            # max_y_hist = 1.05

            plt.close()
            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(311)

            ax1.bar(edges[0:-1], hist_ED_ratio_found_c, bin_width, align='edge', color=hist_color_fit_c,
                   edgecolor='#000000', lw=2, alpha=1, label='Continuous')
            ax1.bar(edges[0:-1], hist_ED_ratio_found_c, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)
            ax1.plot([1, 1], [0, max_y_hist], linestyle='--', color=vert_color, lw=3)

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate), fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax1.set_xlabel('Fit/True ED', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax1.set_ylim(0, max_y_hist)
            ax1.set_xlim(0, max(edges))
            # plt.tight_layout()
            # plt.savefig(save_location + 'ED_Ratio/' + 'hist_ratio_c-' + str(sampling_rate) + '_' + str(SNR) + '.pdf', dpi=300)
            # plt.close()

            # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
            ax2 = fig.add_subplot(312)

            ax2.bar(edges[0:-1], hist_ED_ratio_found_p, bin_width, align='edge', color=hist_color_fit_p,
                   edgecolor='#000000', lw=2, alpha=1, label='Piecewise')
            ax2.bar(edges[0:-1], hist_ED_ratio_found_p, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)
            ax2.plot([1,1],[0,max_y_hist], linestyle='--', color=vert_color, lw=3)

            # title_str = 'Rise Resolution: {0:.2f}  Decay Resolution: {1:0.2f}\nSampling Rate: {2:.2f} min   SNR: {3:.1f}'
            # ax.set_title(title_str.format(rise_point_resolution_true[0], decay_point_resolution_true[0], sampling_rate, SNR),
            #              fontsize='medium', style='normal', family='sans-serif')
            ax2.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax2.set_xlabel('Fit/True ED', fontsize=font_size, style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax2.set_ylim(0, max_y_hist)
            ax2.set_xlim(0, max(edges))
            # plt.tight_layout()
            # plt.savefig(save_location + 'ED_Ratio/' + 'hist_ratio_p-' + str(sampling_rate) + '_' + str(
            #     SNR) + '.pdf', dpi=300)
            # plt.close()

            # plt.close()
            # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
            ax3 = fig.add_subplot(313)

            ax3.bar(edges[0:-1], hist_ED_ratio_found_p, bin_width, align='edge', color=hist_color_obs,
                   edgecolor='#000000', lw=2, alpha=1, label='Observed Points')
            ax3.bar(edges[0:-1], hist_ED_ratio_found_p, bin_width, align='edge', color='None', edgecolor='#000000',
                   lw=2, alpha=1)
            ax3.plot([1, 1], [0, max_y_hist], linestyle='--', color=vert_color, lw=3)

            # title_str = 'Rise Resolution: {0:.2f}  Decay Resolution: {1:0.2f}\nSampling Rate: {2:.2f} min   SNR: {3:.1f}'
            # ax.set_title( title_str.format(rise_point_resolution_true[0], decay_point_resolution_true[0], sampling_rate, SNR),fontsize='medium', style='normal', family='sans-serif')
            ax3.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax3.set_xlabel('Obs/True ED', fontsize=font_size, style='normal', family='sans-serif')
            ax3.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax3.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax3.set_ylim(0, max_y_hist)
            ax3.set_xlim(0, max(edges))
            plt.tight_layout()
            plt.savefig(save_location + 'ED_Ratio_Distribution/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()

        do_tpeak_fit_diff_cadence_ratio_distribution = True
        if do_tpeak_fit_diff_cadence_ratio_distribution == True:
            where_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == False)[0]
            where_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == False)[0]

            where_not_legit_c = np.where(np.isnan(eqdur_fit_truth_c) == True)[0]
            where_not_legit_p = np.where(np.isnan(eqdur_fit_truth_p) == True)[0]

            tpeak_fit_diff_cadence_ratio_found_c = tpeak_fit_diff_cadence_ratio_c[where_legit_c]
            tpeak_fit_diff_cadence_ratio_found_p = tpeak_fit_diff_cadence_ratio_p[where_legit_p]

            tpeak_fit_diff_cadence_ratio_not_found_c = tpeak_fit_diff_cadence_ratio_c[where_not_legit_c]
            tpeak_fit_diff_cadence_ratio_not_found_p = tpeak_fit_diff_cadence_ratio_p[where_not_legit_p]

            # max_range = max([max(tpeak_fit_diff_cadence_ratio_found_c), max(tpeak_fit_diff_cadence_ratio_found_p)])
            max_range = 2
            bin_width = 0.05
            edges = np.arange(0, max_range + bin_width, bin_width)

            hist_truth, hist_edges = np.histogram(tpeak_frac_truth, bins=edges)
            hist_tpeak_fit_diff_cadence_ratio_found_c, hist_edges = np.histogram(tpeak_fit_diff_cadence_ratio_found_c, bins=edges)
            hist_tpeak_fit_diff_cadence_ratio_found_p, hist_edges = np.histogram(tpeak_fit_diff_cadence_ratio_found_p, bins=edges)

            max_y_hist = 1.2 * np.nanmax([np.nanmax(hist_tpeak_fit_diff_cadence_ratio_found_c), np.nanmax(hist_tpeak_fit_diff_cadence_ratio_found_p)])
            # max_y_hist = 1.05

            max_x_hist = 1.2*np.nanmax([np.nanmax(tpeak_fit_diff_cadence_ratio_found_c), np.nanmax(tpeak_fit_diff_cadence_ratio_found_p)])

            plt.close()
            fig = plt.figure(figsize=(5, 8), facecolor='#ffffff')  # , dpi=300)
            ax1 = fig.add_subplot(211)

            ax1.bar(edges[0:-1], hist_tpeak_fit_diff_cadence_ratio_found_c, bin_width, align='edge', color=hist_color_fit_c,
                    edgecolor='#000000', lw=2, alpha=1, label='Continuous')
            ax1.bar(edges[0:-1], hist_tpeak_fit_diff_cadence_ratio_found_c, bin_width, align='edge', color='None', edgecolor='#000000',
                    lw=2, alpha=1)
            ax1.plot([1, 1], [0, max_y_hist], linestyle='--', color=vert_color, lw=3)

            title_str = 'Sampling Rate: {2:.2f} min\nCadence Resolution-  Rise: {0:.2f}  Decay: {1:0.2f}'
            ax1.set_title(
                title_str.format(rise_cadence_resolution_true[0], decay_cadence_resolution_true[0], sampling_rate),
                fontsize='medium', style='normal', family='sans-serif')
            ax1.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax1.set_xlabel('Fit/True ED', fontsize=font_size, style='normal', family='sans-serif')
            ax1.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax1.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax1.set_ylim(0, max_y_hist)
            ax1.set_xlim(0, max_x_hist)
            # plt.tight_layout()
            # plt.savefig(save_location + 'ED_Ratio/' + 'hist_ratio_c-' + str(sampling_rate) + '_' + str(SNR) + '.pdf', dpi=300)
            # plt.close()

            # fig = plt.figure(figsize=(6, 5), facecolor='#ffffff')  # , dpi=300)
            ax2 = fig.add_subplot(212)

            ax2.bar(edges[0:-1], hist_tpeak_fit_diff_cadence_ratio_found_p, bin_width, align='edge', color=hist_color_fit_p,
                    edgecolor='#000000', lw=2, alpha=1, label='Piecewise')
            ax2.bar(edges[0:-1], hist_tpeak_fit_diff_cadence_ratio_found_p, bin_width, align='edge', color='None', edgecolor='#000000',
                    lw=2, alpha=1)
            ax2.plot([1, 1], [0, max_y_hist], linestyle='--', color=vert_color, lw=3)

            ax2.set_ylabel('Counts', fontsize=font_size, style='normal', family='sans-serif')
            ax2.set_xlabel('(Fit - True tpeak) / Cadence', fontsize=font_size, style='normal', family='sans-serif')
            ax2.tick_params(axis='both', direction='in', labelsize=font_size, top=True, right=False)
            ax2.legend(loc='upper left', fontsize=font_size, framealpha=1.0, fancybox=False, frameon=True)
            ax2.set_ylim(0, max_y_hist)
            ax2.set_xlim(0, max_x_hist)
            plt.tight_layout()
            plt.savefig(save_location + 'Fit_tpeak-True_tpeak_Cadence_Ratio_Distribution/' + str(sampling_rate) + '.pdf', dpi=300)
            plt.close()


        # import pdb; pdb.set_trace()

    # import pdb; pdb.set_trace()







