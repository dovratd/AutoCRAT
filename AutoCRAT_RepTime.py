
""" Replication time module of AutoCRAT """


import itertools
import logging
from pathlib import Path
import warnings
import multiprocessing

import numpy as np
import pandas as pd
from scipy import optimize, stats, signal

from AutoCRAT_cfg import params, process_num


""" Elucidate replication time by sigmoid fitting """


def logistic(x, base, height, steepness, midpoint):
    """
    Definition of the logistic function, the sigmoidal function used to fit the data
    """

    return base + height / (1 + np.exp(-steepness * (x - midpoint)))


def fit_per_data_window(data_window):
    """
    Performing the sigmoid fitting on a single time window within the data
    """

    warnings.filterwarnings('ignore', message='overflow encountered in exp')

    # The initial guess is important to get a good fit result. The initial guesses are:
    # For base, the bottom (minimal intensity) of the examined data window.
    # For height, the total height of the data window.
    # For steepness, 1, which is an intermediate level of steepness.
    # For midpoint, the half-point of the time window.
    initial_guess = [data_window.min(),
                     data_window.max() - data_window.min(),
                     1,
                     np.median(data_window.index)]
    # Bounds limit the fitting to a reasonable result, and also speed up the fitting process.
    # The sigmoid base is bound to a minimum of a bit under the bottom (minimal intensity)
    # of the data window, and a maximum at the top.
    # The sigmoid height is bound to a minimum of 20% of the data height, to avoid low
    # or declining sigmoids, and a maximum of 110% of the data height.
    # Steepness is bound to be positive.
    # Sigmoid midpoint is bound to be inside the examined time window.
    bounds = ([data_window.min() * 0.9,
               (data_window.max() - data_window.min()) * 0.2,
               0,
               data_window.index[0]],
              [data_window.max(),
               (data_window.max() - data_window.min()) * 1.1,
               np.inf,
               data_window.index[-1]])

    try:
        popt, pcov = optimize.curve_fit(logistic, data_window.index, data_window, p0=initial_guess,
                                        bounds=bounds, ftol=0.01, xtol=0.01)
        return [data_window.index[0], data_window.index[-1], *popt, pcov[3, 3]]

    # If no fit is found, ignore this time window and continue.
    except RuntimeError:
        return None


def sigmoid_fit(celldf, particle_names, cell_num):
    """
    Fit sigmoidal function to dot intensity data using SciPy
    """

    midpoints = {}
    dev_data = {}
    particles_to_fit = [p for p in particle_names if p.split('_')[0] in params['channels_to_fit']]
    for particle_name in particles_to_fit:

        data_to_fit = celldf[particle_name, 'Intensity']
        # Discard NaNs at the beginning and end of the data.
        data_to_fit = data_to_fit.loc[data_to_fit.first_valid_index():data_to_fit.last_valid_index()]
        # Linear interpolation to replace NaNs within the data (unless large gaps exist).
        data_to_fit = data_to_fit.interpolate(limit=params['link_memory'][particle_name.split('_')[0]]*2)

        # Run curve fitting on rolling windows within the data, using a range of window sizes.
        # To save some time, not all possible window sizes are used, but only 1 in 3.
        data_windows = [list(data_to_fit.rolling(windowSize))[(windowSize - 1):]
                        for windowSize in range(params['window_sizes'][0], params['window_sizes'][1], 3)]
        # Flatten list of lists.
        data_windows = list(itertools.chain.from_iterable(data_windows))

        # Preliminary screening of the window to save time on pointless curve fitting.
        # If the mean intensity during the last third of the time window is not higher than the
        # mean intensity during the first third, a good increasing sigmoid is unlikely to be found.
        # Also, ignore data windows that contain NaNs, since these can't be fitted (NaNs should
        # only remain after interpolation if two tracks were merged and the resulting merged
        # track contains a large gap).
        data_windows = [dw for dw in data_windows
                        if dw.notna().all()
                        and dw.tail(len(dw) // 3).mean() > dw.head(len(dw) // 3).mean()]

        # If there are relatively few windows on which to run the curve fitting, there's no
        # point using multiprocessing, as spinning up a pool takes longer than it saves.
        if len(data_windows) < 300 or process_num == 1:
            fit_params = [fit_per_data_window(data_window) for data_window in data_windows]
        else:
            pn = process_num
            # If the number of processes to use is set to 'Auto', determine how many
            # processes to actually use based on the number of windows. For relatively
            # small inputs, using lots of processes can actually slow down the computation,
            # due to the time it takes to spin them up.
            if pn == 'auto':
                pn = ((lambda x:
                      x // 75 if (x < 75 * multiprocessing.cpu_count()) else multiprocessing.cpu_count())
                      (len(data_windows)))
            with multiprocessing.Pool(pn) as pool:
                fit_params = pool.map(fit_per_data_window, data_windows)

        # Turn the fitting parameters list into a dataframe for ease of use.
        fit_params = pd.DataFrame(
            data=[f for f in fit_params if f],
            columns=['Window start', 'Window end', 'Base', 'Height', 'Steepness', 'Midpoint', 'Midpoint CoV']
        )
        # Clean up a bit by discarding the worst 50% of fits based on parameter estimation variance.
        fit_params = fit_params[fit_params['Midpoint CoV'] < fit_params['Midpoint CoV'].quantile(0.5)]
        # Discard fits where the midpoint is too close to the edge of the data.
        fit_params = fit_params[(fit_params['Midpoint'] > data_to_fit.index[0] + params['dist_from_edge']) &
                                (fit_params['Midpoint'] < data_to_fit.index[-1] - params['dist_from_edge'])]
        # Give the remaining fits weights, based on time window length (longer is better),
        # relative height of the fit (higher is better, many bad fits are low),
        # and parameter estimation variance (lower is better).
        # Note: the parameter estimation variance ('Midpoint CoV') is rescaled by bumping all
        # its values up by a small constant. This prevents divide-by-zero errors and over-weighing
        # spurious low-CoV fits, without meaningfully altering the results.
        fit_params['Weights'] = (fit_params['Window end'] - fit_params['Window start']) \
            * (fit_params['Height'] / (data_to_fit.max() - data_to_fit.min())) \
            / (fit_params['Midpoint CoV'] + (fit_params['Midpoint CoV'].max() / 10))

        peaks = peak_properties = peak_width_results = None
        peak_info = {}
        # Screen fits by quality. Make sure the total weight of all remaining fits is above a threshold, since
        # low total weight indicates that most fits are not very good or that few fits were identified.
        # Also calculate the median of the parameter estimation variance of the fits, normalized by the weights,
        # and check if it's lower than a quality cutoff parameter.
        if (fit_params['Weights'].sum() / len(data_to_fit) > params['min_total_weight']
                and (fit_params['Midpoint CoV'] / fit_params['Weights']).median() < params['max_wcov']
                and len(fit_params) > 25):

            # Perform gaussian kernel density estimation to get a smoothed outline of the distribution of
            # identified midpoints, weighted (as explained above) to emphasize the more meaningful fits.
            kernel = stats.gaussian_kde(fit_params['Midpoint'], weights=fit_params['Weights'])
            # Get the probability density function of the kernel over the time axis.
            sigmoid_pdf = kernel.evaluate(data_to_fit.index)
            # Find prominent peaks in the PDF.
            # Minimum required prominence is the user-defined level divided by the minimum ratio between
            # prominences of nearby peaks. This ensures that peaks lower than the minimum will also be
            # identified, even if they will never be taken as the final result, to ensure that the real
            # peaks do not have meaningfully high neighbors.
            # The user defined minimum prominence level is first adjusted according to the track length,
            # since the gaussian KDE algorithm tends to identify peaks as lower in longer tracks.
            # If there are 2 peaks but they are just 2 timepoints apart, ignore the smaller one,
            # they should be treated as one peak.
            adj_min_prominence = params['prominence'] * 250 / len(fit_params)
            peaks, peak_properties = signal.find_peaks(
                sigmoid_pdf,
                prominence=adj_min_prominence / params['prominence_ratio'],
                distance=3
            )

            if peaks.size > 0:

                # Get the width of the peaks at half their height.
                peak_width_results = signal.peak_widths(sigmoid_pdf, peaks)
                # Construct a convenient dict with relevant info about the peaks.
                for peak_num, peak in enumerate(peaks):
                    peak_info[peak] = {'prominence': peak_properties['prominences'][peak_num],
                                       'width': peak_width_results[0][peak_num],
                                       'left_ips': peak_width_results[2][peak_num],
                                       'right_ips': peak_width_results[3][peak_num]}

                # If there's more than one peak in the sigmoid PDF, that means there's probably more than
                # one sigmoid in the dot intensity data. These could be different replication events taking
                # place in different cell cycles that were consecutively tracked, or it could just be noise.
                # We don't like noisy dots that display multiple sigmoids in a single cell cycle, since it's
                # hard to tell which is the real replication event. Therefore, only peaks that can be
                # attributed to separate cell cycles (separated by at least min_sigmoid_dist timepoints)
                # will be taken. If nearby peaks do exist, but they are much less prominent (by at least
                # prominence_ratio), they can be safely ignored.
                candidate_peaks = peak_info.copy()
                good_peaks = {}
                while len(candidate_peaks) > 0:

                    # Find the highest peak among those that are candidates for representing real sigmoids
                    # (which initially means all peaks).
                    highest_candidate = max(candidate_peaks, key=(lambda k: candidate_peaks[k]['prominence']))

                    # Check how far the current highest candidate peak is from all other peaks,
                    # and the ratios between their prominences.
                    peak_dists = {}
                    prominence_ratios = {}
                    for peak in peak_info:
                        if peak != highest_candidate:
                            peak_dists[peak] = abs(highest_candidate - peak)
                            prominence_ratios[peak] = (peak_info[highest_candidate]['prominence'] /
                                                       peak_info[peak]['prominence'])
                            # Any peak that is too close to a candidate cannot itself be a candidate.
                            if peak_dists[peak] < params['min_sigmoid_dist'] and peak in candidate_peaks:
                                del candidate_peaks[peak]

                    # Ensure the candidate peak is higher than the minimum required prominence, and not
                    # too wide, since a wide peak means less certainty about the exact midpoint of the
                    # sigmoid. The maximum allowed width is adjusted according to the track length,
                    # since the gaussian KDE algorithm tends to identify peaks as wider in longer tracks.
                    # Also, compare the location and prominence of the candidate peak with other peaks.
                    # If any of the other peaks are both too close and relatively high, this candidate
                    # peak will not be accepted.
                    # If all is good, move the peak from the candidate list to the good list.
                    adj_max_width = params['max_width'] * len(fit_params) / 250
                    if (peak_info[highest_candidate]['prominence'] > adj_min_prominence and
                            peak_info[highest_candidate]['width'] < adj_max_width and
                            not any(np.logical_and(
                                list(peak_dist < params['min_sigmoid_dist']
                                     for peak_dist in peak_dists.values()),
                                list(prominence_ratio < params['prominence_ratio']
                                     for prominence_ratio in prominence_ratios.values())
                            ))):
                        good_peaks[highest_candidate] = candidate_peaks[highest_candidate]

                    del candidate_peaks[highest_candidate]

                midpoints[particle_name] = {}
                for peak_num, peak in enumerate(good_peaks):
                    # The final "midpoint" we take is actually the median of many fitted midpoints,
                    # that are all within the half-height width of the selected peak.
                    # Also add 1 at the end because we eventually move to 1-indexed time axis.
                    midpoint_low_range = good_peaks[peak]['left_ips'] + data_to_fit.index[0]
                    midpoint_high_range = good_peaks[peak]['right_ips'] + data_to_fit.index[0]
                    midpoints[particle_name][peak_num] = fit_params['Midpoint'][
                        (fit_params['Midpoint'] > midpoint_low_range) &
                        (fit_params['Midpoint'] < midpoint_high_range)
                    ].median() + 1

            if params['dev_mode']:
                celldf[particle_name, 'sigmoidPDF'] = pd.DataFrame(sigmoid_pdf, index=data_to_fit.index)

        if params['dev_mode']:

            print('\nCompleted sigmoid fitting for track ' + particle_name + ' in cell ' + str(cell_num + 1))
            logging.info('Sigmoid fitting for track %s in cell num. %s:', particle_name, str(cell_num + 1))
            logging.info('Number of possible fits identified: %s', len(fit_params))
            logging.info('Sigmoid identified at timepoint: %s\n', peaks)

            dev_data[particle_name] = {
                'TotalWeight': fit_params['Weights'].sum() / len(data_to_fit),
                'WCov': (fit_params['Midpoint CoV'] / fit_params['Weights']).median()
            }
            if isinstance(peaks, np.ndarray) and peaks.size > 0:
                dev_data[particle_name]['Peaks'] = peaks
                dev_data[particle_name]['Prominences'] = [peak_info[peak]['prominence'] for peak in peak_info]
                dev_data[particle_name]['Peak Widths'] = [peak_info[peak]['width'] for peak in peak_info]

    # After finding midpoints for each particle, cluster them by cell cycle.
    midpoints_per_cycle = {}
    total_midpoints = sum(len(v) for v in midpoints.values())
    counter = 0
    while counter < total_midpoints:
        try:
            # Find the earliest midpoint (among all midpoints in all particles in this cell).
            # First find which particle contains the earliest midpoint, and then its position
            # in the 'midpoints' dict.
            p_min = min(midpoints, key=(lambda k: min(list(midpoints[k].values())) if midpoints[k] else np.inf))
            p_min_key = min(midpoints[p_min], key=(lambda k: midpoints[p_min][k]))
        except (ValueError, KeyError):
            # If the 'midpoints' dict is empty the loop is done.
            break
        midpoints_per_cycle[counter] = {}
        # Place the earliest midpoint in a new dict. The counter counts separate cell cycles.
        midpoints_per_cycle[counter][p_min] = midpoints[p_min][p_min_key]
        other_ps = [p for p in midpoints.keys() if p != p_min]
        for p in other_ps:
            # For each particle other than the one in which the earliest midpoint was found,
            # find the midpoints that are in the same cell cycle as the earliest.
            # There should be at most one midpoint in this list.
            midpoints_in_range = [k for k, v in midpoints[p].items()
                                  if abs(midpoints[p_min][p_min_key] - v) < params['min_sigmoid_dist']]
            if midpoints_in_range:
                # Place this midpoint in the new dict under the same cell cycle.
                midpoints_per_cycle[counter][p] = midpoints[p][midpoints_in_range[0]]
                del midpoints[p][midpoints_in_range[0]]
        del midpoints[p_min][p_min_key]
        # After deleting all midpoints belonging to the current cell cycle from the
        # 'midpoints' dict, the loop can run again for the next cell cycle.
        counter += 1

    return celldf, midpoints_per_cycle, dev_data


""" Summarize replication time results for all positions """


def export_rep_summary(rep_summary, excel_path, c_names, delta_t_names, delta_t_range=None):
    """
    Export replication summary to Excel using XlsxWriter
    """

    # Create summary Excel file.
    writer = pd.ExcelWriter(excel_path,
                            engine='xlsxwriter',
                            engine_kwargs={'options': {'nan_inf_to_errors': True}})
    # Write summary table to Excel.
    rep_summary.to_excel(
        writer,
        sheet_name='Summary',
        float_format="%.3f",
        freeze_panes=(1, 0)
    )

    workbook = writer.book
    float_format = workbook.add_format({'num_format': '0.00'})
    worksheet = writer.sheets['Summary']

    # Put the field name in each row of the table, in case it
    # only appears in the first row of each field.
    for row_num, row_value in rep_summary['Field'].items():
        if pd.isna(row_value):
            rep_summary.at[row_num, 'Field'] = rep_summary.at[row_num - 1, 'Field']

    # Write the name of each movie in the first column of the summary table, and merge cells.
    cell_format = workbook.add_format({'align': 'center', 'valign': 'vcenter', 'text_wrap': True})
    row_counter = 1
    for movie_name, size in rep_summary.groupby('Field', sort=False).size().items():
        if size > 1:
            worksheet.merge_range(row_counter, 1,
                                  row_counter + size - 1, 1,
                                  movie_name,
                                  cell_format)
        elif size == 1:
            worksheet.write(row_counter, 1,
                            movie_name,
                            cell_format)
        row_counter += size

    # Write some stats summarizing the midpoint results.
    row_counter = 2
    for c_name in c_names:
        worksheet.write(row_counter,
                        rep_summary.shape[1] + 2,
                        'Num. of midpoints found - ' + c_name + ':')
        worksheet.write(row_counter,
                        rep_summary.shape[1] + 6,
                        rep_summary[c_name].count())
        logging.info('Num. of midpoints found in channel %s: %s\n',
                     c_name, rep_summary[c_name].count())
        row_counter += 1

    good_cells = np.logical_and.reduce(rep_summary[c_names].notna(), axis=1).sum()
    worksheet.write(row_counter, rep_summary.shape[1] + 2, 'Num. of cells with all midpoints:')
    worksheet.write(row_counter, rep_summary.shape[1] + 6, good_cells)
    logging.info('Num. of cells with all midpoints: %s\n', good_cells)

    row_counter += 2
    for i, delta_t_name in enumerate(delta_t_names.values()):

        worksheet.write(row_counter + i * 3, rep_summary.shape[1] + 2,
                        'Median ' + delta_t_name.split('_')[1] + ' (' +
                        str(rep_summary[delta_t_name].count()) + ' cells):')
        worksheet.write(row_counter + i * 3,
                        rep_summary.shape[1] + 6,
                        rep_summary[delta_t_name].median(),
                        float_format)

        if delta_t_range:
            delta_t_in_range = rep_summary[delta_t_name][np.logical_and(
                rep_summary[delta_t_name] > delta_t_range[0],
                rep_summary[delta_t_name] < delta_t_range[1]
            )]
            worksheet.write(row_counter + 1 + i * 3, rep_summary.shape[1] + 2,
                            'Median (' + str(delta_t_in_range.count()) +
                            ' cells in range ' + str(delta_t_range[0]) +
                            '-' + str(delta_t_range[1]) + '):')
            worksheet.write(row_counter + 1 + i * 3,
                            rep_summary.shape[1] + 6,
                            delta_t_in_range.median(),
                            float_format)

    writer.close()


def create_rep_summary(excel_path, cells, prev_summary):
    """
    Create summary of replication results (from all movies) and export to Excel
    """

    # Construct a summary dataframe for all cells for which midpoints were found.
    # For each cell, the table will contain the midpoint of each track, as well
    # as the deltas between each pair of midpoints.
    c_names = [c for c in params['channel_order']
               if c in params['channels_to_fit'] and c is not None]
    for c in params['channels_to_fit']:
        if c not in params['channel_order']:
            c_names.append(c)
    delta_t_names = {c_pair: 'deltaT_' + c_pair[0] + '->' + c_pair[1]
                     for c_pair in itertools.combinations(c_names, 2)}
    summarydf = pd.DataFrame(columns=['Field', 'Cell', *c_names, *delta_t_names.values()])
    dsbdf = pd.DataFrame(columns=['DSB'])

    # For each cell in which a midpoint has been identified in at least one channel,
    # add a row to the dataframe with the midpoints and deltaT.
    row_counter = 0
    for movie_name, movie_data in cells.items():
        for cell_num, cell in movie_data.items():
            for cycle_midpoints in cell.midpoints.values():

                row_counter += 1
                summarydf.loc[row_counter, 'Field'] = movie_name
                # In Excel, cell numbering is 1-indexed.
                summarydf.loc[row_counter, 'Cell'] = cell_num + 1

                # Write midpoint for each track.
                midpoint_names = {}
                for particle_name in cycle_midpoints:
                    midpoint_names[particle_name] = particle_name.split('_')[0]
                    # Check that there is only one midpoint in this channel and cycle. If more than one
                    # midpoint was found per channel in a single cell cycle (this should only happen if
                    # there is more than one track in this channel), there is no point in outputting
                    # either of them in the summary, since no meaningful deltaT can be identified.
                    if (len([pn for pn, c in midpoint_names.items() if c == midpoint_names[particle_name]])
                            == 1):
                        summarydf.loc[row_counter, midpoint_names[particle_name]] = cycle_midpoints[particle_name]
                    else:
                        summarydf.loc[row_counter, midpoint_names[particle_name]] = 'Multiple'

                # Write deltaT for each pair of tracks in a given cell cycle, if exactly
                # one midpoint was found for each channel in this pair.
                for c_pair, delta_t_name in delta_t_names.items():
                    if (sum(c == c_pair[0] for c in midpoint_names.values()) == 1 and
                            sum(c == c_pair[1] for c in midpoint_names.values()) == 1):
                        summarydf.loc[row_counter, delta_t_name] = \
                            cycle_midpoints[
                                list(midpoint_names.keys())[list(midpoint_names.values()).index(c_pair[1])]
                            ] - \
                            cycle_midpoints[
                                list(midpoint_names.keys())[list(midpoint_names.values()).index(c_pair[0])]
                            ]

                # In DSB mode, prepare and additional dataframe with the DSB results.
                if params['dsb_channels']:
                    dsbdf.loc[row_counter, 'DSB'] = cell.dsb

    # Add columns with deltaT values in a specific range.
    if params['delta_t_range']:
        delta_t_names_range = ['deltaT_' + c_pair[0] + '->' + c_pair[1] + str(params['delta_t_range'])
                               for c_pair in itertools.combinations(c_names, 2)
                               ]
        summarydf[delta_t_names_range] = summarydf[delta_t_names.values()][np.logical_and(
            summarydf[delta_t_names.values()] > params['delta_t_range'][0],
            summarydf[delta_t_names.values()] < params['delta_t_range'][1]
        )]

    # Add DSB column to the summary.
    if params['dsb_channels']:
        dsbdf.loc[dsbdf.loc[:, 'DSB'] == 'Unknown', 'DSB'] = ''
        summarydf = summarydf.join(dsbdf)

    # Merge the current summary table with the summary of a previous AutoCRAT analysis.
    if prev_summary:
        try:
            # Import previous summary table.
            old_summary = pd.read_excel(Path(prev_summary),
                                        sheet_name='Summary',
                                        index_col=0)
            old_summary = old_summary.loc[:, [c for c in old_summary.columns if 'Unnamed' not in c]]
            # Make sure the column titles are identical; otherwise, the channel names,
            # deltaT names or deltaT range may be different and the tables can't be merged.
            if all(old_summary.columns == summarydf.columns):
                # Merge old and new summary tables.
                summarydf = pd.concat([old_summary, summarydf], axis=0, ignore_index=True)
                # 1-indexing in Excel for convenience.
                summarydf.index += 1
            else:
                print('Warning: Column titles of the previous summary file do not match '
                      'the current analysis. Summaries will not be merged.')
                logging.info('Warning: Column titles of the previous summary file do not match '
                             'the current analysis. Summaries will not be merged.')
        except FileNotFoundError:
            print('Previous summary file was not found. Summaries will not be merged.')
            logging.info('Previous summary file was not found. Summaries will not be merged.')
        # Value error could result from wrong sheet name in the previous
        # summary file or some other formatting mismatch.
        except ValueError:
            print('Previous summary file not properly formatted. Summaries will not be merged.')
            logging.info('Previous summary file not properly formatted. Summaries will not be merged.')

    if summarydf.shape[0] > 0:

        # Export replication summary to Excel.
        export_rep_summary(summarydf, excel_path, c_names, delta_t_names, params['delta_t_range'])

    else:
        print('\nReplication summary file was not created because no sigmoid midpoints were found.')
        logging.info('Replication summary file was not created because no sigmoid midpoints were found.\n')
