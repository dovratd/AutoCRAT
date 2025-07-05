
""" Replisome-Normalized Signal Averaging (RNSA) module of AutoCRAT """


import logging
from pathlib import Path

import pandas as pd

from AutoCRAT_cfg import params


def create_rnsa_summary(rnsadf_dict, rnsa_channels):
    """
    Create RNSA summary table
    """

    # Generate a summary dataframe with the mean of replisome-normalized signals
    # from all cells -/+ standard error of the mean, for each of the channels.
    rnsa_summarydf = pd.DataFrame(index=rnsadf_dict[rnsa_channels[-1]].index)
    for c_name in rnsa_channels:
        rnsa_summarydf = rnsa_summarydf.join([
            rnsadf_dict[c_name].mean(axis=1).rename('Mean-sem_' + c_name) -
            rnsadf_dict[c_name].sem(axis=1, ddof=0).rename('Mean-sem_' + c_name)
        ])
        rnsa_summarydf = rnsa_summarydf.join(
            rnsadf_dict[c_name].mean(axis=1).rename('Mean_' + c_name)
        )
        rnsa_summarydf = rnsa_summarydf.join([
            rnsadf_dict[c_name].mean(axis=1).rename('Mean+sem_' + c_name) +
            rnsadf_dict[c_name].sem(axis=1, ddof=0).rename('Mean+sem_' + c_name)
        ])

    rnsa_summarydf = rnsa_summarydf.set_axis(pd.MultiIndex.from_product(
        [rnsa_channels, ['-SEM', 'Mean', '+SEM']]
    ), axis=1)
    # Remove lines that are all NaNs before and after the data.
    rnsa_summarydf = rnsa_summarydf.loc[
                     rnsa_summarydf.first_valid_index():
                     rnsa_summarydf.last_valid_index()
                     ]

    return rnsa_summarydf


def export_rnsa_summary(rnsa_summarydf, writer, rnsa_channels, colors, rnsa_x_axis, rnsa_y_axis, rnsa_y2_axis=None,
                        rnsa_x_name='Time', rnsa_y_name='Fluorescent Intensity', rnsa_y2_name=None):
    """
    Export RNSA summary table to Excel and create RNSA chart
    """

    # Write table to Excel.
    rnsa_summarydf.to_excel(
        writer,
        sheet_name='RNSA_Summary',
        float_format="%.3f",
        freeze_panes=(2, 0)
    )
    # Row 3 will be hidden due to a bug in Pandas that creates an empty row.
    # If/when this bug is fixed, the following line should be deleted:
    writer.sheets['RNSA_Summary'].set_row(2, None, None, {'hidden': True})

    # Construct RNSA chart.
    max_row = rnsa_summarydf.shape[0] + 2
    chart = writer.book.add_chart({'type': 'scatter', 'subtype': 'straight'})
    for i, c in enumerate(rnsa_channels):
        color = colors[c]
        # The first two channels (arrays) will be thinner and more transparent than the third.
        c_width = [1, 1, 2, 2][i]
        c_transparency = [40, 40, 0, 0][i]
        for j in range(3):
            # SEM lines will be dashed and more transparent than mean lines.
            dash_type = ['round_dot', 'solid', 'round_dot'][j]
            s_width = c_width * [1, 1.5, 1][j]
            s_transparency = c_transparency + [30, 0, 30][j]
            s_name = c + ['-SEM', '_Mean', '+SEM'][j]
            chart.add_series({
                'categories': ['RNSA_Summary', 3, 0, max_row, 0],
                'values': ['RNSA_Summary', 3, i * 3 + j + 1, max_row, i * 3 + j + 1],
                'line': {
                    'color': color,
                    'width': s_width,
                    'dash_type': dash_type,
                    'transparency': s_transparency
                },
                'name': s_name,
                'y2_axis': ' - Nuc. Loc.' in c
            })

    chart.set_legend({'none': True})
    # Set the axes to correspond to the min and max of the data.
    chart.set_x_axis({
        'name': rnsa_x_name,
        'min': rnsa_x_axis[0],
        'max': rnsa_x_axis[1],
        'crossing': rnsa_x_axis[0]
    })
    chart.set_y_axis({
        'name': rnsa_y_name,
        'min': rnsa_y_axis[0],
        'max': rnsa_y_axis[1],
        'major_gridlines': {'visible': False}
    })
    if rnsa_y2_name:
        chart.set_y2_axis({
            'name': rnsa_y2_name,
            'min': rnsa_y2_axis[0],
            'max': rnsa_y2_axis[1],
            'major_gridlines': {'visible': False}
        })
    writer.sheets['RNSA_Summary'].insert_chart(3, rnsa_summarydf.shape[1] + 2,
                                               chart,
                                               {'x_scale': 2, 'y_scale': 2})

    # Return the RNSA chart object: this is not actually necessary for
    # AutoCRAT but is used in some other scripts that call this function.
    return chart


def export_rnsa(rnsadf_dict, excel_path, rnsa_channels, colors, rnsa_x_axis, rnsa_y_axis, rnsa_y2_axis=None,
                rnsa_x_name='Time', rnsa_y_name='Fluorescent Intensity', rnsa_y2_name=None):
    """
    Export RNSA results to Excel using XlsxWriter
    """

    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    # Create an Excel sheet for each RNSA channel.
    for c_name in rnsa_channels:
        # Remove lines that are all NaNs before and after the data.
        rnsadf_dict[c_name] = rnsadf_dict[c_name].loc[
                              rnsadf_dict[c_name].first_valid_index():
                              rnsadf_dict[c_name].last_valid_index()
                              ]

        # Write table to Excel.
        rnsadf_dict[c_name].to_excel(
            writer,
            sheet_name=c_name,
            float_format="%.3f",
            freeze_panes=(2, 0)
        )
        # Row 3 will be hidden due to a bug in Pandas that creates an empty row.
        # If/when this bug is fixed, the following line should be deleted:
        writer.sheets[c_name].set_row(2, None, None, {'hidden': True})

    # Export RNSA summary table to Excel and create RNSA chart.
    _ = export_rnsa_summary(rnsadf_dict['Summary'], writer, rnsa_channels, colors, rnsa_x_axis,
                            rnsa_y_axis, rnsa_y2_axis, rnsa_x_name, rnsa_y_name, rnsa_y2_name)

    writer.close()


def rnsa(excel_path, cells, colors, prev_rnsa):
    """
    Run Replisome-Normalized Signal Averaging analysis and export to Excel
    """

    # If the signal to be normalized to replisome progression (the third RNSA channel)
    # is the intensity of a fluorescent dot that was tracked, the RNSA module will
    # operate in 'dot mode', creating three replisome-normalized data series (one for
    # each channel specified in the rnsa_channels parameter).
    # If the signal to be normalized is the nuclear location of the two replication-
    # reporting dots, the RNSA module will operate in 'nuc mode', creating four data
    # series (one for dot intensity and one for nuclear location, for each of the
    # first two channels in rnsa_channels)
    if params['rnsa_channels'][2] in {*params['dot_tracking_channels'], *params['identify_by_coloc']}:
        rnsa_mode = 'dot'
        rnsa_c_names = params['rnsa_channels']
    elif params['rnsa_channels'][2] == params.get('nuc_channel'):
        rnsa_mode = 'nuc'
        rnsa_c_names = [params['rnsa_channels'][0],
                        params['rnsa_channels'][1],
                        params['rnsa_channels'][0] + ' - Nuc. Loc.',
                        params['rnsa_channels'][1] + ' - Nuc. Loc.'
                        ]
    else:
        raise ValueError('The third channel in rnsa_channels must be included in  '
                         'dot_tracking_channels, identify_by_coloc, or nuc_channel!')

    # Import RNSA Excel file from a previous AutoCRAT analysis.
    if prev_rnsa:
        try:
            old_rnsa = pd.read_excel(Path(prev_rnsa),
                                     sheet_name=rnsa_c_names,
                                     header=[0, 1],
                                     index_col=0)
        except FileNotFoundError:
            print('Previous RNSA file was not found. Files will not be merged.')
            logging.info('Previous RNSA file was not found. Files will not be merged.')
        # Value error probably means the sheet names in the previous RNSA file
        # don't match the current RNSA channels, so merging is impossible.
        except ValueError:
            print('Previous RNSA file does not match current analysis. Files will not be merged.')
            logging.info('Previous RNSA file does not match current analysis. Files will not be merged.')

    # Identify which cells should undergo RNSA.
    cells_for_rnsa = {}
    cell_counter = 0
    for movie_name, movie_data in cells.items():

        cells_for_rnsa[movie_name] = {}
        for cell_num, cell in movie_data.items():

            cells_for_rnsa[movie_name][cell_num] = {}
            for cycle, cycle_midpoints in cell.midpoints.items():

                pn = []
                # Find cells that have exactly one midpoint for each required replication channel.
                if [[k.split('_')[0] for k in cycle_midpoints.keys()].count(c)
                        for c in rnsa_c_names[:2]] == [1, 1]:
                    # Get the particle names for the tracks for which midpoints were found.
                    pn.append(*[k for k in cycle_midpoints.keys() if k.split('_')[0] == rnsa_c_names[0]])
                    pn.append(*[k for k in cycle_midpoints.keys() if k.split('_')[0] == rnsa_c_names[1]])
                    # In dot mode, get the correct particle name for the third RNSA channel
                    # using the track clustering results.
                    if rnsa_mode == 'dot':
                        try:
                            # Find the relevant cluster than contains both of the replication tracks.
                            r_cluster = [cl for cl in cell.clusters if pn[0] in cl and pn[1] in cl][0]
                            # Get the track that corresponds to the RNSA channel from the same cluster.
                            pn.append(*[k for k in r_cluster if k.split('_')[0] == rnsa_c_names[2]])
                        # If this fails, it's because no cluster exists that contains tracks from all
                        # three relevant channels, and RNSA analysis can't be performed for this cell.
                        except (IndexError, TypeError):
                            if params['dev_mode']:
                                logging.info('Cell %s in movie %s not included in RNSA because no appropriate '
                                             'cluster of tracks was found.', cell_num, movie_name)
                            continue
                    # In nuc mode, two signals will undergo replisome normalization: the nuclear
                    # locations for each of the two tracks for which midpoints were found.
                    elif rnsa_mode == 'nuc':
                        pn.append(pn[0] + ' - Nuc. Loc.')
                        pn.append(pn[1] + ' - Nuc. Loc.')
                    # Make sure the deltaT for this cell is inside the defined range.
                    delta_t = cycle_midpoints[pn[1]] - cycle_midpoints[pn[0]]
                    if params['delta_t_range'][0] <= delta_t <= params['delta_t_range'][1]:
                        # Create a dict with the relevant particle names and midpoints.
                        cells_for_rnsa[movie_name][cell_num][cycle] = {}
                        for p in pn[:2]:
                            cells_for_rnsa[movie_name][cell_num][cycle][p] = cycle_midpoints[p]
                        for p in pn[2:]:
                            cells_for_rnsa[movie_name][cell_num][cycle][p] = []
                        cell_counter += 1
                    else:
                        if params['dev_mode']:
                            logging.info('Cell %s in movie %s not included in RNSA because the deltaT '
                                         'value is out of range.', cell_num, movie_name)
                        continue

    if cell_counter > 0:

        # For each of the RNSA channels, create a dataframe with replisome-normalized data for all cells.
        rnsadfs = {}

        for c_name in rnsa_c_names:

            if len(params['normalize_x_axis_to']) == 1:
                # Create dataframe for each RNSA channel.
                rnsadfs[c_name] = pd.DataFrame()
            elif len(params['normalize_x_axis_to']) == 2:
                # Create dataframe with finely-spaced index, which will act as the new T axis for all
                # cells, as each cell will have its T axis normalized differently.
                rnsadfs[c_name] = pd.DataFrame(index=[i/1000 for i in range(-10000, 20000)])
            else:
                raise ValueError('normalize_x_axis_to parameter must contain either 1 or 2 channel designations!')

            for movie_name, movie_data in cells.items():

                for cell_num in cells_for_rnsa[movie_name]:

                    for cycle in cells_for_rnsa[movie_name][cell_num]:

                        if rnsa_mode == 'dot':
                            # Get the relevant particle name.
                            p_name = [k for k in cells_for_rnsa[movie_name][cell_num][cycle].keys()
                                      if k.split('_')[0] == c_name]
                            # Get the intensity data for the relevant track.
                            tempdf = (movie_data[cell_num].df.loc[:, (p_name, 'Intensity')]
                                      .xs('Intensity', axis=1, level=1))
                        elif rnsa_mode == 'nuc':
                            # Get the relevant particle name (without 'Nuc. Loc.').
                            p_name = [[k for k in cells_for_rnsa[movie_name][cell_num][cycle].keys()
                                      if k.split('_')[0] == c_name.split(' - Nuc. Loc.')[0]][0]]
                            # Get the intensity data or nuclear location data for the relevant track.
                            if ' - Nuc. Loc.' in c_name:
                                tempdf = (movie_data[cell_num].df.loc[:, (p_name, 'Nuc. Loc.')]
                                          .xs('Nuc. Loc.', axis=1, level=1))
                            else:
                                tempdf = (movie_data[cell_num].df.loc[:, (p_name, 'Intensity')]
                                          .xs('Intensity', axis=1, level=1))

                        if len(params['normalize_x_axis_to']) == 1:
                            # Get the replication time for the array to which the time axis should be aligned.
                            midpoint = cells_for_rnsa[movie_name][cell_num][cycle][
                                [k for k in cells_for_rnsa[movie_name][cell_num][cycle].keys()
                                 if k.split('_')[0] == params['normalize_x_axis_to'][0]][0]
                            ]
                            tempdf.index = (tempdf.index - midpoint).round()

                        elif len(params['normalize_x_axis_to']) == 2:
                            # Get the replication times for the first and second array according
                            # to which the time axis should be normalized.
                            first_midpoint = cells_for_rnsa[movie_name][cell_num][cycle][
                                [k for k in cells_for_rnsa[movie_name][cell_num][cycle].keys()
                                 if k.split('_')[0] == params['normalize_x_axis_to'][0]][0]
                            ]
                            second_midpoint = cells_for_rnsa[movie_name][cell_num][cycle][
                                [k for k in cells_for_rnsa[movie_name][cell_num][cycle].keys()
                                 if k.split('_')[0] == params['normalize_x_axis_to'][1]][0]
                            ]
                            # Normalize the T axis, such that the time at which the first array is replicated will
                            # be 0 for each cell, and the time at which the second array is replicated will be 1.
                            tempdf.index = (tempdf.index - first_midpoint) / (second_midpoint - first_midpoint)
                            i_dist = round(1000 * (tempdf.index[-1]-tempdf.index[0])/len(tempdf))
                            # Re-index the T axis so it will fit the new, finely-spaced T axis. Each timepoint
                            # will be matched with the nearest timepoint in the new axis.
                            tempdf = tempdf.reindex(rnsadfs[c_name].index, method='nearest', tolerance=0.0005)
                            # Interpolate to fill in the small gaps created by the new, finely-spaced T axis.
                            # Don't interpolate any large gaps (those that were present in the original data).
                            tempdf = tempdf.interpolate(limit=i_dist)

                        if params['normalize_y_axis']:
                            # Normalize the intensity values to between 0 and 1.
                            tempdf = (tempdf - tempdf.min()) / (tempdf.max() - tempdf.min())

                        # Move the temporary dataframe into the final one.
                        rnsadfs[c_name] = rnsadfs[c_name].join(tempdf, how='outer', rsuffix=('_' + str(movie_name)))

        # Create lists of movie and cell names for dataframe titles.
        # For Excel export, cell numbering is 1-indexed.
        movie_list = []
        cell_list = []
        for movie_name in cells_for_rnsa:
            for cell_num in cells_for_rnsa[movie_name]:
                for cycle in cells_for_rnsa[movie_name][cell_num]:
                    movie_list.append(movie_name)
                    if len(cells_for_rnsa[movie_name][cell_num]) == 1:
                        cell_list.append('Cell_' + str(cell_num + 1))
                    elif len(cells_for_rnsa[movie_name][cell_num]) > 1:
                        cell_list.append('Cell_' + str(cell_num + 1) + '_' + str(cycle))

        for c_name in rnsa_c_names:

            # Rename according to Excel file numbering.
            rnsadfs[c_name] = rnsadfs[c_name].set_axis(pd.MultiIndex.from_arrays([movie_list, cell_list]), axis=1)
            try:
                # Concatenate old and new RNSA results.
                rnsadfs[c_name] = pd.concat([old_rnsa[c_name], rnsadfs[c_name]], axis=1)
            except NameError:
                pass

        # Create RNSA summary table.
        rnsadfs['Summary'] = create_rnsa_summary(rnsadfs, rnsa_c_names)

        # Appropriate names for the axis titles on the RNSA summary chart.
        if len(params['normalize_x_axis_to']) == 2:
            rnsa_x_name = 'Replisome-Normalized Time'
        else:
            rnsa_x_name = 'Time'
        if params['normalize_y_axis']:
            rnsa_y_name = 'Normalized Fluorescent Intensity'
        else:
            rnsa_y_name = 'Fluorescent Intensity'
        if rnsa_mode == 'nuc':
            if params.get('nuc_vol_norm'):
                rnsa_y2_name = 'Volume-normalized Nuclear Location'
            else:
                rnsa_y2_name = 'Nuclear Location'
            colors = {k: v for k, v in colors.items() if k in rnsa_c_names}
            colors[rnsa_c_names[2]] = colors[rnsa_c_names[0]]
            colors[rnsa_c_names[3]] = colors[rnsa_c_names[1]]
        else:
            rnsa_y2_name = None

        # Export RNSA results and summary to Excel.
        export_rnsa(rnsadfs, excel_path, rnsa_c_names, colors, params['rnsa_x_axis'], params['rnsa_y_axis'],
                    params['rnsa_y2_axis'], rnsa_x_name, rnsa_y_name, rnsa_y2_name)

    else:
        print('\nRNSA file was not created because no appropriate cells were found.')
        logging.info('RNSA file was not created because no appropriate cells were found.\n')
