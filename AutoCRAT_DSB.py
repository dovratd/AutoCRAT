
""" Double-Strand Break (DSB) module of AutoCRAT """


import logging

from AutoCRAT_cfg import params


def find_dsb(distdf, particle_names, cell_num):
    """
    Identify double-strand breaks by distance between two dots
    """

    # The relevant particles are those in the two channels defined by the user.
    p_in_c0 = [p for p in particle_names if p.split('_')[0] == params['dsb_channels'][0]]
    p_in_c1 = [p for p in particle_names if p.split('_')[0] == params['dsb_channels'][1]]

    # To look for DSBs, there must be just one particle per cell in each of the relevant channels.
    if len(p_in_c0) > 1 or len(p_in_c1) > 1:
        if params['dev_mode']:
            logging.info('Cannot identify DSB for cell %s because there is more than one particle '
                         'per channel.', cell_num + 1)
        return 'Unknown'

    else:
        try:
            # Find the relevant column in the distances dataframe, if it exists.
            relevant_dist = distdf.loc[:, ('Distances', p_in_c0[0] + '->' + p_in_c1[0])]
            # Create rolling time windows within the distance data,
            # according to the defined time window size.
            data_windows = list(relevant_dist.rolling(params['dsb_time_window']))[(params['dsb_time_window'] - 1):]
            # Discard time windows in which more than 40% of the timepoints
            # are NaNs, since these are not informative.
            data_windows = [dw for dw in data_windows if sum(dw.isna()) / len(dw) <= 0.4]

            # If there are relatively few informative time windows (probably because
            # at least one of the relevant particles was not tracked well), a definitive
            # determination about DSBs cannot be made.
            if len(data_windows) / (distdf.shape[0] - params['dsb_time_window']) < 0.25:
                if params['dev_mode']:
                    logging.info('Cannot identify DSB for cell %s because the relevant particles '
                                 'are not sufficiently well tracked.', cell_num + 1)
                return 'Unknown'

            else:
                # Identify a DSB if the distance between the relevant dots exceeds the DSB
                # threshold distance defined by the user in at least one of the time windows.
                if [dw for dw in data_windows if dw.mean() > params['dsb_distance']]:
                    return 'Yes'
                else:
                    return 'No'

        except (KeyError, IndexError, ZeroDivisionError):
            if params['dev_mode']:
                logging.info('Cannot identify DSB for cell %s because distance between relevant '
                             'particles is unavailable.', cell_num + 1)
            return 'Unknown'


def dsb_summary(cells):
    """
    Log double-strand break statistics for all movies
    """

    dsb_list = []
    for movie_data in cells.values():
        for cell in movie_data.values():
            dsb_list.append(cell.dsb)
    dsb_yes = dsb_list.count('Yes')
    # Total cells includes 'Yes' and 'No', ignoring cells with 'Unknown'.
    dsb_total = dsb_list.count('Yes') + dsb_list.count('No')
    logging.info('Double-strand break analysis: %s cells were found to have a DSB out of %s cells '
                 'analyzed (%s%%).\n', dsb_yes, dsb_total, round((dsb_yes / dsb_total) * 100, 1))
