
""" Identify Dots by Co-Localization (IDCL) module of AutoCRAT """


import logging
import warnings

import trackpy as tp
import numpy as np
import pandas as pd

from AutoCRAT_cfg import params
from AutoCRAT_Slicer import slicer


def locate_in_movie_slice(movie_slice, dot_diameter, dot_separation, dev_mode):
    """
    Locate a single dot in a small movie slice using TrackPy
    """

    if not dev_mode:
        warnings.filterwarnings('ignore', message='Image contains no local maxima.')
        warnings.filterwarnings('ignore', message='All local maxima were in the margins.')
        warnings.filterwarnings('ignore', message='Image is completely black.')
        warnings.filterwarnings('ignore', message='I am interpreting the image as 3-dimensional. '
                                'If it is actually a 2-dimensional color image, convert it to grayscale first.')

    # Use TrackPy to identify a particle in the movie slice. No sensitivity threshold
    # is defined, and the single brightest particle is taken.
    particle = tp.locate(
        movie_slice,
        diameter=dot_diameter,
        separation=dot_separation,
        topn=1
    )
    if particle.empty:
        return None
    else:
        return particle[['z', 'y', 'x', 'mass']][particle['mass'].notna()]


def idcl(cells, movie, coloc_only_channels, gap_fill_channels, c_names):
    """
    Identify Dots by Co-Localization with dots in other channels
    """

    frame_dims = np.array(movie.shape[1:4])
    num_of_timepoints = movie.sizes['t']

    for c_name in [*coloc_only_channels, *gap_fill_channels]:

        print('\nIdentifying dots by co-localization for ' + str(c_name) + ' channel...')

        # For IDCL, a small cube will be identified around an anchor point, which is the
        # average location of the reference particles, and sliced out of the movie.
        # The radius around the anchor point will define the cube. Half of dot_diameter
        # plus half of dot_separation are used, because TrackPy can't find local maxima
        # that are within separation/2 from the edge of the movie, so really dots will
        # be found only within a distance of dot_diameter from the anchor point.
        radius = np.ceil(
            (np.array(params['idcl_slice_diameter'][c_name]) + np.array(params['dot_separation'][c_name]))
            / 2
        ).astype(int)
        # The movie iterates by 'ct', so find the first frame that corresponds
        # to the relevant channel.
        c_frame = ([num for num, name in c_names.items() if name == c_name][0]
                   * num_of_timepoints)

        p_in_c = {}
        anchors = {}
        for cell_num, cell in cells.items():

            # Find which tracks in this cell belong to the channel currently undergoing IDCL,
            # and which belong to other channels (which will be used as reference tracks).
            p_in_c[cell_num] = [p for p in cell.particle_names if p.split('_')[0] == c_name]
            oc_names = [n for n in params['dot_tracking_channels'] if n != c_name]
            ref_particles = [p for p in cell.particle_names if p.split('_')[0] in oc_names]

            # Check if any of the reference channels have more than one track each. If so,
            # IDCL can't be performed since we can't be sure which of these reference
            # tracks we should be co-localizing with.
            ref_cs = [p.split('_')[0] for p in ref_particles]
            if len(ref_cs) > len(oc_names):
                if params['dev_mode']:
                    logging.info('Identifying dots by co-localization is not '
                                 'possible for %s channel in cell %s.',
                                 c_name, str(cell_num + 1))
                continue

            # Find cluster that contains a track for each of the reference channels.
            r_cluster = [cl for cl in cell.clusters if all(p in cl for p in ref_particles)]

            # If there are multiple tracks in the channel currently selected for IDCL,
            # find which of these tracks is clustered with the reference tracks, and
            # select it for IDCL.
            if len(p_in_c[cell_num]) > 1:
                if len(r_cluster) == 1:
                    p_in_c[cell_num] = [p for p in r_cluster[0] if p.split('_')[0] == c_name]
                # If the tracks in the reference channels aren't in the same cluster,
                # IDCL can't be performed.
                else:
                    if params['dev_mode']:
                        logging.info('Identifying dots by co-localization is not '
                                     'possible for %s channel in cell %s.',
                                     c_name, str(cell_num + 1))
                    continue

            # If the channel selected for IDCL was not previously tracked and quantified,
            # then we are in "Identify only by co-localization" mode rather than
            # "Gap-filling by co-localization" mode. If this is the case, create a new
            # empty track and add it to the relevant data structures.
            if c_name in coloc_only_channels:
                p_in_c[cell_num] = [c_name + '_' + str(cell_num + 1)]
                cell.df = pd.concat([
                    cell.df,
                    pd.DataFrame(index=cell.df.index,
                                 columns=pd.MultiIndex.from_product([p_in_c[cell_num], ['x', 'y', 'z', 'Intensity']]))
                ], axis=1)
                cell.particle_names.append(*p_in_c[cell_num])
                if len(r_cluster) == 1:
                    cell.clusters[cell.clusters == r_cluster[0]].append(*p_in_c[cell_num])

            # If there is no track in the channel currently selected for IDCL in this cell,
            # or none of the tracks in this channel cluster with the tracks in the required
            # channels, IDCL can't be performed.
            if not p_in_c[cell_num]:
                if params['dev_mode']:
                    logging.info('Identifying dots by co-localization not '
                                 'possible for %s channel in cell %s.',
                                 c_name, str(cell_num + 1))
                continue

            anchors[cell_num] = {}
            # For every timepoint that does not yet have data:
            for t in cell.df[p_in_c[cell_num]][cell.df[p_in_c[cell_num]].isna().any(axis=1)].index:

                # If at least one of the reference tracks has values at this timepoint:
                if ~cell.df.loc[t, (ref_particles, slice(None))].isna().all():

                    # Get the anchor point: the average location of the reference particles,
                    # rounded to the closest integer.
                    anchors[cell_num][t] = np.array(
                        [cell.df.loc[t, (ref_particles, d)].mean().round().astype('int') for d in ['z', 'y', 'x']]
                    )

        # The anchor points are a nested dict in the form cell_num -> timepoint.
        # Re-configure it into a nested dict with the form timepoint -> cell_num.
        anchors_per_timepoint = {}
        for cell_num, cell_dict in anchors.items():
            for t, anchor in cell_dict.items():
                anchors_per_timepoint.setdefault(t, {})[cell_num] = anchor

        # For each timepoint, get multiple movie slices around all the anchor points,
        # then find a dot in each of them.
        for t, anchor_data in anchors_per_timepoint.items():

            # The movie iterates by 'ct', so the frame reference for the selected channel
            # has to be added to the timepoint to find the correct frame in the movie.
            frame_num = t + c_frame
            # Use the slicer module to get movie slices around each anchor point in
            # the current timepoint.
            movie_slices, loc_refs = slicer(anchor_data, radius, movie, frame_dims, frame_num)

            # Use TrackPy to find the brightest dot within each movie slice.
            results = {}
            for cell_num, movie_slice in movie_slices.items():
                results[cell_num] = locate_in_movie_slice(
                    movie_slice,
                    params['dot_diameter'][c_name],
                    params['dot_separation'][c_name],
                    params['dev_mode']
                )

            # Re-adjust the result coordinates for the full movie and fill in the cell dataframe.
            for cell_num, result in results.items():
                if result is not None:
                    result[['z', 'y', 'x']] = result[['z', 'y', 'x']] + loc_refs[cell_num]
                    cells[cell_num].df.loc[t, (p_in_c[cell_num], ['x', 'y', 'z', 'Intensity'])] = (
                        result[['x', 'y', 'z', 'mass']].values)

        # Fix issue with TrackPy results typing.
        for cell in cells.values():
            cell.df = cell.df.astype('float64')

        if params['dev_mode']:
            logging.info('IDCL performed for %s channel.\n', c_name)

    return cells
