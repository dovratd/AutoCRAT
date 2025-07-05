
""" Automated Chromosome Replication Analysis & Tracking (AutoCRAT) """

#
# To run this script through the command line, navigate to the directory
# in which the script and the accompanying config file are saved, and enter:
# python AutoCRAT.py "path" "filename" "config" --positions --prev_summary --prev_RNSA
#
# To run from a Python console, enter:
# from AutoCRAT import main
# main(r"path", "filename", "config", "positions", r"prev_summary", r"prev_RNSA")
#


import sys
import argparse
import itertools
import time
import logging
from pathlib import Path
import warnings
import multiprocessing
from dataclasses import dataclass

import pims
import trackpy as tp
import numpy as np
import pandas as pd
import yaml
from scipy import cluster, stats
# For testing purposes:
#import matplotlib.pyplot as plt

# Additional dependencies:
# xlsxwriter
# openpyxl
# scikit-image to import TIFFs
# JPype1 and the Java runtime environment (JRE) to import using BioFormats
# With Python 3.12 or above: setuptools (required by PIMS)
# Optional: numba to significantly speed up parts of the analysis.


""" Movie import and input validation """


def import_modules():
    """
    Import optional modules
    """

    # Pass globals to modules.
    import AutoCRAT_cfg
    AutoCRAT_cfg.params = params
    AutoCRAT_cfg.process_num = process_num

    autocrat_modules = {}
    # Import optional AutoCRAT modules according to user input in the config file.
    if params.get('identify_by_coloc') or params.get('channels_to_gap_fill'):
        from AutoCRAT_IDCL import idcl
        autocrat_modules['idcl'] = idcl
    if params.get('dsb_channels'):
        from AutoCRAT_DSB import find_dsb, dsb_summary
        autocrat_modules['find_dsb'] = find_dsb
        autocrat_modules['dsb_summary'] = dsb_summary
    if params.get('channels_to_fit'):
        from AutoCRAT_RepTime import sigmoid_fit, create_rep_summary
        autocrat_modules['sigmoid_fit'] = sigmoid_fit
        autocrat_modules['create_rep_summary'] = create_rep_summary
    if params.get('rnsa_channels'):
        from AutoCRAT_RNSA import rnsa
        autocrat_modules['rnsa'] = rnsa
    if params.get('nuc_channel'):
        from AutoCRAT_Nuc import nuc_env, export_nuc_data
        autocrat_modules['nuc_env'] = nuc_env
        autocrat_modules['export_nuc_data'] = export_nuc_data

    # Inform user if numba is not available.
    if 'numba' not in sys.modules:
        print('\nThe Numba package is not available. '
              'This will make some aspects of the analysis significantly slower.')
        logging.info('The Numba package is not available. '
                     'This will make some aspects of the analysis significantly slower.\n')

    return autocrat_modules


def import_movies(path, filename, positions):
    """
    Import image data using PIMS
    """

    print('\nImporting movies...')

    movies = {}

    if params['import_mode'].casefold() == 'BioFormats'.casefold():

        # PIMS requires the loci_tools.jar file from BioFormats.
        # If it isn't found, it will be downloaded automatically.
        try:
            pims.bioformats._find_jar()
        except Exception:
            pims.bioformats.download_jar(version='6.7.0')

        # Silence spurious warning.
        warnings.filterwarnings('ignore', message='Due to an issue with JPype 0.6.0, reading is slower.')

        # To run AutoCRAT on multiple movies in BioFormats mode, the user can enter a 'generic' filename
        # by replacing the part of the filename that varies from movie to movie with an asterisk.
        movie_paths = list(Path(path).glob(filename))

        if not [movie_path.is_file() for movie_path in movie_paths]:
            raise ValueError('The path or filename provided are incorrect!')

        for movie_path in movie_paths:

            # Import the movie using PIMS BioFormats.
            movies[movie_path.stem] = pims.Bioformats(movie_path, java_memory='4096m')
            movies[movie_path.stem].iter_axes = 'ct'

            # Each file may contain multiple movies (different positions/fields of view in the same experiment).
            if movies[movie_path.stem].size_series > 1:

                # The Positions argument must be a number (the number of the position to analyze, 1-indexed)
                # or list of two numbers (the first and last positions to analyze, 1-indexed and inclusive).
                # This is intentionally inconsistent with Python convention for user convenience:
                # [1, 5] will analyze the first 5 positions, [6, 10] the next 5 positions, 4 or [4]
                # will analyze the 4th position.
                if positions:
                    if not isinstance(positions, list):
                        positions = [int(positions)]
                    if len(positions) == 1:
                        if 0 <= positions[0] - 1 < movies[movie_path.stem].size_series:
                            first_p = positions[0] - 1
                            last_p = positions[0]
                        else:
                            raise ValueError('Position argument is out of range!')
                    elif len(positions) == 2:
                        if 0 <= positions[0] - 1 < positions[1] and \
                                positions[0] <= positions[1] - 1 < movies[movie_path.stem].size_series:
                            first_p = positions[0] - 1
                            last_p = positions[1]
                        else:
                            raise ValueError('Positions arguments are out of range!')
                    else:
                        raise ValueError('Positions argument must be a number or list of two numbers!')
                else:
                    first_p = 0
                    last_p = movies[movie_path.stem].size_series

                # Import each position as a separate movie.
                for i in range(first_p, last_p):
                    try:
                        movies[movie_path.stem + ' - Position ' + str(i + 1)] = pims.Bioformats(
                            movie_path, series=i, java_memory='4096m')
                        movies[movie_path.stem + ' - Position ' + str(i + 1)].iter_axes = 'ct'
                    except (TypeError, IndexError):
                        # In some cases the file contains "phantom positions" that
                        # cannot actually be imported. They will be ignored.
                        continue
                del movies[movie_path.stem]

            elif positions:
                print('File "' + movie_path.stem + '" contains only one position, ignoring positions argument.')

    elif params['import_mode'].casefold() == 'TIFF sequence'.casefold():

        # To run AutoCRAT on multiple movies in TIFF sequence mode, each movie should be in a separate
        # directory. The user can enter a 'generic' path by replacing the part of the directory name
        # that varies from movie to movie with an asterisk.
        movie_paths = list(Path(path).parent.glob(Path(path).parts[-1]))
        movie_paths = [movie_path for movie_path in movie_paths if movie_path.is_dir()]

        if not movie_paths:
            raise ValueError('The path provided is incorrect!')

        for movie_path in movie_paths:

            try:
                # Import the movie using PIMS ImageSequenceND.
                movies[movie_path.stem] = pims.ImageSequenceND(str(Path(movie_path, filename)),
                                                               axes_identifiers=params['axes_identifiers'])
                movies[movie_path.stem].iter_axes = [params['axes_identifiers'][0],
                                                     params['axes_identifiers'][1]]
            except OSError:
                raise ValueError('The path provided does not contain files of the expected type!')

    else:
        raise ValueError('The import_mode parameter is incorrect!')

    return movies


def channel_check(movie):
    """
    Channel configuration and validation
    """

    try:
        # Try to find channel names in the movie metadata.
        metadata_c_names = [movie.metadata.ChannelName(0, c) for c in range(movie.sizes['c'])]
        # The channel names inputted by the user must be a subset of those in the metadata.
        if set(params['channel_names']).issubset(metadata_c_names):
            # Generate dict of channel numbers and names that is the same length as the number
            # of channels in the movie, with the user-selected channels in the correct order
            # and None for channels that will not be analyzed.
            c_names = {c_num: c_name
                       if c_name in params['channel_names'] else None
                       for c_num, c_name in enumerate(metadata_c_names)}
            all_c_names = metadata_c_names
        else:
            raise ValueError('Channel names in the config file do not match movie metadata! '
                             'The channels in the movie are: ' + str(metadata_c_names) +
                             '. Please change the channel_names parameter accordingly.')

    except (AttributeError, KeyError):
        # If channel name metadata was not found, the analysis relies on the channel names
        # in the config file, which must have a value for each channel in the movie.
        if len(params['channel_names']) == movie.sizes['c']:
            c_names = {c_num: c_name for c_num, c_name in enumerate(params['channel_names'])}
            print('\nChannel names not found, analyzing channels as described in the config file.')
            logging.info('Channel names not found, analyzing channels as described in the config file: %s\n',
                         params['channel_names'])
            all_c_names = params['channel_names']
        else:
            raise ValueError('The channel_names parameter lists ' + str(len(params['channel_names'])) +
                             ' channels, but the movie has ' + str(movie.sizes['c']) + ' channels!')

    if params['dot_tracking_channels']:
        # Channels to be analyzed using TrackPy.
        track_channels = {c_num: c_name
                          if c_name in params['dot_tracking_channels'] and
                          c_name not in params['identify_by_coloc'] else None
                          for c_num, c_name in c_names.items()}
    else:
        raise ValueError('At least one channel must be selected for tracking!')

    # Channels in which dots will be identified only by co-localization with other channels.
    # If the user also included these channels in dot_tracking_channels, it won't cause an
    # error, the identify_by_coloc parameter simply overrides dot_tracking_channels.
    coloc_only_channels = params['identify_by_coloc']

    if not set(params['required_channels']).issubset(track_channels.values()):
        raise ValueError('Required channels must be among those selected in dot_tracking_channels!')
    if set(params['required_channels']) & set(coloc_only_channels):
        raise ValueError('Channels cannot appear in both required_channels and identify_by_coloc!')

    # Gap filling only applies to channels that are analyzed by TrackPy.
    gap_fill_channels = [c_name for c_name in params.get('channels_to_gap_fill')
                         if c_name in track_channels.values()]

    if not set(params.get('channels_to_fit')).issubset([*track_channels.values(), *coloc_only_channels]):
        raise ValueError('Channels selected for sigmoid fitting must be among those selected '
                         'in dot_tracking_channels or identify_by_coloc!')

    if params.get('rnsa_channels'):
        if len(params['rnsa_channels']) != 3:
            raise ValueError('For Replisome-Normalized Signal Averaging, the rnsa_channels '
                             'parameter must contain exactly 3 channels!')
        if not set(params['rnsa_channels'][0:2]).issubset(params['channels_to_fit']):
            raise ValueError('The first two channels in rnsa_channels must be among those '
                             'in channels_to_fit!')
        if not {params['rnsa_channels'][2]}.issubset(
                [*track_channels.values(), *coloc_only_channels, params.get('nuc_channel')]
        ):
            raise ValueError('The third channel in rnsa_channels must be included in  '
                             'dot_tracking_channels, identify_by_coloc, or nuc_channel!')

    if not set(params['channel_order']).issubset([*track_channels.values(), *coloc_only_channels]):
        print('\nChannel order parameter contains channels that are not analyzed and will '
              'therefore be ignored.')
        logging.info('Channel order parameter contains channels that are not analyzed and '
                     'will therefore be ignored.\n')

    return c_names, all_c_names, track_channels, coloc_only_channels, gap_fill_channels


""" Dot identification and tracking """


def screen_dots(raw_particles, c_name):
    """
    Screen raw TrackPy results by dynamic thresholding
    """

    # The expected number of dots is determined by user input (plus another
    # 50% safety margin, which was empirically determined to improve output).
    expected_dots = round(params['num_of_cells'] * params['dots_per_cell'][c_name] * 1.5)

    screened_particles = []
    for frame, frame_particles in raw_particles.groupby('frame', sort=False):
        # For every timepoint, calculate Z-score for particle intensity,
        # to identify the dots that are statistical outliers in terms of
        # their unusually high brightness.
        frame_particles['Z-score'] = stats.zscore(frame_particles['mass'], nan_policy='omit')
        frame_particles = frame_particles.sort_values(by=['mass'])
        # The Z-score threshold for filtering is taken as an average of a fixed
        # value (2.5 standard deviations from the mean), and the Z-score
        # threshold attributable to the expected number of dots.
        # This tends to partially correct inaccuracies in user input.
        if len(frame_particles) > expected_dots:
            z_threshold = np.average([frame_particles['Z-score'].iloc[-expected_dots], 2.5])
        # If the total number of particles identified is smaller than the
        # expected number of dots (which should only happen in rare edge cases),
        # just take the fixed Z-score value.
        else:
            z_threshold = 2.5
        # Filter particles by Z-score to get the real dots and discard the noise.
        # Put filtered rows in new dataframe.
        screened_particles.append(frame_particles.loc[frame_particles['Z-score'] > z_threshold])

    return pd.concat(screened_particles, ignore_index=True)


def discard_chunks(particles, c_name, pixel_size):
    """
    Discard clusters of highly co-localized dots
    """

    screened_particles = particles
    for frame, frame_particles in particles.groupby('frame', sort=False):
        # Pre-screening to ignore frames with few particles.
        if frame_particles.shape[0] > params['dots_per_cell'][c_name] * 10:
            # Use Ward's hierarchical clustering algorithm on all particles in each frame.
            z = cluster.hierarchy.ward(
                frame_particles.loc[:, ['z', 'y', 'x']] * pixel_size)
            # Generate flattened clusters with a distance criterion equal to 1.5 times
            # the defined distance threshold for track co-localization. This ensures
            # particles will only be ignored if a large agglomeration is found within
            # an area that is not much larger than a single cell.
            frame_particles['cluster'] = cluster.hierarchy.fcluster(
                z, params['max_dist_threshold'] * 1.5, criterion='distance')
            # If a cluster is found with at least 3 times the expected average
            # number of dots per cell, all particles will be removed.
            p_to_del = []
            for _, cluster_particles in frame_particles.groupby('cluster', sort=False):
                if cluster_particles.shape[0] >= params['dots_per_cell'][c_name] * 3:
                    p_to_del.append(cluster_particles.index.values.astype(int))
            if p_to_del:
                p_to_del = np.concatenate(p_to_del).ravel()
            screened_particles = screened_particles.drop(p_to_del, axis=0)

    return screened_particles


def tracker(movie, track_channels, movie_length, pixel_size):
    """
    Locating fluorescent dots and particle tracking using TrackPy
    """

    tracks = {}
    for c_num, c_name in track_channels.items():

        if c_name is not None:

            print('\nAnalyzing ' + c_name + ' channel...')
            logging.info('Analysis of channel: %s\n', c_name)

            # If the user specifies a range of timepoints to analyze, take only these timepoints.
            if params['timepoints']:
                if params['timepoints'][0] >= 0 and params['timepoints'][1] <= movie_length:
                    timepoints_start = params['timepoints'][0]
                    timepoints_end = params['timepoints'][1]
                else:
                    raise IndexError('The timepoint parameters are out of range!')
            # If not specified, analyze the entire movie.
            else:
                timepoints_start = 0
                timepoints_end = movie_length

            # Find dots in current channel.
            # No intensity thresholding is used, since dynamic thresholding
            # will be performed later by screen_dots.
            particles = tp.batch(
                movie[c_num * movie_length + timepoints_start:
                      c_num * movie_length + timepoints_end],
                diameter=params['dot_diameter'][c_name],
                separation=params['dot_separation'][c_name],
                processes=trackpy_process_num,
            )

            # Get rid of rows that have nans where the coordinates should be for some reason.
            particles = particles[particles.x.notnull()]

            # For testing purposes:
            # tp.annotate(particles[particles['frame'] == ], movie[][])
            # tp.subpx_bias(particles)

            # Screen raw TrackPy results by dynamic thresholding, relying both on a
            # statistical approach and on user input (regarding the expected number of dots)
            # in order to distinguish between real dots and noise.
            particles_filtered = screen_dots(particles, c_name)

            # In rare cases (such as dead cells with high autofluorescence or several cells
            # on top of each other), multiple spurious dots may be found close together.
            # Such agglomerations of dots are identified by SciPy clustering methods and
            # removed before tracking is attempted (since they can cause tp.link to fail).
            particles_filtered = discard_chunks(particles_filtered, c_name, pixel_size)

            # Track dots over time. Search within search_range of the dot's previous location.
            # Remember dot even if it disappears for several timepoints (memory).
            tracks[c_name] = tp.link(
                particles_filtered,
                params['search_range'][c_name],
                memory=params['link_memory'][c_name]
            )

            # Filter all tracks shorter than a certain number of timepoints.
            tracks[c_name] = tp.filter_stubs(tracks[c_name], params['min_track_length'][c_name])

            # For testing purposes:
            # plt.imshow(movie[][], cmap='gray')
            # tp.plot_traj(tracks[c_name])

            # Print and log some info on segmentation and tracking.
            avg_particles = np.round(particles_filtered.shape[0] / particles_filtered['frame'].nunique(), 1)
            print('\nAverage num. of particles identified per timepoint in ' + c_name + ' channel: ' +
                  str(avg_particles))
            logging.info('Average num. of particles identified per timepoint in %s channel: %s\n',
                         c_name, avg_particles)

            num_of_tracks = tracks[c_name]['particle'].nunique()
            print('\nNum. of tracks identified in ' + c_name + ' channel: ' + str(num_of_tracks))
            logging.info('Num. of tracks identified in %s channel: %s\n', c_name, num_of_tracks)

            avg_track_len = np.round(
                tracks[c_name].groupby('particle', sort=False)['frame'].count().mean(),
                1)
            print('\nAverage track length for ' + c_name + ' channel: ' + str(avg_track_len))
            logging.info('Average track length for %s channel: %s\n', c_name, avg_track_len)

            # Change frame number and index to correspond to actual timepoints.
            tracks[c_name]['frame'] = tracks[c_name]['frame'] - (c_num * movie_length)
            tracks[c_name].index = tracks[c_name]['frame']

            # This column will be used later.
            tracks[c_name]['cell'] = np.nan
            tracks[c_name]['cell'] = tracks[c_name]['cell'].astype('Int64')

    # Re-arrange channels according to user-defined order. Channels not explicitly ordered
    # will be added based on the order in which they appear in the movie file.
    ordered_tracks = {c_name: tracks[c_name] for c_name in params['channel_order']
                      if c_name in track_channels.values()}
    for c_name in tracks.keys():
        if c_name not in ordered_tracks.keys():
            ordered_tracks[c_name] = tracks[c_name]

    if params['timepoints']:
        movie_length = params['timepoints'][1] - params['timepoints'][0]

    return ordered_tracks


""" Identification of cells as groups of co-localized tracks """


@dataclass
class Cell:
    name: str


def track_pair_coloc(track1, track2, overlap_threshold, avg_dist_threshold, pixel_size):
    """
    Examine if a given pair of tracks are co-localized
    """

    # Find temporal overlap between each pair of tracks.
    overlap, tr1_ind, tr2_ind = np.intersect1d(
        track1[:, 0],
        track2[:, 0],
        assume_unique=True,
        return_indices=True
    )

    if len(overlap) > overlap_threshold:
        # Calculate mean 3D-Euclidean distance (converted from pixels
        # to um using pixel size) during the temporal overlap.
        d = np.linalg.norm(
            (track1[tr1_ind, 1:4] -
             track2[tr2_ind, 1:4]) *
            pixel_size,
            axis=1
        ).mean()

        if d < avg_dist_threshold:
            return True
        else:
            return False
    else:
        return False


def identify_cells(tracks, pixel_size):
    """
    Examine track co-localization to define cells
    """

    # At first, each track in each channel is assigned a different cell number.
    counter = 0
    for c_name in tracks:
        for track_num in tracks[c_name]['particle'].unique():
            tracks[c_name].loc[tracks[c_name]['particle'] == track_num, 'cell'] = counter
            counter += 1

    # Convert the track coordinates from the pandas dataframe to a dict of
    # NumPy arrays of the form [t, z, y, x]. This makes alignment faster.
    track_coords = {c_name:
                    {track_num: group.to_numpy()
                     for track_num, group in tracks[c_name].groupby('particle', sort=False)[['frame', 'z', 'y', 'x']]
                     }
                    for c_name in tracks
                    }
    # Make a flattened list of all tracks in all channels, in (channel_name, track_number) format.
    track_list = list(itertools.chain.from_iterable(
        [[(c_name, track) for track in track_coords[c_name]]
         for c_name in track_coords]
    ))
    # Make a list of all unique pairs of tracks.
    track_pairs = list(itertools.combinations(track_list, 2))

    # Create list of sufficiently co-localized track pairs.
    # First, create a list of tuples containing the data on each pair of tracks
    # to feed track_pair_coloc using starmap.
    # This is done to enable multiprocessing.
    starmap_data = [(track_coords[tr_pair[0][0]][tr_pair[0][1]],
                     track_coords[tr_pair[1][0]][tr_pair[1][1]],
                     params['overlap_threshold'],
                     params['avg_dist_threshold'],
                     pixel_size)
                    for tr_pair in track_pairs]

    # Then, use track_pair_coloc to check if each pair of tracks is co-localized.
    # If there are not many track pairs on which to run track_pair_coloc, there's no point
    # using multiprocessing, as spinning up a pool takes longer than it saves.
    if len(starmap_data) < 800000 or process_num == 1:
        coloc_pairs_bool = [track_pair_coloc(*t) for t in starmap_data]
    else:
        pn = process_num
        # If the number of processes to use is set to 'Auto', determine how many processes
        # to actually use based on the number of track pairs. For relatively small inputs,
        # using lots of processes can actually slow down the computation, due to the time
        # it takes to spin them up.
        if pn == 'auto':
            pn = ((lambda x:
                   x // 200000 if (x < 200000 * multiprocessing.cpu_count()) else multiprocessing.cpu_count())
                  (len(starmap_data)))
        with multiprocessing.Pool(pn) as pool:
            coloc_pairs_bool = pool.starmap(track_pair_coloc, starmap_data)

    coloc_pairs = list(itertools.compress(track_pairs, coloc_pairs_bool))

    assigned_tracks = set()
    for tr_pair in coloc_pairs:
        # For each pair of co-localized tracks, get the first value from the 'cell' column
        # of each track (the values should be identical across the column, so the first one
        # is taken for convenience). Change the cell number of one of the tracks to match
        # the other. If one of the tracks in the pair has already been assigned a cell
        # number from a different track, make sure that cell number is assigned to both,
        # so all co-localized tracks will get the same cell number.
        if tr_pair[0] in assigned_tracks:
            tracks[tr_pair[1][0]].loc[tracks[tr_pair[1][0]]['particle'] == tr_pair[1][1], ['cell']] = (
                tracks[tr_pair[0][0]].loc[tracks[tr_pair[0][0]]['particle'] == tr_pair[0][1], ['cell']]
                .iloc[0].values)
        else:
            tracks[tr_pair[0][0]].loc[tracks[tr_pair[0][0]]['particle'] == tr_pair[0][1], ['cell']] = (
                tracks[tr_pair[1][0]].loc[tracks[tr_pair[1][0]]['particle'] == tr_pair[1][1], ['cell']]
                .iloc[0].values)
        assigned_tracks.add(tr_pair[0])
        assigned_tracks.add(tr_pair[1])

    # Find the number of cells with at least one track for each of the required channels and print.
    if params['required_channels']:
        cells_in_channel = {c_name: set(tracks[c_name].groupby('cell', sort=True).groups.keys())
                            for c_name in tracks if c_name in params['required_channels']}
        aligned_cells = set.intersection(*cells_in_channel.values())
    else:
        cells_in_channel = {c_name: set(tracks[c_name].groupby('cell', sort=True).groups.keys())
                            for c_name in tracks}
        aligned_cells = set.union(*cells_in_channel.values())
    print('\nNumber of cells identified with at least one track in each required channel: ' + str(len(aligned_cells)))
    logging.info('Number of cells identified with at least one track in each required channel: %s\n',
                 len(aligned_cells))

    return tracks, aligned_cells


def arrange_tracks(tracks, old_cell_num, movie_length):
    """
    Re-arrange track data into a new dataframe
    """

    # Create a new dataframe for each cell.
    celldf = pd.DataFrame(index=[i for i in range(movie_length)])

    particle_nums = {}
    for c_name in tracks:
        # Create temporary dataframe for all cell data in a particular channel.
        tempdf = tracks[c_name].loc[tracks[c_name]['cell'] == old_cell_num]
        # Re-organize dataframe such that each track is separate. This is important for cells with multiple
        # tracks ('particles') in the same channel.
        particle_nums[c_name] = []
        for particleNum, track in tempdf.groupby('particle', sort=True):
            celldf = celldf.join(track[['x', 'y', 'z', 'mass']], rsuffix=('_' + str(particleNum)))
            particle_nums[c_name].append(particleNum)

    # Create new hierarchical column titles for the dataframe, including channel name and track number.
    particle_names = []
    for c_name, particles in particle_nums.items():
        for particleNum in particles:
            particle_names.append(c_name + '_' + str(particleNum))
    celldf = celldf.set_axis(pd.MultiIndex.from_product([particle_names, ['x', 'y', 'z', 'Intensity']]), axis=1)

    return celldf, particle_names


""" Per-cell processing: dot distance calculation, clean-up, merging and clustering of tracks """


def track_dist_per_cell(celldf, particle_names, pixel_size):
    """
    Calculate distances between each pair of tracks in a given cell
    """

    dist_names = []
    distdf = pd.DataFrame(index=celldf.index)
    for i, pair in enumerate(itertools.combinations(particle_names, 2)):
        track_name1, track_name2 = pair
        dist_names.append(track_name1 + '->' + track_name2)
        distdf[dist_names[i]] = np.nan
        # Calculate 3D-Euclidean distance (converted from pixels to um
        # using pixel size) for each timepoint in the temporal overlap.
        distdf[dist_names[i]] = np.linalg.norm(
            (celldf[track_name1].loc[:, ['x', 'y', 'z']] -
             celldf[track_name2].loc[:, ['x', 'y', 'z']]) *
            list(reversed(pixel_size)),
            axis=1
        )

    distdf = distdf.set_axis(pd.MultiIndex.from_product([['Distances'], dist_names]), axis=1)

    return distdf


def screen_dots_by_dist(celldf, particle_names, distdf):
    """
    Screen dots by distance
    """

    # If the distance between any two dots is higher than the maximum allowable in any
    # particular timepoint, one of the dots identified at this timepoint is probably spurious,
    # the result of a tracking error (probably because the real dot was too weak).
    if distdf[distdf > params['max_dist_threshold']].any(axis=None):

        # To identify tracking errors by distance, calculate the Z-score for the location
        # of each dot at each timepoint (by summing the Z-scores for each dimension).
        # A high Z-score means the dot is unusually far from its location in most
        # other timepoints, so the data for this timepoint may be a spurious signal
        # that was accidentally identified as part of this track.
        zdf = pd.DataFrame(index=celldf.index, columns=particle_names)
        for particle_name in particle_names:
            zdf[particle_name] = np.abs(
                stats.zscore(celldf.loc[:, (particle_name, ['x', 'y', 'z'])], nan_policy='omit')
            ).sum(axis=1, skipna=False)

        # For each timepoint at which the distance between any two dots is higher than
        # the maximum allowable, check which of the dots has the highest Z-score,
        # with a minimum of 3 required for a dot to be considered an outlier.
        outliers = zdf[zdf > 3][distdf[distdf > params['max_dist_threshold']].any(axis=1)]
        outliers = outliers[outliers.notna().any(axis=1)].idxmax(axis=1)

        # Remove the data for the outlier dots.
        for row in outliers.index:
            celldf.loc[row, (outliers[row], slice(None))] = np.nan

    return celldf


def merge_tracks(celldf, particle_names):
    """
    Merge de-concatenated tracks in the same channel
    """

    # For each channel, if there's more than 1 track in that channel, and the overlap between
    # the tracks along the time axis is under the defined threshold, merge the tracks.
    for c_name in set([p.split('_')[0] for p in particle_names]):

        tracks_in_c = [p for p in particle_names if p.split('_')[0] == c_name]
        if len(tracks_in_c) > 1:

            # For each pair of tracks in the same channel:
            for pair in itertools.combinations(tracks_in_c, 2):

                # Making sure none of these tracks weren't already merged.
                if pair[0] in particle_names and pair[1] in particle_names:
                    # intersect1d can't be used here because the intersection might be incomplete,
                    # we need the actual delta between the ends of the two tracks.
                    earlier_track = np.argmin([celldf.loc[:, (pair[0], 'Intensity')].first_valid_index(),
                                               celldf.loc[:, (pair[1], 'Intensity')].first_valid_index()])
                    later_track = [0, 1][~earlier_track]
                    earlier_track = pair[earlier_track]
                    later_track = pair[later_track]
                    overlap = celldf.index[
                        ((celldf.index >= celldf[later_track].first_valid_index()) &
                         (celldf.index <= celldf[earlier_track].last_valid_index()))
                    ]

                    if overlap.size <= params['same_track_max_overlap']:
                        # Find the non-overlapping indices from the later track.
                        later_noi = celldf.index[
                            celldf.index > celldf[earlier_track].last_valid_index()
                            ]
                        # Copy the values from the non-overlapping indices of the later track
                        # into the earlier track.
                        celldf.loc[later_noi, earlier_track] = celldf.loc[later_noi, later_track].values

                        if overlap.size > 0:
                            # For each overlapping timepoint (if any), if the later track has higher
                            # intensity, replace x, y, z, and Intensity values from the earlier track
                            # with those from the later track.
                            celldf.loc[overlap, earlier_track] = \
                                celldf.loc[overlap, earlier_track].where(
                                    celldf.loc[overlap, ([earlier_track, later_track], ['Intensity'])]
                                    [celldf.loc[overlap, ([earlier_track, later_track], ['Intensity'])]
                                     .notna().any(axis=1)]
                                    .idxmax(axis=1) == (earlier_track, 'Intensity'),
                                    celldf.loc[overlap, later_track], axis=0
                                ).values

                        # Delete the later track.
                        celldf = celldf.drop(later_track, axis=1, level=0)
                        particle_names.remove(later_track)
                        if params['dev_mode']:
                            logging.info('Tracks %s and %s merged.', earlier_track, later_track)

    return celldf, particle_names


def cluster_tracks(distdf, particle_names, pixel_size):
    """
    Cluster tracks by co-localization
    """

    # Get new (post-merging) pairs of tracks.
    p_pairs = list(itertools.combinations(particle_names, 2))
    # Mean distance for each pair.
    mean_dists = {d: distdf.loc[:, ('Distances', d[0] + '->' + d[1])].mean() for d in p_pairs}
    # Ignore pairs of tracks that are not co-localized to within a range defined by dot_separation.
    # If the tracks come from channels that have different dot_separation parameters, take the mean.
    mean_dists = {pair: dist for pair, dist in mean_dists.items() if
                  dist < np.mean(
                      [np.linalg.norm(np.multiply(params['dot_separation'][pair[0].split('_')[0]], pixel_size)),
                       np.linalg.norm(np.multiply(params['dot_separation'][pair[1].split('_')[0]], pixel_size))]
                  ) / 2}

    # Any track that is not closely co-localized with other tracks shouldn't be relevant.
    # However, the relevant tracks may still contain several non-co-localized clusters,
    # or there could be several co-localized tracks from the same channel.
    relevant_particles = list(set(itertools.chain(*[[pair[0], pair[1]] for pair in mean_dists.keys()])))
    # Get the channel names for the relevant tracks.
    rp_cs = [p.split('_')[0] for p in relevant_particles]
    # If there is more than one track in any of the channels, find the closest partner
    # for every track in each of the other relevant channels.
    if len(set(rp_cs)) != len(rp_cs):

        partners = {}
        for particle in relevant_particles:

            # For each relevant track, find the tracks from other channels.
            oc_names = [c for c in params['dot_tracking_channels'] if c != particle.split('_')[0]]
            ps_in_oc = [p for p in relevant_particles if p.split('_')[0] in oc_names]
            # Any co-localized track that is unique in its channel is a partner.
            partners[particle] = [p for p in ps_in_oc if rp_cs.count(p.split('_')[0]) == 1]
            # Find the channels that contain more than one relevant track.
            oc_names_m = [c for c in oc_names if rp_cs.count(c) > 1]
            for oc in oc_names_m:
                ps_in_oc = [p for p in relevant_particles if p.split('_')[0] == oc]
                d_names = []
                # For every track in this other channel, find the distance pair
                # that contains both the track in question and this other track.
                for op in ps_in_oc:
                    d_names.append([d for d in list(mean_dists.keys()) if particle in d and op in d])
                if any(d_names):
                    # Find which track in this other channel is closest on average
                    # to the track in question. This will be the partner.
                    closest_pair = list(mean_dists.keys())[
                        list(mean_dists.values()).index(min([mean_dists[d[0]] for d in d_names if d]))
                    ]
                    partners[particle].append(*[p for p in closest_pair if p != particle])
                else:
                    for p in ps_in_oc:
                        partners[particle].append(p)

        # Re-order the partners dict so the tracks that are unique in their channel will
        # appear first. This can be meaningful for cluster construction in some edge cases.
        partners = {**{k: v for k, v in partners.items() if rp_cs.count(k.split('_')[0]) == 1},
                    **{k: v for k, v in partners.items() if rp_cs.count(k.split('_')[0]) > 1}}

        # To construct clusters, create lists of track names that are each other's
        # partners. Each track can appear in just one of these lists.
        clusters = []
        for k, vl in partners.items():
            if not any([k in cl for cl in clusters]):
                clusters.append([k])
            for vi in vl:
                if k in partners[vi] and not any([vi in cl for cl in clusters]):
                    clusters[next(i for i, cl in enumerate(clusters) if k in cl)].append(vi)

        # For certain rare particle configurations, it's possible there would still be more
        # than one track per channel in a single cluster. If that is the case, choose just one
        # track per channel in every cluster, by checking which of the tracks is closest on
        # average to the other tracks in the cluster (those that are unique in their channel).
        for i, cl in enumerate(clusters):
            p_cl = [p.split('_')[0] for p in cl]
            if len(set(p_cl)) != len(p_cl):
                # Singles - tracks that are the only representatives of their channel in this cluster.
                p_cl_s = [p for p in cl if p_cl.count(p.split('_')[0]) == 1]
                # Multiples - tracks which are not unique in their channel.
                p_cl_m = [p for p in cl if p_cl.count(p.split('_')[0]) > 1]
                a_dist = {}
                # For each one of the multiples, check the distance to all singles.
                for pm in p_cl_m:
                    d_names = []
                    for ps in p_cl_s:
                        d_names.append([d for d in list(mean_dists.keys()) if pm in d and ps in d])
                    a_dist[pm] = np.mean([mean_dists[d[0]] for d in d_names if d])
                # Of the multiples, only the one that is closest to the singles will remain.
                clusters[i] = [p for p in cl if p not in [pm for pm in p_cl_m if pm != min(a_dist, key=a_dist.get)]]

    # If there is only one track per channel among the relevant tracks, then these tracks
    # are the one cluster in this cell.
    else:
        clusters = [relevant_particles]

    return clusters


""" Export results of all cells in a single position to Excel """


def add_cell_chart(writer, cell, data_type, chart_row, colors, y_axis_name, y_axis_min, y_axis_max):
    """
    Construct chart with data for each particle in a cell
    """

    chart = writer.book.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
    for particle_name in cell.particle_names:

        color = colors[particle_name.split('_')[0]]

        try:
            data_col = cell.df.columns.get_loc((particle_name, data_type)) + 1
        except KeyError:
            continue

        chart.add_series({
            'categories': [cell.name, 3, 0, cell.df.shape[0] + 2, 0],
            'values': [cell.name, 3, data_col, cell.df.shape[0] + 2, data_col],
            'line': {'color': color},
            'name': particle_name,
            'marker': {
                'type': 'circle',
                'fill': {'color': color},
                'border': {'color': color}
            }
        })
    chart.set_x_axis({
        'name': 'Time',
        'name_font': {'size': 12},
        'min': cell.df.index[0],
        'max': cell.df.index[-1]
    })
    chart.set_y_axis({
        'name': y_axis_name,
        'name_font': {'size': 12},
        'min': y_axis_min,
        'max': y_axis_max
    })
    chart.set_title({'none': True})
    writer.sheets[cell.name].insert_chart(chart_row, cell.df.shape[1] + 2,
                                          chart,
                                          {'x_scale': 2, 'y_scale': 2})

    chart_row += 31

    return writer, chart_row


def export_results(excel_path, cells, colors):
    """
    Export results to Excel using XlsxWriter
    """

    # Create Excel file with raw data and sigmoid midpoints for each cell.
    writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

    # Make an Excel sheet for every cell.
    for cell in cells.values():

        # In Excel, we prefer the time axis to be 1-indexed.
        cell.df.index += 1

        # Write data to Excel.
        cell.df.to_excel(
            writer,
            sheet_name=cell.name,
            float_format="%.3f",
            freeze_panes=(2, 0)
        )
        # Row 3 will be hidden due to a bug in Pandas that creates an empty row.
        # If/when this bug is fixed, the following line should be deleted:
        writer.sheets[cell.name].set_row(2, None, None, {'hidden': True})

        chart_row = 3
        # Construct dot intensity chart.
        writer, chart_row = add_cell_chart(
            writer,
            cell,
            'Intensity',
            chart_row,
            colors,
            'Fluorescent Intensity',
            int(np.floor(cell.df.loc[:, (slice(None), 'Intensity')].min().min())),
            int(np.ceil(cell.df.loc[:, (slice(None), 'Intensity')].max().max()))
        )

        start_col = cell.df.shape[1] + 18

        midpointsdf = None
        # Write additional dataframe to Excel containing the sigmoid midpoints.
        if params.get('channels_to_fit'):

            if len(cell.midpoints) > 0:
                # Re-arrange sigmoid midpoint data into new single-level dict.
                midpoint_data = {}
                if len(cell.midpoints) == 1:
                    midpoint_data = cell.midpoints[0]
                elif len(cell.midpoints) > 1:
                    for c in cell.midpoints:
                        for p in cell.midpoints[c]:
                            midpoint_data[str(str(p) + '_' + str(c))] = cell.midpoints[c][p]

                midpointsdf = pd.DataFrame(index=list(midpoint_data.keys()), columns=['Midpoint'])
                for particle_name in midpoint_data:
                    midpointsdf.at[particle_name, 'Midpoint'] = midpoint_data[particle_name]

                midpointsdf.to_excel(
                    writer,
                    sheet_name=cell.name,
                    float_format="%.3f",
                    startrow=3,
                    startcol=start_col
                )

        # Construct dot distances chart.
        if params['dist_mode']:

            chart = writer.book.add_chart({'type': 'scatter', 'subtype': 'straight_with_markers'})
            for distName in cell.df.loc[:, 'Distances'].columns:
                data_col = cell.df.columns.get_loc(('Distances', distName)) + 1
                chart.add_series({
                    'categories': [cell.name, 3, 0, cell.df.shape[0] + 2, 0],
                    'values': [cell.name, 3, data_col, cell.df.shape[0] + 2, data_col],
                    'name': distName,
                    'marker': {'type': 'circle'}
                })
            # Set the axes to correspond to the min and max of the data.
            chart.set_x_axis({
                'name': 'Time',
                'name_font': {'size': 12},
                'min': cell.df.index[0],
                'max': cell.df.index[-1]
            })
            chart.set_y_axis({
                'name': 'Distance (um)',
                'name_font': {'size': 12},
                'min': 0,
                'max': 4
            })
            chart.set_title({'none': True})
            writer.sheets[cell.name].insert_chart(chart_row, cell.df.shape[1] + 2,
                                                  chart,
                                                  {'x_scale': 2, 'y_scale': 2})
            chart_row += 31

        # Exporting information about double-strand breaks.
        if params.get('dsb_channels'):

            if isinstance(midpointsdf, pd.DataFrame):
                dsb_row = midpointsdf.shape[0] + 6
            else:
                dsb_row = 3
            writer.sheets[cell.name].write(dsb_row, start_col, 'DSB:')
            writer.sheets[cell.name].write(dsb_row, start_col + 1, cell.dsb)

        # Construct nuclear location chart.
        if params.get('nuc_channel') and 'Nuc. Loc.' in cell.df.columns.get_level_values(1):

            if params.get('nuc_vol_norm'):
                nuc_yaxis_name = 'Center  <--  Volume-normalized Nuclear Location  -->  Periphery'
            else:
                nuc_yaxis_name = 'Center  <--      Nuclear Location      -->  Periphery'

            writer, chart_row = add_cell_chart(
                writer,
                cell,
                'Nuc. Loc.',
                chart_row,
                colors,
                nuc_yaxis_name,
                0,
                1
            )

        # Exporting extra information, mostly for testing purposes and parameter refinement.
        if params['dev_mode'] and params.get('channels_to_fit'):

            if isinstance(midpointsdf, pd.DataFrame):
                dev_row = midpointsdf.shape[0] + 10
            else:
                dev_row = 7
            for i, particle_name in enumerate(cell.dev_data):

                writer.sheets[cell.name].write(i * 7 + dev_row, start_col, particle_name)
                dd_counter = 0
                for k, v in cell.dev_data[particle_name].items():

                    dd_counter += 1
                    writer.sheets[cell.name].write(i * 7 + dev_row + dd_counter, start_col, k)
                    writer.sheets[cell.name].write(i * 7 + dev_row + dd_counter, start_col + 2, str(v))

            # Construct sigmoid PDF chart.
            if 'sigmoidPDF' in cell.df.columns.get_level_values(level=1):

                writer, chart_row = add_cell_chart(
                    writer,
                    cell,
                    'sigmoidPDF',
                    chart_row,
                    colors,
                    'Sigmoid PDF',
                    0,
                    int(np.ceil(cell.df.loc[:, (slice(None), 'sigmoidPDF')].max().max()))
                )

    writer.close()


""" Summarize results for all positions """


def dist_summary(cells):
    """
    Log distance statistics for all movies
    """

    # Initialize an array for each pair of channels.
    all_dists = {}
    for pair in itertools.combinations(
            [c_name for c_name in params['channel_names'] if
             c_name is not None and
             (c_name in params['dot_tracking_channels'] or
              c_name in params['identify_by_coloc'])],
            2
    ):
        all_dists[pair] = np.array([])

    # For each distance column in each cell, get the channel names of the two particles
    # for which the distance was measured, and then append the data from this column
    # to the distance array pertaining to that pair of channels.
    # This only happens for particles that are from two different channels. Distances
    # between dots in the same channel are not included in this summary.
    for movie_data in cells.values():
        for cell in movie_data.values():
            for dist_column in cell.df['Distances']:
                c_names = dist_column.split('->')
                c_names = (c_names[0].split('_')[0], c_names[1].split('_')[0])
                if len(set(c_names)) == 2:
                    all_dists[[pair for pair in all_dists if set(pair) == set(c_names)][0]] = np.append(
                        all_dists[[pair for pair in all_dists if set(pair) == set(c_names)][0]],
                        cell.df['Distances'][dist_column],
                        axis=0)

    for pair in all_dists:
        logging.info('Distance between %s and %s dots: %sum (median of %s data points)\n',
                     pair[0],
                     pair[1],
                     round(np.nanmedian(all_dists[pair]), 3),
                     np.count_nonzero(~np.isnan(all_dists[pair])))


""" Main script """


def main(path, filename, config, positions=None, prev_summary=None, prev_rnsa=None):

    global params, process_num, trackpy_process_num

    try:
        # Read the YAML config file.
        # The config argument handed to AutoCRAT can include the '.yml' or not.
        yaml_file = open(Path(Path(config).stem + '.yml'), 'r')
        params = yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise FileNotFoundError('No YAML file called ' + Path(config).stem + '.yml was found!')

    # Prepare number of processes for multiprocessing.
    process_num = params['multi_processes']
    if isinstance(process_num, str):
        process_num = process_num.casefold()
    if isinstance(process_num, int):
        if process_num < 1:
            process_num = 1
        elif process_num > multiprocessing.cpu_count():
            process_num = multiprocessing.cpu_count()
    # Due to a bug with PIMS/TrackPy, TrackPy multiprocessing does not currently work with BioFormats files.
    if params['import_mode'].casefold() == 'BioFormats'.casefold():
        trackpy_process_num = 1
    else:
        trackpy_process_num = process_num

    # Prevent notices from TrackPy about each frame unless in dev mode.
    tp.quiet()
    if params['dev_mode']:
        tp.quiet(suppress=False)

    # Import optional modules.
    autocrat_modules = import_modules()

    # Import image data using PIMS.
    movies = import_movies(path, filename, positions)

    if params['import_mode'].casefold() == 'TIFF sequence'.casefold():
        path = Path(path).parent
        filename = Path(path).parts[-1]

    pos_string = ''
    if positions:
        if len(positions) == 1:
            pos_string = ' - Position ' + str(positions[0])
        elif len(positions) == 2:
            pos_string = ' - Positions ' + str(positions[0]) + '-' + str(positions[1])

    # Create a log file with details about the run.
    now = time.strftime('%Y%m%d %H%M', time.localtime(time.time()))
    logging.basicConfig(filename=Path(path, Path(filename.split('*')[0]).stem +
                                      pos_string + ' (AutoCRAT ' + now + ').log'),
                        level=logging.INFO, filemode='w',
                        format='%(asctime)s %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S')
    logging.info('\n --- Automated Chromosome Replication Analysis & Tracking (AutoCRAT) --- \n')
    logging.info('Movies that will be analyzed: %s\n', [movie_name for movie_name in movies.keys()])
    logging.info('Config file used: %s', Path(config).stem + '.yml')
    logging.info('Parameters in config file: %s\n', params)

    # Analyze each movie.
    cells = {}
    all_c_names = {}
    ext_nuc_data = {}
    for movie_name, movie in movies.items():

        print('\nAnalyzing ' + movie_name + '...')
        logging.info('\nAnalysis of movie: %s\n', movie_name)

        # Channel configuration and validation.
        c_names, all_c_names[movie_name], track_channels, coloc_only_channels, gap_fill_channels = channel_check(movie)

        movie_length = movie.sizes['t']

        # Get pixel dimensions, from movie metadata if possible.
        try:
            pixel_size = [movie.metadata.PixelsPhysicalSizeZ(0),
                          movie.metadata.PixelsPhysicalSizeY(0),
                          movie.metadata.PixelsPhysicalSizeX(0)]
        except (AttributeError, TypeError):
            pixel_size = params['pixel_size']

        # Locating fluorescent dots and particle tracking using TrackPy.
        tracks = tracker(movie, track_channels, movie_length, pixel_size)

        # Examine track co-localization to define cells.
        tracks, aligned_cells = identify_cells(tracks, pixel_size)

        # Analyze each cell.
        print('\nAnalyzing individual cells...')
        cells[movie_name] = {}
        for cell_num, old_cell_num in enumerate(aligned_cells):

            # Create Cell object. For display, the cell numbering is 1-indexed.
            cell = Cell('Cell_' + str(cell_num + 1))

            # Re-arrange track data into a new dataframe structure.
            cell.df, cell.particle_names = arrange_tracks(tracks, old_cell_num, movie_length)

            # Remove lines that are all NaNs before and after the data.
            cell.df = cell.df.loc[cell.df.first_valid_index():cell.df.last_valid_index()]

            # For each track identified in this cell, regardless of channel,
            # calculate pairwise Euclidean distance from all other tracks.
            cell.dists = track_dist_per_cell(cell.df, cell.particle_names, pixel_size)

            # Remove dots that are unusually distant from other dots and are probably tracking errors.
            cell.df = screen_dots_by_dist(cell.df, cell.particle_names, cell.dists)

            # Merge de-concatenated tracks in the same channel.
            cell.df, cell.particle_names = merge_tracks(cell.df, cell.particle_names)

            if params.get('max_tracks'):
                # Cells with many tracks (after merging) are probably spurious; discard them.
                if len(cell.particle_names) > params['max_tracks']:
                    if params['dev_mode']:
                        logging.info('Cell %s dropped because it had too many tracks.', cell_num + 1)
                    continue

            # After removing outliers and merging tracks, it's necessary to re-calculate
            # distances for the new pairs of tracks.
            cell.dists = track_dist_per_cell(cell.df, cell.particle_names, pixel_size)

            # Identify clusters of closely co-localized tracks (average distance within dot_separation
            # parameter) that contain no more than one track from each channel.
            cell.clusters = cluster_tracks(cell.dists, cell.particle_names, pixel_size)

            cells[movie_name][cell_num] = cell

        if coloc_only_channels or gap_fill_channels:
            # Identify dots by co-localization with other channels.
            cells[movie_name] = autocrat_modules['idcl'](
                cells[movie_name], movie, coloc_only_channels, gap_fill_channels, c_names
            )
            # Re-calculate distances again after IDCL.
            for cell in cells[movie_name].values():
                cell.dists = track_dist_per_cell(cell.df, cell.particle_names, pixel_size)

        if params['dist_mode']:
            # Add a distance column to the cells dataframe for each pair of tracks.
            for cell in cells[movie_name].values():
                cell.df = pd.concat([cell.df, cell.dists], axis=1)

        if params.get('dsb_channels'):
            # Identify double-strand breaks by distance between two dots.
            for cell_num, cell in cells[movie_name].items():
                cell.dsb = autocrat_modules['find_dsb'](cell.dists, cell.particle_names, cell_num)

        if params.get('channels_to_fit'):
            # Fit sigmoidal function to dot intensity data to find replication times.
            print('\nIdentifying replication times by sigmoid fitting...')
            for cell_num, cell in cells[movie_name].items():
                cell.df, cell.midpoints, cell.dev_data = (
                    autocrat_modules['sigmoid_fit'](cell.df, cell.particle_names, cell_num)
                )

        if params.get('nuc_channel'):
            # Model nuclear envelope by ellipsoid fitting and locate dots relative to it.
            print('\nIdentifying nuclear envelopes by ellipsoid fitting...')
            cells[movie_name], ext_nuc_data[movie_name] = (
                autocrat_modules['nuc_env'](cells[movie_name], movie, c_names, pixel_size)
            )

    print('\nSummarizing and exporting results to Excel...')

    for movie_name, movie in movies.items():

        # Color for each channel is taken from the PIMS metadata, converted to hex, and will be
        # used for the Excel charts. If channel color is not available, use the default colors
        # taken from the config file.
        try:
            colors = {c_name: '#%02x%02x%02x' %
                              tuple(np.int64(np.multiply(np.array(movie.colors), 255))
                                    [[num for num, name in enumerate(all_c_names[movie_name])
                                      if name == c_name][0]])
                      for c_name in all_c_names[movie_name]}
        except (AttributeError, TypeError):
            colors = params['channel_colors']

        # Export results (per movie) to Excel using XlsxWriter.
        excel_path = Path(path, movie_name + ' - Results (AutoCRAT ' + now + ').xlsx')
        export_results(excel_path, cells[movie_name], colors)

    if params['dist_mode']:
        # Log distance statistics for all movies.
        dist_summary(cells)

    if params.get('dsb_channels'):
        # Log double-strand break statistics for all movies.
        autocrat_modules['dsb_summary'](cells)

    if params.get('channels_to_fit'):
        # Create summary of replication results (from all movies) and export to Excel.
        excel_path = Path(path, Path(filename.split('*')[0]).stem +
                          pos_string + ' - Rep Summary (AutoCRAT ' + now + ').xlsx')
        autocrat_modules['create_rep_summary'](excel_path, cells, prev_summary)

        # Run "replisome-normalized signal averaging" analysis and export to Excel.
        if params.get('rnsa_channels'):
            excel_path = Path(str(excel_path).replace('Rep Summary', 'RNSA'))
            autocrat_modules['rnsa'](excel_path, cells, colors, prev_rnsa)

    if params.get('nuc_channel') and params.get('nuc_data'):
        # Export extended information regarding nuclear envelope fitting.
        excel_path = Path(path, Path(filename.split('*')[0]).stem +
                          pos_string + ' - Nuc. Data (AutoCRAT ' + now + ').xlsx')
        autocrat_modules['export_nuc_data'](ext_nuc_data, excel_path)

    print('\nAnalysis complete!')
    logging.info('Analysis complete!')


if __name__ == '__main__':

    if sys.argv[1:]:
        parser = argparse.ArgumentParser()
        parser.add_argument('path', type=str)
        parser.add_argument('filename', type=str)
        parser.add_argument('config', type=str)
        parser.add_argument('-p', '--positions', type=int, nargs='*')
        parser.add_argument('-s', '--prev_summary', type=str)
        parser.add_argument('-r', '--prev_RNSA', type=str)

        args = parser.parse_args()
        main(args.path, args.filename, args.config, args.positions, args.prev_summary, args.prev_RNSA)

    else:
        # Default arguments with which AutoCRAT will run if it wasn't
        # called from the command line with the arguments provided.
        # Fill in manually.
        path = r''
        filename = ''
        config = ''
        positions = []
        prev_summary = r''
        prev_RNSA = r''

        main(path, filename, config, positions, prev_summary, prev_RNSA)
