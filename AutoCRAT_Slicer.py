
""" Movie slicer module of AutoCRAT """


import numpy as np


def slicer(anchors, radius, movie, frame_dims, frame_num):
    """
    Slice cubes of a certain radius from a 3D movie frame around anchor points
    """

    # Get the relevant frame (3D image) from the movie.
    frame = movie[frame_num]

    slices = {}
    refs = {}
    for cell_num, a in anchors.items():

        # Define bottom and top radii around the anchor point.
        br = np.array(radius, copy=True)
        tr = np.array(radius, copy=True) + 1
        # In case the anchor point is close to the edge of the movie, the
        # bottom radius and the top radius are adjusted according to the
        # distance between the anchor point and the edge of the movie.
        br[a - br < 0] = a[a - br < 0]
        tr[a + tr > frame_dims - 1] = \
            frame_dims[a + tr > frame_dims - 1] \
            - a[a + tr > frame_dims - 1] \
            - 1
        # Slice the movie around the anchor point.
        slices[cell_num] = frame[
            a[0] - br[0]: a[0] + tr[0],
            a[1] - br[1]: a[1] + tr[1],
            a[2] - br[2]: a[2] + tr[2]
        ]
        # Save location reference for later use.
        # These are the numbers that must be added to each movie slice pixel
        # coordinate to get the original movie coordinates of that pixel.
        refs[cell_num] = a - br

    return slices, refs
