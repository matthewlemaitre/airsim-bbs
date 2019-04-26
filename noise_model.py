import numpy as np
import skimage.draw


def swapRB(im):
    """Swaps the first and third channels
    
    Parameters
    ----------
    im : np.ndarray
        (GBR) image array.
    
    Returns
    -------
    np.ndarray
        (RGB) image array.
    """
    cp = im.copy()
    cp[:, :, 0] = im[:, :, 2]
    cp[:, :, 2] = im[:, :, 0]
    return cp


def rectangle_perimeter(r0, c0, width, height, shape=None, clip=False):
    """Generates an array with a rectangle of ones and the rest zeros.
    
    Parameters
    ----------
    r0 : int
        starting row
    c0 : int
        starting column
    width : int
        width of rectangle
    height : int
        height of rectangle
    shape : (int, int), optional
        shape of array to return, by default None
    clip : bool, optional
        whether to clip result, by default False
    
    Returns
    -------
    np.array
        array of zeros except perimeter of rectangle specified.
    """
    rr, cc = [r0, r0 + width, r0 + width,
              r0], [c0, c0, c0 + height, c0 + height]

    return skimage.draw.polygon_perimeter(rr, cc, shape=shape, clip=clip)


def merge_close_bbs(bbs, classes=[],  area_similarity_factor=1.5, overlap_factor=0.5):
    """Merges BBs which are too close, emulating the behaviour of object detection algorithms.
    
    Parameters
    ----------
    bbs : list
        list of tuples of bounding box coordinates, in the form
        (int, int, int, int) corresponding to the min and max x 
        extents and then the min and max y extents.
    classes : list, optional
        list of strings giving the classes of each object. Only 
        objects of the same class are merged. Default is to ignore 
        this constraint if no classes are given.
    area_similarity_factor : float, optional
        factor giving similarity in area required before merging.
        For two BBs of areas A and B, if max(A,B) <
        area_similarity_factor * min(A,B), merging will occur if
        overlap is also sufficient. Default is 1.5.
    overlap_factor : float, optional
        Factor giving required overlap before merging. If, smallest
        BB shares at least overlap_factor * [the area of itself] of
        area with the larger BB, merging will occur if areas are also
        similar enough. Default is 0.5.
    
    Returns
    -------
    list
        List of bbs after merging, in the same format as the bbs param.
    """
    to_del = []
    to_add = []
    to_add_cls = []
    i = 0

    if type(classes) == np.ndarray:
        classes = list(classes)

    for min_x, max_x, min_y, max_y in bbs:
        j = 0
        for min_x2, max_x2, min_y2, max_y2 in bbs[i+1:]:

            if classes[i] != classes[j]:
                continue  # must be same class to merge

            area_1 = (max_x - min_x) * (max_y - min_y)
            area_2 = (max_x2 - min_x2) * (max_y2 - min_y2)
            if np.min((area_1, area_2)) * area_similarity_factor < np.max((area_1, area_2)):
                j += 1
                continue  # not similar enough sizes to merge

            # get the overlap area
            left = max(min_x, min_x2)
            right = min(max_x, max_x2)
            bottom = max(min_y, min_y2)
            top = min(max_y, max_y2)

            if left > right or bottom > top:
                overlap = 0
            else:
                overlap = (right - left) * (top - bottom)

            if overlap > overlap_factor * np.min((area_1, area_2)):
                # sys.stdout.write('m')
                # save the areas to be merged
                to_del.append(i)
                to_del.append(i+j+1)
                to_add.append((np.min((min_x, min_x2)),
                               np.max((max_x, max_x2)),
                               np.min((min_y, min_y2)),
                               np.max((max_y, max_y2))
                               ))
                to_add_cls.append(classes[i])
            j += 1
        i += 1

    # now merge the areas
    rng = np.sort(np.unique(to_del))[::-1]
    for i in rng:
        try:
            del bbs[i]
            del classes[i]
        except IndexError:
            print('This should not happen...')

    bbs += to_add
    classes += to_add_cls

    return bbs, classes


def add_jitter(bbs, length_scale_fraction=0.01, center_error_fraction=0.01, shape=(256, 128)):
    """Adds gaussian noise to the size and position of the BB.
    
    Parameters
    ----------
    bbs : list
        list of BBs, in the same form as for
    length_scale_fraction : float, optional
        fractional standard deviation for noise regarding the length scale
        of the BB, by default 0.01
    center_error_fraction : float, optional
        fractional standard deviation for the noise regarding the x-y
        position of the center of the BB, by default 0.01
    shape : tuple, optional
        shape of overall input image, by default (256,128)
    
    Returns
    -------
    list
        list of BBs, in the same form as the input parameter.
    """
    if len(bbs) == 0:
        return []
    bbs_w_error = []
    if type(bbs) == np.ndarray:
        bbs = list(bbs.astype(int))
    for l, r, b, t in bbs:
        error_x = np.random.normal(scale=center_error_fraction*(r-l))
        error_x_scale = np.random.normal(scale=length_scale_fraction*(r-l))
        error_y = np.random.normal(scale=center_error_fraction*(t-b))
        error_y_scale = np.random.normal(scale=length_scale_fraction*(t-b))

        l = np.max((int(np.round(l + error_x - error_x_scale / 2.)), 0))
        r = np.min(
            (int(np.round(r + error_x + error_x_scale / 2.)), shape[0]-1))
        b = np.max((int(np.round(b + error_y - error_y_scale / 2.)), 0))
        t = np.min(
            (int(np.round(t + error_y + error_y_scale / 2.)), shape[1]-1))

        bbs_w_error.append([l, r, b, t])

    return bbs_w_error


def draw_bbs_on_image(_im, bbs):
    """Returns an image with the BBs shown
    
    Parameters
    ----------
    _im : np.ndarray
        RBG image array to which the BBs apply
    bbs : list
        list of BBs to annotate, in the same form as merge_close_bbs
    
    Returns
    -------
    np.ndarray
        RGB image with annotations
    """
    im = _im.copy()
    for bb in bbs:
        min_x, max_x, min_y, max_y = bb[0], bb[1], bb[2], bb[3]
        min_x = int(min_x)
        max_x = int(max_x)
        min_y = int(min_y)
        max_y = int(max_y)
        rr, cc = rectangle_perimeter(min_x, min_y, max_x-min_x, max_y-min_y)
        im[rr, cc, 2] = 255
        # im[rr, cc, 0] = 0
        # im[rr, cc, 2] = 0
        rr, cc = rectangle_perimeter(
            min_x+1, min_y+1, max_x-min_x-2, max_y-min_y-2)
        im[rr, cc, 2] = 255
    return im


def introduce_false_negatives(bbs, classes=[], min_size=6, p=0.03):
    """Introduce false negatives to the bb list.
    
    Parameters
    ----------
    bbs : list
        list of BBs, in the same form as merge_close_bbs
    classes : list, optional
        list of classes corresponding to the BBs, by default []
    min_size : int, optional
        Minimum dimension of false negatives, by default 6
    p : float, optional
        probability of false negative, by default 0.03
    
    Returns
    -------
    list, list
        Returns BBs after adjestment for false negatives, and 
        also returns the adjusted class list, if given.
    """
    bbs_w_fns = []
    newclasses = []
    i = -1
    for l, r, b, t in bbs:
        i += 1
        if r-l < min_size or t-b < min_size:
            continue
        elif np.random.random() < p:
            continue
        else:
            newclasses.append(classes[i])
            bbs_w_fns.append([l, r, b, t])

    return bbs_w_fns, newclasses


def strip_out_too_small(bbs, classes=[], min_size=6):
    """Removes BBs which are unrealistically small
    
    Parameters
    ----------
    bbs : list
        list of BBs, in the same form as for merge_close_bbs
    classes : list, optional
        list of classes, by default []
    min_size : int, optional
        size below which to remove BBs, by default 6
    
    Returns
    -------
    list, list
        returns BB list and classes list after removing small BBs.
    """
    bbs_no_big = []
    newclasses = []
    i = -1
    for _ in bbs:
        l, r, b, t = _[:4]
        i += 1
        if r-l < min_size or t-b < min_size:
            continue
        else:
            newclasses.append(classes[i])
            bbs_no_big.append([l, r, b, t])

    return bbs_no_big, newclasses


def introduce_false_positives(bbs, all_classes, classes=[], min_size=6, p=0.05, shape=(256, 128)):
    """introduces false positives at random locations
    which are larger than the minimum BB size.
    FPs are added consecutively, s.t. the chance of 
    one FP is p, the chance of 2 FPs is p**2 etc.
    
    Parameters
    ----------
    bbs : list
        list of BBs, in the same format as for merge_close_bbs
    all_classes : list
        list of all classes from which FP classes are chosen
    classes : list, optional
        list of classes of the BBs, by default []
    min_size : int, optional
        size below which no FPs will be added, by default 6
    p : float, optional
        probability of adding one FP, by default 0.05
    shape : tuple, optional
        shape of the whole images, by default (256, 128)
    
    Returns
    -------
    list
        returns BBs and classes after adding FPs
    """
    bbs_w_fps = bbs
    while np.random.random() < p:
        x_start = np.random.randint(shape[0] - min_size)
        x_end = np.random.randint(x_start, shape[0])
        y_start = np.random.randint(shape[1] - min_size)
        y_end = np.random.randint(y_start, shape[1])
        bbs_w_fps.append([x_start, x_end, y_start, y_end])
        classes.append(np.random.choice(all_classes))
    return bbs_w_fps, classes


def misclassify(classes, CLASSES, cm):
    for i in range(len(classes)):
        c = CLASSES.index(classes[i])
        ps = np.squeeze(cm[:,c])
        classes[i] = np.random.choice(CLASSES, p=ps)

    return classes



