from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt
import glob
import os 
import numpy as np 
import time
import cv2
import sys
from statsmodels.distributions.empirical_distribution import ECDF
from noise_model import *


CONFUSION_MATRIX = np.array(
        [[0.85714286,0.,0.14285714,0.,0.,0.,0.],
        [0.06666667,0.8,0.06666667,0.,0.,0.06666667,0.],
        [0.,0.,1.,0.,0.,0.,0.],
        [0.03636364,0.,0.01818182,0.94545455,0.,0.,0.],
        [0.,0.,0.,0.,1.,0.,0.],
        [0.,0.,0.,0.,0.,1.,0.],
        [0.,0.,0.,0.01408451,0.,0.,0.98591549]]).T
            


plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

GENERATE_NEW_BBS = True
GENERATE_IMGS = False

TEST_DIR = 'custom_fixed'
TEST_DIR = 'custom_fixed_bigtrain'
#TEST_DIR = 'custom_fixed_big-thresh0.05'
#TEST_DIR = 'custom_fixed_big-thresh0.1'
#TEST_DIR = 'custom_fixed_big-thresh0.3'
TEST_DIR = 'custom_fixed_big'

REFERENCE_RESULTS_DIR = 'results_old'


CORRESPONDING_CLASSES = {
        'person': 'mountainbike',
        'bicycle': 'car',
        'car' : 'truck', 
        'motorbike': 'dog',
        'aeroplane': 'horse', 
        'bus' : 'sheep',
        'train' : 'giraffe'
        }


CLASSES = ['mountainbike',
           'car',
           'truck',
           'dog',
           'horse',
           'sheep',
           'giraffe']


USE_CORRESPONDING = (TEST_DIR.find('yolo') == -1)


def get_overlap_fraction(bb1, bb2, getAreaDiff=False):
    """Returns the fraction of overlap (as a fraction of
    the smallest of the BBs) between two BBs.
    
    Parameters
    ----------
    bb1 : tuple(int, int, int, int)
        BB extent defined by min and max x-coordinates and
        min and max y-coordinates.
    bb2 : tuple(int, int, int, int)
        BB extent defined by min and max x-coordinates and
        min and max y-coordinates.
    getAreaDiff : bool, optional
        if true, will return the difference in area, by default False
    
    Returns
    -------
    float
        The overlap fraction, or area difference.
    """

    xmin1, xmax1, ymin1, ymax1, cls1 = bb1
    xmin2, xmax2, ymin2, ymax2, cls2 = bb2

    xmin1 = int(xmin1)
    xmax1 = int(xmax1)
    ymin1 = int(ymin1)
    ymax1 = int(ymax1)
    xmin2 = int(xmin2)
    xmax2 = int(xmax2)
    ymin2 = int(ymin2)
    ymax2 = int(ymax2)

    dx = np.min((xmax1, xmax2)) - np.max((xmin1, xmin2))
    dy = np.min((ymax1, ymax2)) - np.max((ymin1, ymin2))

    dx = float(dx)
    dy = float(dy)

    if (dx>=0) and (dy>=0):
        overlapping = dx*dy
        area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
        area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
        not_overlapping = area1 + area2 - 2 * overlapping
        if getAreaDiff: return float(np.abs(area1 - area2))/area1 if area1 != 0 else 0
        if overlapping == 0: return 0
        return overlapping /(not_overlapping+overlapping)
    else:
        return 0


def dist(p1, p2):
    return np.linalg.norm(p2-p1)


def center(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)

    x = xmax - xmin
    x /= 2
    y = ymax - ymin
    y /= 2

    return np.array([x,y])


def area(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return (xmax - xmin) * (ymax - ymin)


def get_min_dim(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return np.min((xmax-xmin, ymax-ymin))


def get_max_dim(bb):
    xmin, xmax, ymin, ymax, label = bb
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return np.max((xmax-xmin, ymax-ymin))


def plot_confusion_matrix(cm,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    plt.show() must be run to view the plot, this is not done
    by this function.
    
    Parameters
    ----------
    cm : np.ndarray
        the confusion matrix to plot
    normalize : bool, optional
        whether to normalise the matrix, by default True
    title : string, optional
        title of the plot, otherwise otherwise a sensible title is generated. By default None
    cmap : matplotlib.colormap, optional
        matplotlib colormap, by default plt.cm.Blues
    
    Returns
    -------
    matplotlib.Axes
        axes object of the plot generated
    """
    if not title:
        title = 'Normalized confusion matrix' if normalize else 'Confusion matrix, without normalization'

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(4,2))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=CLASSES, yticklabels=CLASSES,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax


def analyse(get_ps = True, lerr=0.8, cerr=0.8, plot=False):
    raise NotImplementedError("Currently in development")

    FPs = 0
    FNs = 0
    TPs = 0
    TNs = 0
    center_errors = []
    overlap_fraction = []
    overall_areas = []
    min_dims = []
    max_dims = []
    NFrames = 0
    FP_areas = []
    FP_mindims = []
    area_errors = []

    airsim_FPs = 0
    airsim_FNs = 0
    airsim_TPs = 0
    airsim_TNs = 0
    airsim_center_errors = []
    airsim_overlap_fraction = []
    airsim_overall_areas = []
    airsim_min_dims = []
    airsim_max_dims = []
    airsim_FP_areas = []
    airsim_FP_mindims = []
    airsim_area_errors = []

    norm = 0
    norm_airsim = 0

    confusion_matrix = np.zeros((7,7))
    airsim_confusion_matrix = np.zeros((7,7))

    for airsimFile, yoloFile, imname in zip(np.sort(list(glob.glob('../%s/bbs/airsim/*.dat' % (REFERENCE_RESULTS_DIR,)))), np.sort(list(glob.glob('../results/bbs/%s/*.dat' % (TEST_DIR, )))), np.sort(list(glob.glob('../results_old/images/true/*.png' )))):
        
        # imname = None

        NFrames += 1
        #print('comparing %s and %s...' % (os.path.basename(airsimFile), os.path.basename(yoloFile)))    

        groundTruthBBs = []
        airsimBBs = []
        yoloBBs = []

        for line in open(airsimFile).readlines():
            line = line.split(', ')
            groundTruthBBs.append(list(map(int, line[:4])) + [line[-1].replace('\n', '')] )
            
            if GENERATE_NEW_BBS:
                pass
            else:
                airsimBBs.append(list(map(int, line[4:8])) + [line[-1].replace('\n', '')] )

        if GENERATE_NEW_BBS and len(groundTruthBBs):

            classes = np.array(groundTruthBBs)[:,4]

            # also strip small out of truth...
            classes = np.array(groundTruthBBs)[:,4]
            groundTruthBBs, classes = strip_out_too_small(groundTruthBBs, classes, min_size=10)
            if len(groundTruthBBs):
                groundTruthBBs = np.concatenate((groundTruthBBs, np.array(classes)[:, None]), axis=1)
            else:
                groundTruthBBs = np.array([])


            classes = misclassify(list(classes), cm=CONFUSION_MATRIX)
            bbs = np.array(groundTruthBBs)[:,:4] if len(groundTruthBBs) else []
            airsimBBs = add_jitter(bbs, shape=(1000,500),length_scale_fraction = lerr, center_error_fraction=cerr)
            airsimBBs, classes = introduce_false_negatives(airsimBBs, classes, p=0.1, min_size=60)
            

            airsimBBs, classes = introduce_false_positives(airsimBBs, classes, shape=(1000,500), p=0.05)
            airsimBBs, classes = merge_close_bbs(airsimBBs, classes, area_similarity_factor=1.5, overlap_factor=0.7)
            _airsimBBs = airsimBBs
            if len(airsimBBs):
                airsimBBs = np.concatenate((np.array(airsimBBs), np.array(classes)[:, np.newaxis]), axis=1)
            else:
                airsimBBs = []

        for line in open(yoloFile).readlines():
            line = line.split(', ')
            if USE_CORRESPONDING: yoloBBs.append(list(map(int, line[:4])) + [CORRESPONDING_CLASSES[line[-1].replace('\n', '')]] )
            else: yoloBBs.append(list(map(int, line[:4])) + [line[-1].replace('\n', '')] )

            # fix min max problem
            _cp = yoloBBs[-1][2]
            yoloBBs[-1][2] = yoloBBs[-1][3]
            yoloBBs[-1][3] = _cp

            # fix coordinate problem?
            #yoloBBs[-1][2] = 500 - yoloBBs[-1][2]
            #yoloBBs[-1][3] = 500 - yoloBBs[-1][3]

        if GENERATE_IMGS and len(_airsimBBs):
            im = cv2.imread(imname) 
            

            airsimBBs = np.array(airsimBBs)[:, :4]
            airsimBBs = airsimBBs.astype(int)

            airsimBBs[:,2] = 500 - airsimBBs[:,2]
            airsimBBs[:,3] = 500 - airsimBBs[:,3]

            cp = airsimBBs.copy()
            airsimBBs[:,2:] = airsimBBs[:, :2]
            airsimBBs[:, :2] = cp[:, 2:]
            im = draw_bbs_on_image(im, airsimBBs)

            # plt.imshow(swapRB(im), cmap='gray', interpolation='bicubic')
            # plt.show()

            if not os.path.isdir('../results/images/new'):
                os.mkdir('../results/images/new')
            cv2.imwrite('../results/images/new/'+os.path.basename(imname), im)
            quit()


        # Step 1: Find matching BBs by finding pairwise overlap percentages
        overlap_matrix = np.zeros((len(groundTruthBBs), len(yoloBBs)))
        for i in range(len(groundTruthBBs)):
            for j in range(len(yoloBBs)):
                overlap_matrix[i][j] = get_overlap_fraction(groundTruthBBs[i], yoloBBs[j])
        
        area_matrix = np.zeros((len(groundTruthBBs), len(yoloBBs)))
        for i in range(len(groundTruthBBs)):
            for j in range(len(yoloBBs)):
                area_matrix[i][j] = get_overlap_fraction(groundTruthBBs[i], yoloBBs[j], getAreaDiff=True)
        
        if len(yoloBBs) and len(groundTruthBBs):
            cp = overlap_matrix.copy()
            overlap_matrix = np.zeros_like(overlap_matrix)
            overlap_matrix[cp.argmax(0), np.arange(cp.shape[1])] = cp[cp.argmax(0), np.arange(cp.shape[1])]


        # Step 1 repeated for airsim noise model
        overlap_matrix_airsim = np.zeros((len(groundTruthBBs), len(airsimBBs)))
        for i in range(len(groundTruthBBs)):
            for j in range(len(airsimBBs)):
                overlap_matrix_airsim[i][j] = get_overlap_fraction(groundTruthBBs[i], airsimBBs[j])
        
        airsim_area_matrix = np.zeros((len(groundTruthBBs), len(airsimBBs)))
        for i in range(len(groundTruthBBs)):
            for j in range(len(airsimBBs)):
                airsim_area_matrix[i][j] = get_overlap_fraction(
                    groundTruthBBs[i], airsimBBs[j], getAreaDiff=True)

        if len(airsimBBs) and len(groundTruthBBs):
            cp = overlap_matrix_airsim.copy()
            overlap_matrix_airsim = np.zeros_like(overlap_matrix_airsim)
            overlap_matrix_airsim[cp.argmax(0), np.arange(cp.shape[1])] = cp[cp.argmax(0), np.arange(cp.shape[1])]

        # Step 2: Cull by class if duplicate
        for i in range(len(groundTruthBBs)):
            trueClass = groundTruthBBs[i][-1]
            foundClasses = []
            correctFound = False
            for j in range(len(yoloBBs)):
                if overlap_matrix[i][j] == 0:
                    continue

                correctFound = correctFound or (trueClass == yoloBBs[j][-1])
                foundClasses.append((j, yoloBBs[j][-1]))

            for j in range(len(yoloBBs)):
                if correctFound and yoloBBs[j][-1] != trueClass:
                    overlap_matrix[i][j] = 0

        # Step 2 repeated for airsim noise model
        for i in range(len(groundTruthBBs)):
            trueClass = groundTruthBBs[i][-1]
            foundClasses = []
            correctFound = False
            for j in range(len(airsimBBs)):
                if overlap_matrix_airsim[i][j] == 0:
                    continue

                correctFound = correctFound or (trueClass == airsimBBs[j][-1])
                foundClasses.append((j, airsimBBs[j][-1]))

            for j in range(len(airsimBBs)):
                if correctFound and airsimBBs[j][-1] != trueClass:
                    overlap_matrix_airsim[i][j] = 0


        # Step 3: Cull by error if still duplicates
        if len(yoloBBs):
            cp = overlap_matrix.copy()
            overlap_matrix = np.zeros_like(overlap_matrix)
            overlap_matrix[np.arange(cp.shape[0]), cp.argmax(1)] = cp[np.arange(cp.shape[0]), cp.argmax(1)]

        # Step 3 repeated for the airsim noise model
        if len(airsimBBs):
            cp = overlap_matrix_airsim.copy()
            overlap_matrix_airsim = np.zeros_like(overlap_matrix_airsim)
            overlap_matrix_airsim[np.arange(cp.shape[0]), cp.argmax(1)] = cp[np.arange(cp.shape[0]), cp.argmax(1)]


        # Now get the data
        FPs += (len(yoloBBs) - np.sum(overlap_matrix != 0))
        FNs += np.sum(np.sum(overlap_matrix, axis=1) == 0)
        TPs += np.sum(overlap_matrix != 0)

        airsim_FPs += (len(airsimBBs) - np.sum(overlap_matrix_airsim != 0))
        airsim_FNs += np.sum(np.sum(overlap_matrix_airsim, axis=1) == 0)
        airsim_TPs += np.sum(overlap_matrix_airsim != 0)

        for j in range(len(yoloBBs)):
            for i in range(len(groundTruthBBs)):
                if overlap_matrix[i][j] == 0:
                    FP_areas.append(area(yoloBBs[j]))
                    FP_mindims.append(get_min_dim(yoloBBs[j]))
                    continue
                overlap_fraction.append(overlap_matrix[i][j])
                center_errors.append(dist(center(yoloBBs[j]), center(groundTruthBBs[i])))
                overall_areas.append(area(yoloBBs[j]))
                min_dims.append(get_min_dim(yoloBBs[j]))
                max_dims.append(get_max_dim(yoloBBs[j]))

                trueClassIndex = CLASSES.index(groundTruthBBs[i][-1])
                foundClassIndex= CLASSES.index(yoloBBs[j][-1])
                confusion_matrix[trueClassIndex][foundClassIndex] += 1

                area_errors.append(area_matrix[i][j])
                


        # now for airsim
        for j in range(len(airsimBBs)):
            for i in range(len(groundTruthBBs)):
                if overlap_matrix_airsim[i][j] == 0:
                    airsim_FP_areas.append(area(airsimBBs[j]))
                    airsim_FP_mindims.append(get_min_dim(airsimBBs[j]))
                    continue
                airsim_overlap_fraction.append(overlap_matrix_airsim[i][j])
                airsim_center_errors.append(dist(center(airsimBBs[j]), center(groundTruthBBs[i])))
                airsim_overall_areas.append(area(airsimBBs[j]))
                airsim_min_dims.append(get_min_dim(airsimBBs[j]))
                airsim_max_dims.append(get_max_dim(airsimBBs[j]))

                trueClassIndex = CLASSES.index(groundTruthBBs[i][-1])
                foundClassIndex= CLASSES.index(airsimBBs[j][-1])
                airsim_confusion_matrix[foundClassIndex][trueClassIndex] += 1

                airsim_area_errors.append(airsim_area_matrix[i][j])



    max_dims = np.array(max_dims)
    min_dims = np.array(min_dims)
    airsim_max_dims = np.array(airsim_max_dims)
    airsim_min_dims = np.array(airsim_min_dims)
    center_errors = np.array(center_errors)
    airsim_center_errors = np.array(airsim_center_errors)
    center_errors_norm = center_errors / (min_dims + max_dims) * 2
    airsim_center_errors_norm = airsim_center_errors / (airsim_min_dims + airsim_max_dims) * 2



    print('                  TinyYolo | Model')    
    print('FP Mean Per Frame: %.3f   | %.3f' % (FPs / float(NFrames), airsim_FPs / float(NFrames)))    
    print('FN Mean          : %.3f   | %.3f' % (FNs / float(NFrames), airsim_FNs / float(NFrames)))    
    print('TP Mean          : %.3f   | %.3f' % (TPs / (TPs + FNs), airsim_TPs / (airsim_TPs + airsim_FNs)))    
    print('Mean offsets     : %.3f   | %.3f' % (np.mean(center_errors_norm), np.mean(airsim_center_errors_norm)))    
    print('Mean overlap frac: %.3f   | %.3f' % (np.mean(overlap_fraction), np.mean(airsim_overlap_fraction)))
    print('Mean area error  : %.3f   | %.3f' % (np.mean(area_errors), np.mean(airsim_area_errors)))
    print('Normalisation    : %.3f   | %.3f' % (len(area_errors), len(airsim_area_errors)))



    overlap_fraction_large = np.delete(overlap_fraction, np.where(np.array(overall_areas) < 0))

    outDir = '../results/plots/nice'
    # outDir = '../results/plots/'+TEST_DIR
    if not os.path.isdir(outDir):
        os.mkdir(outDir)

    cm = confusion_matrix.astype(
        'float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    cmm = airsim_confusion_matrix.astype(
        'float') / airsim_confusion_matrix.sum(axis=1)[:, np.newaxis]
    MAP = np.mean(np.diagonal(cm))
    MAPmodel = np.mean(np.diagonal(cmm))
    if plot:
        name = 'Center Errors - Noise Model'
        ecdf = ECDF(airsim_center_errors_norm)

        fig, ax1 = plt.subplots(figsize=(4,2))
        ax1.set_xlabel(r'Bounding Box center offset')
        ax1.set_ylabel(r'Frequency in test set')
        ax1.hist(airsim_center_errors_norm, bins=np.arange(0, 2.05, 0.1))
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        color = 'tab:red'
        # we already handled the x-label with ax1
        ax2.set_ylabel('ECDF', color=color)
        ax2.plot(ecdf.x, ecdf.y, color=color)
        ax2.set_xlim((0, 2.0))
        ax2.set_ylim((0, 1.0))
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()
        plt.savefig(outDir + '/' + name + '.pdf', format='pdf', bbox_inches='tight')
        
        
        
        name = 'Center Errors - TinyYOLO'
        ecdf = ECDF(center_errors_norm)

        fig, ax1 = plt.subplots(figsize=(4,2))
        ax1.set_xlabel(r'Bounding Box center offset')
        ax1.set_ylabel(r'Frequency in test set')
        ax1.hist(center_errors_norm, bins=np.arange(0, 2.05, 0.1))
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        color = 'tab:red'
        # we already handled the x-label with ax1
        ax2.set_ylabel('ECDF', color=color)
        ax2.plot(ecdf.x, ecdf.y, color=color)
        ax2.set_xlim((0, 2.0))
        ax2.set_ylim((0, 1.0))
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()
        plt.savefig(outDir + '/' + name + '.pdf',
                    format='pdf', bbox_inches='tight')


        # name = 'Overlap Fraction - Model'
        # plt.figure(name)
        # plt.hist(airsim_overlap_fraction, bins=np.arange(0,1.05, 0.05))
        # plt.xlim((0,1))
        # plt.tight_layout()
        # plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')


        name = 'Area Errors - Noise Model'
        ecdf = ECDF(airsim_area_errors)

        fig, ax1 = plt.subplots(figsize=(4,2))
        ax1.set_xlabel(r'Bounding Box length scale offset / pixels')
        ax1.set_ylabel(r'Frequency in test set')
        ax1.hist(airsim_area_errors, bins=np.arange(0, 2.05, 0.1))
        ax1.tick_params(axis='y')

        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        color = 'tab:red'
        ax2.set_ylabel('ECDF', color=color)  # we already handled the x-label with ax1
        ax2.plot(ecdf.x, ecdf.y, color=color)
        ax2.set_xlim((0,2.0))
        ax2.set_ylim((0,1.0))
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()
        plt.savefig(outDir + '/' + name + '.pdf', format='pdf', bbox_inches='tight')

        name = 'Area Errors - YOLO'
        ecdf = ECDF(area_errors)

        fig, ax1 = plt.subplots(figsize=(4,2))
        ax1.set_xlabel(r'Bounding Box length scale offset / pixels')
        ax1.set_ylabel(r'Frequency in test set')
        ax1.hist(area_errors, bins=np.arange(0, 2.05, 0.1))
        ax1.tick_params(axis='y')
        ax1.set_ylim((0, 65))

        ax2 = ax1.twinx()  # instantiate a second axis that shares the same x-axis
        color = 'tab:red'
        # we already handled the x-label with ax1
        ax2.set_ylabel('ECDF', color=color)
        ax2.plot(ecdf.x, ecdf.y, color=color)
        ax2.set_xlim((0, 2.0))
        ax2.set_ylim((0, 1.0))
        ax2.tick_params(axis='y', labelcolor=color)

        plt.tight_layout()
        plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')

        # name = 'Overlap Fraction vs area - Model'
        # plt.figure(name)
        # plt.scatter(airsim_overlap_fraction, airsim_overall_areas)
        # plt.tight_layout()
        # plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')

        # name = 'Center Errors vs area - Model'
        # plt.figure(name)
        # plt.scatter(airsim_center_errors, airsim_overall_areas)
        # plt.tight_layout()
        # plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')

        # name = 'Overlap Fraction'
        # plt.figure(name)
        # plt.hist(overlap_fraction_large, bins=np.arange(0,1.05,0.05))
        # plt.xlim((0,1))
        # plt.tight_layout()
        # plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')

        # name = 'Overlap Fraction vs area'
        # plt.figure(name)
        # plt.scatter(overlap_fraction, overall_areas)
        # plt.tight_layout()
        # plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')

        # name = 'Center Errors vs area'
        # plt.figure(name)
        # plt.scatter(center_errors, overall_areas)
        # plt.tight_layout()
        # plt.savefig(outDir+'/'+name+'.pdf', format='pdf', bbox_inches='tight')

        print('Confusion Matrix - TinyYOLO')
        plot_confusion_matrix(confusion_matrix, title="Confusion Matrix Actual")
        plt.tight_layout()
        plt.savefig(outDir+'/Confusion.pdf', format='pdf', bbox_inches='tight')

        print('Confusion Matrix - Noise Model')
        plot_confusion_matrix(airsim_confusion_matrix, title="Confusion Matrix Model")
        plt.tight_layout()
        plt.savefig(outDir+'/AirsimConfusion.pdf',
                    format='pdf', bbox_inches='tight')

    print('NORMALISATION SIMULATOR: %d (%d)' % (len(airsim_area_errors), len(airsim_center_errors)))
    print('mAP: %.3f' % MAPmodel)
    print('NORMALISATION REAL:      %d (%d)' % (len(area_errors), len(center_errors)))
    print('mAP: %.3f' % MAP)

    KS_stat, p_val = ks_2samp(airsim_center_errors_norm, center_errors_norm)
    print('K-S Test for center errors: P-Value that distributions are same:', p_val, KS_stat)

    KS_stat, p_val2 = ks_2samp(airsim_overlap_fraction, overlap_fraction)
    print('K-S Test for overlap fractions: P-Value that distributions are same:', p_val2, KS_stat)

    KS_stat, p_val3 = ks_2samp(airsim_area_errors, area_errors)
    print('K-S Test for area errors fractions: P-Value that distributions are same:', p_val3, KS_stat)
    print('OVERALL P='+str(p_val*p_val3))

    if plot:
        plt.show()

    return p_val, p_val3, p_val2


if __name__ == '__main__':
    lerr = 0.175
    cerr = 0.7
    analyse(True, lerr, cerr, True)
