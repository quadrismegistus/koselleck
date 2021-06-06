from koselleck.imports import *

# DFPOC=None
# def get_data_paceofchange(ifn=FN_DATA_PACEOFCHANGE):
#     global DFPOC
#     if DFPOC is None:
#         DFPOC=pd.read_pickle(ifn)
#     return DFPOC


def make_foote(quart=FOOTE_W):
    tophalf = [-1] * quart + [1] * quart
    bottomhalf = [1] * quart + [-1] * quart
    foote = list()
    for i in range(quart):
        foote.append(tophalf)
    for i in range(quart):
        foote.append(bottomhalf)
    foote = np.array(foote)
    return foote

def foote_novelty(distdf, foote_size=5):
    foote=make_foote(foote_size)
    distmat = distdf.values if type(distdf)==pd.DataFrame else distdf
    
    axis1, axis2 = distmat.shape
    assert axis1 == axis2
    distsize = axis1
    axis1, axis2 = foote.shape
    assert axis1 == axis2
    halfwidth = axis1 / 2
    novelties = []
    for i in range(distsize):
        start = int(i - halfwidth)
        end = int(i + halfwidth)
        if start < 0 or end > (distsize - 1):
            novelties.append(0)
        else:
            novelties.append(np.sum(foote * distmat[start: end, start: end]))
    return novelties

def getyears():
    years=list(d.columns)
    return years


def diagonal_permute(d):
    newmat = np.zeros(d.shape)
    
    # We create one randomly-permuted list of integers called "translate"
    # that is going to be used for the whole matrix.
    
    xlen,ylen=d.shape
    translate = [i for i in range(xlen)]
    random.shuffle(translate)
    
    # Because distances matrices are symmetrical, we're going to be doing
    # two diagonals at once each time. We only need one set of values
    # (because symmetrical) but we need two sets of indices in the original
    # matrix so we know where to put the values back when we're done permuting
    # them.
    
    for i in range(0, xlen):
        indices1 = []
        indices2 = []
        values = []
        for x in range(xlen):
            y1 = x + i
            y2 = x - i
            if y1 >= 0 and y1 < ylen:
                values.append(d[x, y1])
                indices1.append((x, y1))
            if y2 >= 0 and y2 < ylen:
                indices2.append((x, y2))
        
        # Okay, for each diagonal, we permute the values.
        # We'll store the permuted values in newvalues.
        # We also check to see how many values we have,
        # so we can randomly select values if needed.
        
        newvalues = []
        lenvals = len(values)
        vallist = [i for i in range(lenvals)]
        
        for indexes, value in zip(indices1, values):
            x, y = indexes
            
            xposition = translate[x]
            yposition = translate[y]
            
            # We're going to key the randomization to the x, y
            # values for each point, insofar as that's possible.
            # Doing this will ensure that specific horizontal and
            # vertical lines preserve the dependence relations in
            # the original matrix.
            
            # But the way we're doing this is to use the permuted
            # x (or y) values to select an index in our list of
            # values in the present diagonal, and that's only possible
            # if the list is long enough to permit it. So we check:
            
            if xposition < 0 and yposition < 0:
                position = random.choice(vallist)
            elif xposition >= lenvals and yposition >= lenvals:
                position = random.choice(vallist)
            elif xposition < 0:
                position = yposition
            elif yposition < 0:
                position = xposition
            elif xposition >= lenvals:
                position = yposition
            elif yposition >= lenvals:
                position = xposition
            else:
                position = random.choice([xposition, yposition])
                # If either x or y could be used as an index, we
                # select randomly.
            
            # Whatever index was chosen, we use it to select a value
            # from our diagonal. 
            
            newvalues.append(values[position])
            
        values = newvalues
        
        # Now we lay down (both versions of) the diagonal in the
        # new matrix.
        
        for idxtuple1, idxtuple2, value in zip(indices1, indices2, values):
            x, y = idxtuple1
            newmat[x, y] = value
            x, y = idxtuple2
            newmat[x, y] = value
    
    return newmat

def zeroless(sequence):
    newseq = []
    for element in sequence:
        if element > 0.01:
            newseq.append(element)
    return newseq

def permute_test(distmatrix, foote_size=FOOTE_W, num_runs=100):
    actual_novelties = foote_novelty(distmatrix, foote_size)    
    permuted_peaks = []
    permuted_troughs = []
    xlen,ylen=distmatrix.shape
    for i in range(num_runs):
        randdist = diagonal_permute(distmatrix)
        nov = foote_novelty(randdist, foote_size)
        nov = zeroless(nov)
        permuted_peaks.append(np.max(nov))
        permuted_troughs.append(np.min(nov))
    permuted_peaks.sort(reverse = True)
    permuted_troughs.sort(reverse = True)
    significance_peak = np.ones(len(actual_novelties))
    significance_trough = np.ones(len(actual_novelties))
    for idx, novelty in enumerate(actual_novelties):
        ptop=[i for i,x in enumerate(permuted_peaks) if x and x < novelty]
        ptop=ptop[0]/num_runs if ptop else 1
        pbot=[i for i,x in enumerate(permuted_troughs) if x and x > novelty]
        pbot=pbot[-1]/num_runs if pbot else 1
        significance_peak[idx]=ptop
        significance_trough[idx]=pbot
        
        
    
    return actual_novelties, significance_peak, significance_trough

def colored_segments(novelties, significance, yrwidth=1,min_year=1700):
    x = []
    y = []
    t = []
    idx = 0
    for nov, sig in zip(novelties, significance):
        if nov > 1:
            x.append((idx*yrwidth) + min_year)
            y.append(nov)
            t.append(sig)
        idx += 1
        
    x = np.array(x)
    y = np.array(y)
    t = np.array(t)
    
    points = np.array([x,y]).transpose().reshape(-1,1,2)
    segs = np.concatenate([points[:-1],points[1:]],axis=1)
    lc = LineCollection(segs, cmap=plt.get_cmap('jet'))
    lc.set_array(t)
    
    return lc, x, y
    
    
def test_novelty(distdf, foote_sizes=None, num_runs=100):
    if not foote_sizes: foote_sizes=range(FOOTE_W-3, FOOTE_W+2)
    dq=distdf.fillna(0).values
    o=[]
    for fs in foote_sizes:
        try:
            novelties, significance_peak, significance_trough = permute_test(dq, foote_size=fs, num_runs=num_runs)
        except ValueError as e:
            # print('!!',e,'!!')
            continue
        for year,nov,sigp,sigt in zip(distdf.columns, novelties, significance_peak, significance_trough):
            odx={
                'period':year,
                'foote_novelty':nov,
                'foote_size':fs,
                'p_peak':sigp,
                'p_trough':sigt,
            }
            o.append(odx)
    return pd.DataFrame(o)









#########
