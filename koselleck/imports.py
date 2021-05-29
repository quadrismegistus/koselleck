import os,sys
here=os.path.abspath(__file__)
root=os.path.dirname(os.path.abspath(os.path.join(here,'..')))
PATH_LIB=os.path.join(root,'lib')
sys.path.append(os.path.join(PATH_LIB,'Noise-Aware-Alignment'))
sys.path.append(os.path.join(PATH_LIB,'abslithist'))
sys.path.append('/home/ryan/github/yapmap')
from abslithist import *

here=os.path.abspath(__file__)
root=os.path.dirname(os.path.abspath(os.path.join(here,'..')))
PATH_DATA=os.path.join(root,'data')
PATH_FIGS=os.path.join(root,'figures')
OFN_LINREG=os.path.join(PATH_DATA,'data.becoming_abs.linreg.csv')
CNAME='BPO'
OFN_MEAN_DIFF=os.path.join(PATH_DATA,'data.becoming_abs.mean_diff.csv')
WORD_FN=os.path.join(PATH_DATA,'words.txt')
PATH_SKIPGRAMS=os.path.join(PATH_DATA,'skipgrams')
PATH_MODELS_NEW=os.path.join(PATH_DATA,'models')
PATH_MODELS_OLD='/home/ryan/DH/data/models'
PATH_SKIPGRAMS_YR=os.path.join(PATH_SKIPGRAMS,'years')
PATH_FIELDS=os.path.join(PATH_DATA,'fields')
PATH_FIELD_WILLIAMS_SRC=os.path.join(PATH_FIELDS,'williams-src.txt')
PATH_FIELD_WILLIAMS=os.path.join(PATH_FIELDS,'williams.txt')
PATH_FIELD_KOSELLECK_SRC=os.path.join(PATH_FIELDS,'koselleck-src.txt')
PATH_FIELD_KOSELLECK=os.path.join(PATH_FIELDS,'koselleck.txt')
PATH_MODELS_BPO=os.path.join(PATH_MODELS_NEW,'bpo')
FN_CHANGE_RUNS_AVG = os.path.join(PATH_DATA,'data.measured_change.runs_avg.v2.csv')
FN_CHANGE_RUNS = os.path.join(PATH_DATA,'data.measured_change.runs.v2.csv')
FN_DATA_CACHE_DEC=os.path.join(PATH_DATA,'data.cache.decade_level_data.pkl')
FN_VECTOR_SCORES_RUNS=os.path.join(PATH_DATA,'data.vector_scores_across_models.pkl')
FN_VECTOR_SCORES_DIFFMEANS=os.path.join(PATH_DATA,'data.vector_scores_across_models.diff_means.csv')
FN_FREQ_DEC_MODELS=os.path.join(PATH_DATA,'data.freq_across_decade_models.csv')
# FN_DATA_PACEOFCHANGE = os.path.join(PATH_DATA,'data.semantic_change_over_decades.1run.v5-halfdec.pkl')
# FN_DATA_PACEOFCHANGE = os.path.join(PATH_DATA,'data.semantic_change_over_decades.1run.v9-local-halfdec.pkl')
FN_DATA_PACEOFCHANGE = os.path.join(PATH_DATA,'data.semantic_change_over_decades.1run.v10-local-k50-halfdec.pkl')
FIELD_ABS_KEY='Abs-Conc.Median'
FN_AMBIGUITY=os.path.join(PATH_DATA,'data.ambiguity.runs.csv')
URL_KEYWORDS='https://docs.google.com/spreadsheets/d/e/2PACX-1vRzHA7iqgW7BB9SCtR0Nr3Dge5zSkY9C6lOkUMFV7Bd4Bhap6LVR3sWrXnjovUNhL9HAUNUJNRB62rD/pub?gid=0&single=true&output=csv'
DEFAULT_NUM_SKIP=20000
FOOTE_W=5
# FN_NOVELTY_DATA=os.path.join(PATH_DATA,'data.words_by_rateofchange.pkl')
FN_NOVELTY_DATA=os.path.join(PATH_DATA,'data.words_by_rateofchange.v4.pkl')


import os,sys,json,pickle
import networkx as nx
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0,'../yapmap')
import pandas as pd
import numpy as np
# from abslithist.embeddings import *
import lltk
from fastdist import fastdist
from tqdm import tqdm
tqdm.pandas()
from scipy.stats import ttest_ind
try:
    from ipywidgets import interact, interactive, fixed, interact_manual
    import ipywidgets as widgets
except Exception:
    pass
disable_gensim_logging()
from noise_aware import noise_aware
from ftfy import fix_text
import cv2

from gensim.models import KeyedVectors,Word2Vec#,FastText
from loguru import logger
from scipy.spatial.distance import cosine
from scipy.stats import percentileofscore
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from IPython.display import Image

from .tools import *
from .embeddings import *
from .plots import *
from .novelty import *