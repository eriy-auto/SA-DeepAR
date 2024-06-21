import math
from functools import partial
import logging
from typing import Optional, Union

import numpy as np
from numpy.random import normal

from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks

from model.loss import gaussian_likelihood
from model import NNModel
from model.layers import GaussianLayer