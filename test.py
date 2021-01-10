import pandas as pd
import numpy as np
import random as rd
import math as mt
from pathlib import Path
from datetime import datetime as dt

path = Path(__file__).parent.absolute()/'Learning'/'Deep Training'
for model in (path/'Models').glob('*.h5'):
    print(model)