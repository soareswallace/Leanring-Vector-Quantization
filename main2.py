from scipy.io import arff
import pandas as pd
data = arff.loadarff('kc1.arff')
df = pd.DataFrame(data[0])
