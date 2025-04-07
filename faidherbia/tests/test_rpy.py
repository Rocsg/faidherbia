# These will let us use R packages:
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
import rpy2.robjects.packages as rpackages
from rpy2.robjects.vectors import StrVector

pandas2ri.activate()
#install.packages("glcm")
#"glcm = rpackages.importr("glcm")
# Install the "glcm" package
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)
packnames = StrVector(['glcm'])
utils.install_packages(packnames)