# CONTRIBUTORS: * Ericsson Chenebuah, Michel Nganbe and Alain Tchagang 
# Department of Mechanical Engineering, University of Ottawa, 75 Laurier Ave. East, Ottawa, ON, K1N 6N5 Canada
# Digital Technologies Research Centre, National Research Council of Canada, 1200 Montréal Road, Ottawa, ON, K1A 0R6 Canada
# * email: echen013@uottawa.ca 
# (December-2023)

## THIS SOURCE CODE ILLUSTRATES HOW TO COMPUTE AND ENCODE XRD PATTERNS FOR FEATURE ENGINEERING THE XRD MESH ARRAY OF THE INVERTIBLE MESH-GRID DESCRIPTOR.

# Please note that code must be executed alongside all relevant spreadsheet data in the same file directory.

## References
# Ong, S.P, et al. (2013). Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis, Comput. Mater. Sci., 68, 314–319.
# De Graef, M., and McHenry, M.E. (2012). Structure of materials: An introduction to crystallography, diffraction and symmetry. Cambridge University Press.


from pymatgen.ext.matproj import MPRester #(Legacy API)
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

mp_data = pd.read_excel("mp_data.xlsx")
xrd_data = []

for i in range(mp_data.shape[0]):

  with MPRester(api_key='dHLnBAxZxH4WOWDgL') as mpr:
    struc = mpr.get_structure_by_material_id(mp_data.iloc[i,3])

  sga = SpacegroupAnalyzer(struc)
  conv_struc = sga.get_conventional_standard_structure()
  calc = XRDCalculator(wavelength='CuKa')
  pattern = calc.get_pattern(conv_struc, scaled=True, two_theta_range=(0, 180))

  scaler_x = MinMaxScaler(feature_range=(0, 0.9))
  x=scaler_x.fit_transform(pattern.x.reshape(pattern.x.shape[0],1))

  scaler_y = MinMaxScaler(feature_range=(0, 0.9))
  y=scaler_y.fit_transform(pattern.y.reshape(pattern.y.shape[0],1))

  arr=np.column_stack((x,y))

  grid = 32

  arr = (np.round(arr * 31, 0)).astype(int)
  grid_mesh = np.zeros((grid,grid), dtype=int)

  grid_mesh[arr[:, 0], arr[:, 1]] = 1

  xrd_data.append(grid_mesh)

xrd_mesh = np.array(xrd_data)
xrd_mesh = xrd_mesh.reshape(xrd_mesh.shape[0],grid,grid,1)
