# Interpolation for sorted grid (ascending)
import numpy as np
from numba import jit

#@jit(nopython=True)    
def interp3D_sorted_single_calc(point,xarr,yarr,zarr,grid):

    x_point,y_point,z_point = point
    
    x_below_index = np.searchsorted(xarr,x_point)-1
    y_below_index = np.searchsorted(yarr,y_point)-1 
    z_below_index = np.searchsorted(zarr,z_point)-1 

    x_below = xarr[x_below_index]
    y_below = yarr[y_below_index]
    z_below = zarr[z_below_index]
    
    x_above = xarr[x_below_index+1]
    y_above = yarr[y_below_index+1]
    z_above = zarr[z_below_index+1]
    
    xd = (x_point - x_below)/(x_above-x_below)
    yd = (y_point - y_below)/(y_above-y_below)
    zd = (z_point - z_below)/(z_above-z_below)
    
    grid_bbb = grid[x_below_index][y_below_index][z_below_index]  #b below a above
    grid_abb = grid[x_below_index+1][y_below_index][z_below_index] 
    grid_bba = grid[x_below_index][y_below_index][z_below_index+1] 
    grid_aba = grid[x_below_index+1][y_below_index][z_below_index+1] 
    grid_bab = grid[x_below_index][y_below_index+1][z_below_index] 
    grid_aab = grid[x_below_index+1][y_below_index+1][z_below_index] 
    grid_baa = grid[x_below_index][y_below_index+1][z_below_index+1] 
    grid_aaa = grid[x_below_index+1][y_below_index+1][z_below_index+1] 
    
    #Interpolate along x
    grid_00 = grid_bbb*(1-xd)+grid_abb*xd
    grid_01 = grid_bba*(1-xd)+grid_aba*xd
    grid_10 = grid_bab*(1-xd)+grid_aab*xd
    grid_11 = grid_baa*(1-xd)+grid_aaa*xd
    
    #Interpolate along y
    grid_0 = grid_00*(1-yd)+grid_10*yd
    grid_1 = grid_01*(1-yd)+grid_11*yd

    #Interpolate along z
    return grid_0 * (1 - zd) + grid_1 * zd

#@jit(nopython=True)    
def QuickSearch(tarr,tval):
    t_below_index = np.searchsorted(tarr,tval)-1 
    return t_below_index


class reg_grid_interp:
    
    def __init__(self,grid,xarr,yarr = None,zarr = None, zero_out_of_bounds = False ):
        self.grid = grid
        self.xarr = xarr
        self.yarr = yarr
        self.zarr = zarr
        self.zero_out_of_bounds = zero_out_of_bounds
        
    def interp3D_sorted(self, points):
        if not isinstance(points, np.ndarray):
            points = np.array([points])
    
        x_points, y_points, z_points = points[:, 0], points[:, 1], points[:, 2]
    
        out_of_bounds = np.any(
            (x_points > self.xarr[-1]) | (x_points < self.xarr[0]) |
            (y_points > self.yarr[-1]) | (y_points < self.yarr[0]) |
            (z_points > self.zarr[-1]) | (z_points < self.zarr[0])
        )
        
        if out_of_bounds:
            raise ValueError("Extrapolate = False and points are out of bounds")
    
        x_below_indices = np.searchsorted(self.xarr, x_points) - 1
        y_below_indices = np.searchsorted(self.yarr, y_points) - 1
        z_below_indices = np.searchsorted(self.zarr, z_points) - 1
    
        x_below = self.xarr[x_below_indices]
        y_below = self.yarr[y_below_indices]
        z_below = self.zarr[z_below_indices]
    
        x_above = self.xarr[x_below_indices + 1]
        y_above = self.yarr[y_below_indices + 1]
        z_above = self.zarr[z_below_indices + 1]
    
        xd = (x_points - x_below) / (x_above - x_below)
        yd = (y_points - y_below) / (y_above - y_below)
        zd = (z_points - z_below) / (z_above - z_below)
    
        # Interpolate along x
        grid_00 = self.grid[x_below_indices, y_below_indices, z_below_indices] * (1 - xd) + self.grid[x_below_indices + 1, y_below_indices, z_below_indices] * xd
        grid_01 = self.grid[x_below_indices, y_below_indices, z_below_indices + 1] * (1 - xd) + self.grid[x_below_indices + 1, y_below_indices, z_below_indices + 1] * xd
        grid_10 = self.grid[x_below_indices, y_below_indices + 1, z_below_indices] * (1 - xd) + self.grid[x_below_indices + 1, y_below_indices + 1, z_below_indices] * xd
        grid_11 = self.grid[x_below_indices, y_below_indices + 1, z_below_indices + 1] * (1 - xd) + self.grid[x_below_indices + 1, y_below_indices + 1, z_below_indices + 1] * xd
    
        # Interpolate along y
        grid_0 = grid_00 * (1 - yd) + grid_10 * yd
        grid_1 = grid_01 * (1 - yd) + grid_11 * yd
    
        # Interpolate along z
        return [grid_0 * (1 - zd) + grid_1 * zd]



    def interp3D_sorted_single(self,point):
        x_point,y_point,z_point = point
        
        if x_point>self.xarr[-1] or x_point<self.xarr[0] or y_point>self.yarr[-1] or y_point<self.yarr[0] or z_point>self.zarr[-1] or z_point<self.zarr[0]:
            raise ValueError("Extrapolate = False and point is out of bound")
            #return np.nan
            
        return [interp3D_sorted_single_calc(point,self.xarr,self.yarr,self.zarr,self.grid)]
            
        

    def interp2D_sorted(self, points):

        if not isinstance(points, np.ndarray):
            # If points is not an ndarray, convert it to a 2D ndarray
            points = np.array([points])

        x_points, y_points = points[:, 0], points[:, 1]
        
        if np.any(x_points > self.xarr[-1]) or np.any(x_points < self.xarr[0]) \
                or np.any(y_points > self.yarr[-1]) or np.any(y_points < self.yarr[0]):
            raise ValueError("Extrapolate = False and points are out of bounds")

        x_below_indices = np.searchsorted(self.xarr, x_points) - 1
        y_below_indices = np.searchsorted(self.yarr, y_points) - 1

        x_below = self.xarr[x_below_indices]
        y_below = self.yarr[y_below_indices]
        x_above = self.xarr[x_below_indices + 1]
        y_above = self.yarr[y_below_indices + 1]

        xd = (x_points - x_below) / (x_above - x_below)
        yd = (y_points - y_below) / (y_above - y_below)

        grid_bb = self.grid[x_below_indices, y_below_indices]
        grid_ab = self.grid[x_below_indices + 1, y_below_indices]
        grid_ba = self.grid[x_below_indices, y_below_indices + 1]
        grid_aa = self.grid[x_below_indices + 1, y_below_indices + 1]

        # Interpolate along x
        grid_0 = grid_bb * (1 - xd) + grid_ab * xd
        grid_1 = grid_ba * (1 - xd) + grid_aa * xd

        # Interpolate along z
        interpolated_values = grid_0 * (1 - yd) + grid_1 * yd

        return interpolated_values

    def interp2D_sorted_single(self,point):
        x_point,y_point = point
    
        if x_point>self.xarr[-1] or x_point<self.xarr[0] or y_point>self.yarr[-1] or y_point<self.yarr[0]:
            raise ValueError("Extrapolate = False and point is out of bound")

        x_below_index = QuickSearch(self.xarr,x_point)
        y_below_index = QuickSearch(self.yarr,y_point)
        x_below = self.xarr[x_below_index]
        y_below = self.yarr[y_below_index]
        x_above = self.xarr[x_below_index+1]
        y_above = self.yarr[y_below_index+1]
    
        xd = (x_point - x_below)/(x_above-x_below)
        yd = (y_point - y_below)/(y_above-y_below)
    
        grid_bb = self.grid[x_below_index][y_below_index]
        grid_ab = self.grid[x_below_index+1][y_below_index]
        grid_ba = self.grid[x_below_index][y_below_index+1]
        grid_aa = self.grid[x_below_index+1][y_below_index+1]
    
        #Interpolate along x
        grid_0 = grid_bb*(1-xd)+grid_ab*xd
        grid_1 = grid_ba*(1-xd)+grid_aa*xd
    
        #Interpolate along z
        return [grid_0*(1-yd)+grid_1*yd]


    def interp1D_sorted(self, x_points):
        x_points = np.atleast_1d(x_points) 
    
        # Create mask for points that are in bounds
        in_bounds = (x_points >= self.xarr[0]) & (x_points <= self.xarr[-1])
    
        if not self.zero_out_of_bounds and not np.all(in_bounds):
            raise ValueError("Extrapolate = False and some points are out of bounds")
    
        # Initialize output with zeros (will be filled where in_bounds is True)
        interpolated_values = np.zeros_like(x_points, dtype=float)
    
        if np.any(in_bounds):
            x_valid = x_points[in_bounds]
    
            x_below_indices = np.searchsorted(self.xarr, x_valid) - 1
            # Clip to ensure indices stay in valid range
            #x_below_indices = np.clip(x_below_indices, 0, len(self.xarr) - 2)
    
            x_below = self.xarr[x_below_indices]
            x_above = self.xarr[x_below_indices + 1]
    
            xd = (x_valid - x_below) / (x_above - x_below)
            grid_b = self.grid[x_below_indices]
            grid_a = self.grid[x_below_indices + 1]
    
            interpolated_values[in_bounds] = grid_b + (grid_a - grid_b) * xd
    
        return interpolated_values

            
        
    
    

    


        


    