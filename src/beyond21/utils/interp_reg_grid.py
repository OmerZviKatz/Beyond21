import numpy as np

class reg_grid_interp:
    # Regular (equal spaced) grid interpolation for 1D, 2D and 3D grids.
    
    def __init__(self,grid,xarr,yarr = None,zarr = None, zero_out_of_bounds = False ):
        self.grid = grid
        self.xarr = xarr
        self.yarr = yarr
        self.zarr = zarr
        self.zero_out_of_bounds = zero_out_of_bounds
    
        self.dx = (xarr[1]-xarr[0])
        self.inv_dx = 1/(xarr[1]-xarr[0])
        self.xmin = xarr[0]
        self.xmax = xarr[-1]
        if self.yarr is not None:
            self.dy = (yarr[1]-yarr[0])
            self.inv_dy = 1/(yarr[1]-yarr[0])
            self.ymin = yarr[0]
            self.ymax = yarr[-1]
        if self.zarr is not None:
            self.dz = (zarr[1]-zarr[0])
            self.inv_dz = 1/(zarr[1]-zarr[0])
            self.zmin = zarr[0]
            self.zmax = zarr[-1]


    def interp2D_single(self, point):
        x, y = point

        xmin = self.xmin; xmax = self.xmax
        ymin = self.ymin; ymax = self.ymax
        inv_dx = self.inv_dx; inv_dy = self.inv_dy
        dx = self.dx; dy = self.dy
        grid = self.grid

        if x <= xmin or x >= xmax or y <= ymin or y >= ymax:
            raise ValueError("Extrapolate = False and point is out of bound")

        ix = int((x - xmin) * inv_dx)  # Index ix such that xarr[ix] <= x < xarr[ix+1]
        iy = int((y - ymin) * inv_dy) # Index iy such that yarr[iy] <= y < yarr[iy+1]

        tx = (x - (xmin + ix * dx)) * inv_dx # (x-xarr[ix])/dx
        ty = (y - (ymin + iy * dy)) * inv_dy # (y-yarr[iy])/dy

        g00 = grid[ix,   iy]
        g10 = grid[ix+1, iy]
        g01 = grid[ix,   iy+1]
        g11 = grid[ix+1, iy+1]

        return g00 + tx*(g10 - g00) + ty*(g01 - g00) + tx*ty*(g11 - g10 - g01 + g00)
        


    def interp2D(self, points):
        # points must be a 2d numpy array

        xmin = self.xmin; xmax = self.xmax
        ymin = self.ymin; ymax = self.ymax
        inv_dx = self.inv_dx; inv_dy = self.inv_dy
        dx = self.dx; dy = self.dy
        grid = self.grid

        x, y = points[:, 0], points[:, 1]
        
        if (x.min() < xmin) or (x.max() > xmax) or (y.min() < ymin) or (y.max() > ymax):
            raise ValueError("Extrapolate = False and points are out of bounds")

        ix = ((x - xmin) * inv_dx).astype(int) # Index ix such that xarr[ix] <= x < xarr[ix+1]
        iy = ((y - ymin) * inv_dy).astype(int) # Index iy such that yarr[iy] <= y < yarr[iy+1]

        tx = (x - (xmin + ix * dx)) * inv_dx # (x-xarr[ix])/dx
        ty = (y - (ymin + iy * dy)) * inv_dy # (y-yarr[iy])/dy

        g00 = grid[ix,   iy]
        g10 = grid[ix+1, iy]
        g01 = grid[ix,   iy+1]
        g11 = grid[ix+1, iy+1]

        return g00 + tx*(g10 - g00) + ty*(g01 - g00) + tx*ty*(g11 - g10 - g01 + g00)

        




    



    

    
        
    
    

    


        


    