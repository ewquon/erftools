import numpy as np

def write_binary_simple_ERF(output_binary,
                            lat_erf, lon_erf,
                            x_grid, y_grid, z_grid,
                            point_data):

    x_grid = np.asarray(x_grid)
    y_grid = np.asarray(y_grid)

    # Ensure grids are consistent
    nrow, ncol = x_grid.shape
    nz = len(z_grid[0,0,:])

    #print(nrow, ncol)
    #print(x_grid[0,:])

    # TODO: rewrite without nested loops for efficiency
    with open(output_binary, "wb") as file:
        file.write(struct.pack('iiii', ncol, nrow, nz, len(point_data)))

        for j in range(nrow):  # Iterate over the y-dimension
            for i in range(ncol):  # Iterate over the x-dimension
                file.write(struct.pack('f', lat_erf[i,j,0]))

        for j in range(nrow):  # Iterate over the y-dimension
            for i in range(ncol):  # Iterate over the x-dimension
                file.write(struct.pack('f', lon_erf[i,j,0]))

       # Write grid points using a nested for loop
        for i in range(ncol):
            x = x_grid[0, i]
            file.write(struct.pack('f', x))

        for j in range(nrow):
            y = y_grid[j, 0]
            file.write(struct.pack('f', y))

        for k in range(nz):
            zavg = np.mean(z_grid[:,:,k])
            file.write(struct.pack('f', zavg))

        # Write point data (if any)
        if point_data:
            for name, data in point_data.items():
                for k in range(nz):  # Iterate over the z-dimension
                    for j in range(nrow):  # Iterate over the y-dimension
                        for i in range(ncol):  # Iterate over the x-dimension
                            value = data[i, j, k]
                            file.write(struct.pack('f', value))
