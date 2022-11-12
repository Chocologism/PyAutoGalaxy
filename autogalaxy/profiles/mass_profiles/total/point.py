import copy
import numpy as np
from scipy.integrate import quad
from scipy import special
from typing import Tuple

import autoarray as aa

from autogalaxy.profiles.abstract.abstract import MassProfile

class PointMass(MassProfile):
    def __init__(
        self, centre: Tuple[float, float] = (0.0, 0.0), einstein_radius: float = 1.0
    ):
        """
        Represents a point-mass.

        Parameters
        ----------
        centre
            The (y,x) arc-second coordinates of the profile centre.
        einstein_radius
            The arc-second Einstein radius of the point-mass.
        """
        super().__init__(centre=centre, elliptical_comps=(0.0, 0.0))
        self.einstein_radius = einstein_radius

    def convergence_2d_from(self, grid: aa.type.Grid2DLike):

        squared_distances = np.square(grid[:, 0] - self.centre[0]) + np.square(
            grid[:, 1] - self.centre[1]
        )
        central_pixel = np.argmin(squared_distances)

        convergence = np.zeros(shape=grid.shape[0])
        #    convergence[central_pixel] = np.pi * self.einstein_radius ** 2.0
        return convergence

    @aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        return np.zeros(shape=grid.shape[0])

    @aa.grid_dec.grid_2d_to_vector_yx
    @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        grid_radii = self.grid_to_grid_radii(grid=grid)
        return self.grid_to_grid_cartesian(
            grid=grid, radius=self.einstein_radius**2 / grid_radii
        )

    @property
    def is_point_mass(self):
        return True

    def with_new_normalization(self, normalization):

        mass_profile = copy.copy(self)
        mass_profile.einstein_radius = normalization
        return mass_profile