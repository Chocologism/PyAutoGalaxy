from autoarray.mask import mask_2d
from autoarray.structures.arrays.one_d import array_1d
from autoarray.structures.arrays.two_d import array_2d
from autoarray.structures.grids.one_d import grid_1d
from autoarray.structures.grids.two_d import grid_2d
from autoarray.structures.grids.two_d.grid_2d_irregular import Grid2DIrregular
from autoarray.structures.vector_fields import vector_field_irregular
from autoarray.plot.mat_wrap import visuals as vis

from matplotlib import patches as ptch
import typing
from typing import List, Union, Optional
import numpy as np


class Visuals1D(vis.Visuals1D):
    def __init__(
        self,
        half_light_radius: Optional[float] = None,
        half_light_radius_errors: Optional[List[float]] = None,
        einstein_radius: Optional[float] = None,
        model_fluxes: Optional[grid_1d.Grid1D] = None,
        shaded_region: Optional[
            Union[List[List], List[array_1d.Array1D], List[np.ndarray]]
        ] = None,
    ):

        super().__init__(shaded_region=shaded_region)

        self.half_light_radius = half_light_radius
        self.half_light_radius_errors = half_light_radius_errors
        self.einstein_radius = einstein_radius
        self.model_fluxes = model_fluxes

    def plot_via_plotter(self, plotter, grid_indexes=None, mapper=None):

        super().plot_via_plotter(plotter=plotter)

        if self.half_light_radius is not None:
            plotter.half_light_radius_axvline.axvline_vertical_line(
                vertical_line=self.half_light_radius,
                vertical_errors=self.half_light_radius_errors,
                label="Half-light Radius",
            )

        if self.einstein_radius is not None:
            plotter.einstein_radius_axvline.axvline_vertical_line(
                vertical_line=self.einstein_radius, label="Einstein Radius"
            )

        if self.model_fluxes is not None:

            plotter.model_fluxes_yx_scatter.scatter_yx(
                y=self.model_fluxes, x=np.arange(len(self.model_fluxes))
            )


class Visuals2D(vis.Visuals2D):
    def __init__(
        self,
        origin: grid_2d.Grid2D = None,
        border: grid_2d.Grid2D = None,
        mask: mask_2d.Mask2D = None,
        positions: Optional[Union[Grid2DIrregular, List[Grid2DIrregular]]] = None,
        grid: grid_2d.Grid2D = None,
        pixelization_grid: grid_2d.Grid2D = None,
        vector_field: vector_field_irregular.VectorField2DIrregular = None,
        patches: typing.Union[ptch.Patch] = None,
        array_overlay: array_2d.Array2D = None,
        light_profile_centres: Grid2DIrregular = None,
        mass_profile_centres: Grid2DIrregular = None,
        multiple_images: Grid2DIrregular = None,
        critical_curves: Optional[Union[Grid2DIrregular, List[Grid2DIrregular]]] = None,
        caustics: Optional[Union[Grid2DIrregular, List[Grid2DIrregular]]] = None,
        parallel_overscan=None,
        serial_prescan=None,
        serial_overscan=None,
        indexes: typing.Union[List[int], List[List[int]]] = None,
        pixelization_indexes: typing.Union[List[int], List[List[int]]] = None,
    ):

        super().__init__(
            mask=mask,
            positions=positions,
            grid=grid,
            pixelization_grid=pixelization_grid,
            vector_field=vector_field,
            patches=patches,
            array_overlay=array_overlay,
            origin=origin,
            border=border,
            parallel_overscan=parallel_overscan,
            serial_prescan=serial_prescan,
            serial_overscan=serial_overscan,
            indexes=indexes,
            pixelization_indexes=pixelization_indexes,
        )

        self.light_profile_centres = light_profile_centres
        self.mass_profile_centres = mass_profile_centres
        self.multiple_images = multiple_images
        self.critical_curves = critical_curves
        self.caustics = caustics

    def plot_via_plotter(self, plotter, grid_indexes=None, mapper=None):

        super().plot_via_plotter(
            plotter=plotter, grid_indexes=grid_indexes, mapper=mapper
        )

        if self.light_profile_centres is not None:
            plotter.light_profile_centres_scatter.scatter_grid(
                grid=self.light_profile_centres
            )

        if self.mass_profile_centres is not None:
            plotter.mass_profile_centres_scatter.scatter_grid(
                grid=self.mass_profile_centres
            )

        if self.multiple_images is not None:
            plotter.multiple_images_scatter.scatter_grid(grid=self.multiple_images)

        if self.critical_curves is not None:
            try:
                plotter.critical_curves_plot.plot_grid(grid=self.critical_curves)
            except TypeError:
                pass

        if self.caustics is not None:
            try:
                plotter.caustics_plot.plot_grid(grid=self.caustics)
            except TypeError:
                pass
