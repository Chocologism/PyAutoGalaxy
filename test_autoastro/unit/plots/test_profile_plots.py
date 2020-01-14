import autoarray as aa
import autoastro as aast
import pytest
import os
from os import path
from autoarray import conf

directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="profile_plotter_path")
def make_profile_plotter_setup():
    return "{}/../../../test_files/plotting/profiles/".format(
        os.path.dirname(os.path.realpath(__file__))
    )


@pytest.fixture(autouse=True)
def set_config_path():
    conf.instance = conf.Config(
        path.join(directory, "../test_files/plotters"), path.join(directory, "output")
    )


def test__all_quantities_are_output(
    lp_0, mp_0, sub_grid_7x7, mask_7x7, positions_7x7, include_all, profile_plotter_path, plot_patch
):

    aast.plot.profile.image(
        light_profile=lp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(profile_plotter_path, format="png")
        ),
    )

    assert profile_plotter_path + "image.png" in plot_patch.paths

    aast.plot.profile.convergence(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(profile_plotter_path, format="png")
        ),
    )

    assert profile_plotter_path + "convergence.png" in plot_patch.paths

    aast.plot.profile.potential(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(profile_plotter_path, format="png")
        ),
    )

    assert profile_plotter_path + "potential.png" in plot_patch.paths

    aast.plot.profile.deflections_y(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(profile_plotter_path, format="png")
        ),
    )

    assert profile_plotter_path + "deflections_y.png" in plot_patch.paths

    aast.plot.profile.deflections_x(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(profile_plotter_path, format="png")
        ),
    )

    assert profile_plotter_path + "deflections_x.png" in plot_patch.paths

    aast.plot.profile.magnification(
        mass_profile=mp_0,
        grid=sub_grid_7x7,
        mask=mask_7x7,
        positions=positions_7x7,
        include=include_all,
        plotter=aa.plotter.Plotter(
            output=aa.plotter.Output(profile_plotter_path, format="png")
        ),
    )

    assert profile_plotter_path + "magnification.png" in plot_patch.paths
