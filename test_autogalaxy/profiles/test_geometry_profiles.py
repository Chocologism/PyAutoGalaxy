from __future__ import division, print_function

from os import path

import numpy as np
import pytest

import autogalaxy as ag
from autogalaxy.profiles import geometry_profiles

directory = path.dirname(path.realpath(__file__))


def test__cos_and_sin_from_x():

    elliptical_profile = geometry_profiles.EllProfile(
        centre=(1.0, 1.0), ell_comps=(0.0, 0.0)
    )

    cos_phi, sin_phi = elliptical_profile._cos_and_sin_to_x_axis()

    assert cos_phi == 1.0
    assert sin_phi == 0.0

    # axis_ratio=0.1, angle=45.0

    elliptical_profile = geometry_profiles.EllProfile(
        centre=(1, 1), ell_comps=(0.8181, 0.0)
    )

    cos_phi, sin_phi = elliptical_profile._cos_and_sin_to_x_axis()

    assert cos_phi == pytest.approx(0.707, 1e-3)
    assert sin_phi == pytest.approx(0.707, 1e-3)

    # axis_ratio=0.1, angle=60.0

    elliptical_profile = geometry_profiles.EllProfile(
        centre=(1, 1), ell_comps=(0.70856, -0.4090909)
    )

    cos_phi, sin_phi = elliptical_profile._cos_and_sin_to_x_axis()

    assert cos_phi == pytest.approx(0.5, 1e-3)
    assert sin_phi == pytest.approx(0.866, 1e-3)


def test__grid_to_eccentric_radii():

    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.0))

    eccentric_radius = elliptical_profile.grid_to_eccentric_radii(
        grid=np.array([[1.0, 1.0]])
    )

    assert eccentric_radius == pytest.approx(2.0**0.5, 1e-3)

    eccentric_radius = elliptical_profile.grid_to_eccentric_radii(
        grid=np.array([[1.0, 1.0]])
    )

    assert eccentric_radius == pytest.approx(2.0**0.5, 1e-3)

    # eccentric_radius = sqrt(axis_ratio) * sqrt(  x**2 +   (y**2 / axis_ratio**2))
    # eccentric_radius =        sqrt(0.5) * sqrt(1.0**2 + (1.0**2 /        0.5**2))
    # eccentric radius =        sqrt(0.5) * sqrt( 5 ) = 1.58113

    # axis_ratio=0.5, angle=0.0

    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.333333))

    eccentric_radius = elliptical_profile.grid_to_eccentric_radii(
        np.array([[1.0, 1.0]])
    )

    assert eccentric_radius == pytest.approx(1.58113, 1e-3)


def test__spherical__transform_to_reference_frame__coordinate_shifts_using_centre():

    spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

    transformed_grid = (
        spherical_profile.grid_transformed_from_profile_reference_frame_from(
            np.array([[1.0, 1.0]])
        )
    )

    assert (transformed_grid == np.array([[1.0, 1.0]])).all()

    spherical_profile1 = geometry_profiles.SphProfile(centre=(0, 0))

    grid1 = spherical_profile1.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[1.0, 1.0]])
    )

    spherical_profile2 = geometry_profiles.SphProfile(centre=(-1, -1))
    grid2 = spherical_profile2.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[0.0, 0.0]])
    )

    assert (grid1 == grid2).all()

    spherical_profile1 = geometry_profiles.SphProfile(centre=(1, 1))
    grid1 = spherical_profile1.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[1.0, 1.0]])
    )

    spherical_profile2 = geometry_profiles.SphProfile(centre=(-1, -1))
    grid2 = spherical_profile2.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[-1.0, -1.0]])
    )

    assert (grid1 == grid2).all()


def test__spherical__transform_to_and_from_reference_frame():

    spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

    grid = np.array([[1.0, 1.0]])

    transformed_grid = (
        spherical_profile.grid_transformed_to_profile_reference_frame_from(grid)
    )

    assert transformed_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

    spherical_profile = geometry_profiles.SphProfile(centre=(0.0, 0.0))

    grid_original = np.array([[5.2221, 2.6565]])

    grid_spherical = spherical_profile.grid_transformed_to_profile_reference_frame_from(
        grid_original
    )

    transformed_grid = (
        spherical_profile.grid_transformed_from_profile_reference_frame_from(
            grid_spherical
        )
    )

    assert transformed_grid == pytest.approx(grid_original, 1e-5)


def test__elliptical__transform_grid_to_and_from_reference_frame():

    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.0, 0.0))

    transformed_grid = (
        elliptical_profile.grid_transformed_to_profile_reference_frame_from(
            grid=np.array([[1.0, 1.0]])
        )
    )

    assert transformed_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

    transformed_back_grid = (
        elliptical_profile.grid_transformed_from_profile_reference_frame_from(
            transformed_grid
        )
    )

    assert transformed_back_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

    transformed_back_grid = (
        elliptical_profile.grid_transformed_from_profile_reference_frame_from(
            transformed_grid
        )
    )

    assert transformed_back_grid == pytest.approx(np.array([[1.0, 1.0]]), 1e-3)

    # axis_ratio=0.1, angle=90.0

    elliptical_profile = geometry_profiles.EllProfile(
        centre=(2.0, 3.0), ell_comps=(0.0, -0.81818181)
    )

    transformed_grid = (
        elliptical_profile.grid_transformed_to_profile_reference_frame_from(
            grid=np.array([[3.0, 4.0]])
        )
    )

    assert transformed_grid == pytest.approx(np.array([[-1.0, 1.0]]), 1e-3)

    transformed_back_grid = (
        elliptical_profile.grid_transformed_from_profile_reference_frame_from(
            transformed_grid
        )
    )

    assert transformed_back_grid == pytest.approx(np.array([[3.0, 4.0]]), 1e-3)

    elliptical_profile = geometry_profiles.EllProfile(ell_comps=(0.818181, 0.0))

    grid_original = np.array([[5.2221, 2.6565]])

    grid_elliptical = (
        elliptical_profile.grid_transformed_to_profile_reference_frame_from(
            grid_original
        )
    )

    transformed_grid = (
        elliptical_profile.grid_transformed_from_profile_reference_frame_from(
            grid_elliptical
        )
    )

    assert transformed_grid == pytest.approx(grid_original, 1e-5)


def test__elliptical__transform_grids_with_mapped_centres():
    elliptical_profile1 = geometry_profiles.EllProfile(
        ell_comps=(0.0, 0.0), centre=(0, 0)
    )
    grid1 = elliptical_profile1.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[1.0, 1.0]])
    )

    elliptical_profile2 = geometry_profiles.EllProfile(
        ell_comps=(0.0, 0.0), centre=(-1, -1)
    )
    grid2 = elliptical_profile2.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[0.0, 0.0]])
    )

    assert (grid1 == grid2).all()

    # axis_ratio=0.1, angle=55.0

    elliptical_profile1 = geometry_profiles.EllProfile(
        centre=(0, 0), ell_comps=(0.76883, -0.27983)
    )
    grid1 = elliptical_profile1.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[1.0, 1.0]])
    )

    elliptical_profile2 = geometry_profiles.EllProfile(
        centre=(-1, -1), ell_comps=(0.76883, -0.27983)
    )
    grid2 = elliptical_profile2.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[0.0, 0.0]])
    )

    assert (grid1 == grid2).all()

    elliptical_profile1 = geometry_profiles.EllProfile(
        centre=(1, 1), ell_comps=(0.76883, -0.27983)
    )
    grid1 = elliptical_profile1.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[1.0, 1.0]])
    )

    elliptical_profile2 = geometry_profiles.EllProfile(
        centre=(-1, -1), ell_comps=(0.76883, -0.27983)
    )
    grid2 = elliptical_profile2.grid_transformed_to_profile_reference_frame_from(
        grid=np.array([[-1.0, -1.0]])
    )

    assert (grid1 == grid2).all()
