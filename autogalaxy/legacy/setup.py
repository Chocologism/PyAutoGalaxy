from typing import Optional

import autofit as af

from autogalaxy.legacy.hyper_data import HyperImageSky
from autogalaxy.legacy.hyper_data import HyperBackgroundNoise
from autogalaxy.analysis.setup import SetupAdapt as SetupAdaptBase


class SetupAdapt(SetupAdaptBase):
    def __init__(
        self,
        search_pix_cls: Optional[af.NonLinearSearch] = None,
        search_pix_dict: Optional[dict] = None,
        mesh_pixels_fixed: Optional[int] = None,
        hyper_galaxies: bool = False,
        hyper_image_sky: Optional[type(HyperImageSky)] = None,
        hyper_background_noise: Optional[type(HyperBackgroundNoise)] = None,
        search_noise_cls: Optional[af.NonLinearSearch] = None,
        search_noise_dict: Optional[dict] = None,
    ):
        """
        The adapt setup of a pipeline, which controls how adaptive-features in PyAutoGalaxy template pipelines run,
        for example controlling whether galaxies are used to scale the noise and the non-linear searches used
        in these searchs.

        Users can write their own pipelines which do not use or require the *SetupAdapt* class.

        Parameters
        ----------
        search_pix_cls
            The non-linear search used by every adapt model-fit search.
        search_pix_dict
            The dictionary of search options for the hyper inversion model-fit searches.
        hyper_galaxies
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used to scale the
            noise-map of the dataset throughout the fitting.
        hyper_image_sky
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            image's background sky component in the model.
        hyper_background_noise
            If a hyper-pipeline is being used, this determines if hyper-galaxy functionality is used include the
            noise-map's background component in the model.
        search_noise_dict
            The dictionary of search options for the hyper noise model-fit searches.
        """

        super().__init__(
            search_pix_cls=search_pix_cls,
            search_pix_dict=search_pix_dict,
            mesh_pixels_fixed=mesh_pixels_fixed,
        )

        self.hyper_galaxies = hyper_galaxies

        self.hyper_galaxy_names = None

        self.search_pix_cls = search_pix_cls or af.DynestyStatic
        self.search_pix_dict = search_pix_dict or {
            "nlive": 50,
            "sample": "rwalk",
        }

        self.search_noise_cls = search_noise_cls or af.DynestyStatic
        self.search_noise_dict = search_noise_dict or {"nlive": 50, "sample": "rwalk"}

        self.hyper_image_sky = hyper_image_sky
        self.hyper_background_noise = hyper_background_noise

        self.mesh_pixels_fixed = mesh_pixels_fixed

    @property
    def hypers_all_off(self):
        if not self.hyper_galaxies:
            if self.hyper_image_sky is None and self.hyper_background_noise is None:
                return True
        return False

    @property
    def hypers_all_except_image_sky_off(self):
        if not self.hyper_galaxies:
            if self.hyper_background_noise is None:
                return True
        return False

    def hyper_image_sky_from(self, result: af.Result, as_model=True):

        if self.hyper_image_sky is not None:
            if as_model:
                if hasattr(result, "adapt"):
                    return result.adapt.model.hyper_image_sky
                return result.model.hyper_image_sky
            if hasattr(result, "adapt"):
                return result.adapt.instance.hyper_image_sky
            return result.instance.hyper_image_sky

    def hyper_background_noise_from(self, result: af.Result):

        if self.hyper_background_noise is not None:
            if hasattr(result, "adapt"):
                return result.adapt.instance.hyper_background_noise
            return result.instance.hyper_background_noise
