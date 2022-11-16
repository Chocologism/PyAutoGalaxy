from typing import Tuple

from autogalaxy.profiles.light.snr.abstract import LightProfileSNR
from autogalaxy.profiles.light import base as lp


class EllSersicCore(lp.EllSersicCore, LightProfileSNR):
    def __init__(
        self,
        signal_to_noise_ratio: float = 10.0,
        centre: Tuple[float, float] = (0.0, 0.0),
        elliptical_comps: Tuple[float, float] = (0.0, 0.0),
        effective_radius: float = 0.6,
        sersic_index: float = 4.0,
        radius_break: float = 0.01,
        intensity: float = 0.05,
        gamma: float = 0.25,
        alpha: float = 3.0,
    ):
        """
        The elliptical cored-Sersic light profile.

        Instead of an `intensity` a `signal_to_noise_ratio` is input which sets the signal to noise of the brightest
        pixel of the profile's image when used to simulate imaging data.

        Parameters
        ----------
        signal_to_noise_ratio
            The signal to noise of the light profile when it is used to simulate strong lens imaging.
        centre
            The (y,x) arc-second coordinates of the profile centre.
        elliptical_comps
            The first and second ellipticity components of the elliptical coordinate system, (see the module
            `autogalaxy -> convert.py` for the convention).
        effective_radius
            The circular radius containing half the light of this profile.
        sersic_index
            Controls the concentration of the profile (lower -> less concentrated, higher -> more concentrated).
        radius_break
            The break radius separating the inner power-law (with logarithmic slope gamma) and outer Sersic function.
        intensity
            The intensity at the break radius.
        gamma
            The logarithmic power-law slope of the inner core profiles
        alpha
            Controls the sharpness of the transition between the inner core / outer Sersic profiles.
        """
        super().__init__(
            centre=centre,
            elliptical_comps=elliptical_comps,
            effective_radius=effective_radius,
            sersic_index=sersic_index,
            radius_break=radius_break,
            intensity=0.0,
            alpha=alpha,
            gamma=gamma,
        )

        self.intensity = self.intensity

        LightProfileSNR.__init__(self, signal_to_noise_ratio=signal_to_noise_ratio)
