from typing import Tuple
import numpy as np

import autoarray as aa
from autogalaxy.profiles.mass.abstract.abstract import MassProfile


class dPIESph(MassProfile):
    '''
    The dual Pseudo-Isothermal Elliptical mass distribution introduced in
    Eliasdottir 2007: https://arxiv.org/abs/0710.5636

    This version is without ellipticity, so perhaps the "E" is a misnomer.

    Corresponds to a projected density profile that looks like:

        \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                      (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

    (c.f. Eliasdottir '07 eqn. A3)

    In this parameterization, ra and rs are the scale radii above in angular
    units (arcsec). The parameter `kappa_scale` is \\Sigma_0 / \\Sigma_crit.
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    def _deflection_angle(self, radii):
        '''
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        '''
        r_ra = radii / self.ra
        r_rs = radii / self.rs
        # c.f. Eliasdottir '07 eq. A20
        f = (
            r_ra / (1 + np.sqrt(1 + r_ra * r_ra))
            - r_rs / (1 + np.sqrt(1 + r_rs * r_rs))
        )

        ra, rs = self.ra, self.rs
        # c.f. Eliasdottir '07 eq. A19
        # magnitude of deflection
        alpha = 2 * self.kappa_scale * ra * rs / (rs - ra) * f
        return alpha

    def _convergence(self, radii):
        radsq = radii * radii
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        return (
            self.kappa_scale * (a * s) / (s - a) *
            (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        )

    def _potential(self, radii):
        raise NotImplementedError

    @aa.grid_dec.to_vector_yx
    #@aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        alpha = self._deflection_angle(radii)

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(
            defl_y, defl_x
        )

    #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        return self._convergence(np.sqrt(radsq))

    #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        return self._potential(np.sqrt(radsq))


class dPIE(dPIESph):
    '''
    The dual Pseudo-Isothermal Elliptical mass distribution introduced in
    Eliasdottir 2007: https://arxiv.org/abs/0710.5636

    Corresponds to a projected density profile that looks like:

        \\Sigma(R) = \\Sigma_0 (ra * rs) / (rs - ra) *
                      (1 / \\sqrt(ra^2 + R^2) - 1 / \\sqrt(rs^2 + R^2))

    (c.f. Eliasdottir '07 eqn. A3)

    In this parameterization, ra and rs are the scale radii above in angular
    units (arcsec). The parameter kappa_scale is \\Sigma_0 / \\Sigma_crit.

    WARNING: This uses the "pseud-elliptical" approximation, where the ellipticity
    is applied to the *potential* rather than the *mass* to ease calculation.
    Use at your own risk! (And TODO Jack: fix this!)
    This approximation is used by the lenstronomy PJAFFE profile (which is the
    same functional form), but not by the lenstool PIEMD (also synonymous with this),
    which correctly solved the differential equations for the mass-based ellipticity.
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        kappa_scale: float = 0.1,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)
        if ra > rs:
            ra, rs = rs, ra
        self.ra = ra
        self.rs = rs
        self.kappa_scale = kappa_scale

    def _align_to_major_axis(self, ys, xs):
        '''
        Aligns coordinates to the major axis of this halo. Returns (y', x'),
        where x' is along the major axis and y' is along the minor axis.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        _xs = (costheta * xs + sintheta * ys)
        _ys = (-sintheta * xs + costheta * ys)
        return _ys, _xs

    def _align_from_major_axis(self, _ys, _xs):
        '''
        Given _ys and _xs as offsets along the minor and major axes,
        respectively, this transforms them back to the regular coordinate
        system.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        xs = (costheta * _xs + -sintheta * _ys)
        ys = (sintheta * _xs + costheta * _ys)
        return ys, xs

    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0]**2 + self.ell_comps[1]**2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    @aa.grid_dec.to_vector_yx
    #@aa.grid_dec.grid_2d_to_structure
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(_radii)

        # This is in axes aligned to the major/minor axis
        _defl_xs = alpha_circ * np.sqrt(1 - ellip) * (_xs / _radii)
        _defl_ys = alpha_circ * np.sqrt(1 + ellip) * (_ys / _radii)

        # And here we convert back to the real axes
        defl_ys, defl_xs = self._align_from_major_axis(_defl_ys, _defl_xs)
        return aa.Grid2DIrregular.from_yx_1d(
            defl_ys, defl_xs
        )

    #@aa.grid_dec.grid_2d_to_structure
    def convergence_2d_from(self, grid: aa.type.Grid2DLike):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(_radii)
        alpha_circ = self._deflection_angle(_radii)

        asymm_term = (ellip * (1 - ellip) * _xs**2 - ellip * (1 + ellip) * _ys**2) / _radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / _radii) * asymm_term

    #@aa.grid_dec.grid_2d_to_structure
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)
        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))
        return super(dPIESph, self)._potential(_radii)

class dPIESph_custom1(MassProfile):
    '''
    use einstein radius instead of kappa_scale, just for test.
    I wanna see if I can use a einstein radius which satisfies the "intermediate-axis-convention"
    that will be the same as calculated by critical lines.
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        einstein_radius: float = 1.0,
    ):
        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.einstein_radius = einstein_radius
        self.kappa_scale_fake = self._kappa_scale_fake_sph()

    def _kappa_scale(self):
        return (self.kappa_scale_fake * (self.rs-self.ra)/(2*self.ra*self.rs))

    def _kappa_scale_fake_sph(self):
        return (self.einstein_radius**2/(np.sqrt(self.ra**2+self.einstein_radius**2)-np.sqrt(self.rs**2+self.einstein_radius**2)+self.rs-self.ra))
    
    def _test_E0(self):
        return (self.kappa_scale_fake*self.rs/(self.ra + self.rs))
    
    def _deflection_angle(self, radii):
        '''
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        '''
        # r_ra = radii / self.ra
        # r_rs = radii / self.rs
        # c.f. Eliasdottir '07 eq. A20
        # f = (
        #     r_ra / (1 + np.sqrt(1 + r_ra * r_ra))
        #     - r_rs / (1 + np.sqrt(1 + r_rs * r_rs))
        # )

        ra, rs = self.ra, self.rs
        radii = np.maximum(radii, 1e-8)
        f = (
            radii / (ra + np.sqrt(ra**2 + radii**2))
            - radii / (rs + np.sqrt(rs**2 + radii**2))
        )

        # c.f. Eliasdottir '07 eq. A19
        # magnitude of deflection
        alpha = self.kappa_scale_fake * f
        return alpha

    def _convergence(self, radii):
        radsq = radii * radii
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        return (
            self.kappa_scale_fake / 2 *
            (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        )

    def _potential(self, radii):
        raise NotImplementedError

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        alpha = self._deflection_angle(radii)

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(
            defl_y, defl_x
        )

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        return self._convergence(np.sqrt(radsq))

    # @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # already transformed to center on profile centre so this works
        # radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        # return self._potential(np.sqrt(radsq))
        potential_grid = np.zeros(grid.shape[0])
        return potential_grid
    
class dPIE_custom1(dPIESph_custom1):
    '''

    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        einstein_radius: float = 1.0,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)
        if ra > rs:
            ra, rs = rs, ra
        self.ra = ra
        self.rs = rs
        self.einstein_radius = einstein_radius #如果传入的爱因斯坦半径是imac标准下的，那内部使用的theta_E_star要转换一下，否则不用转换
        self.kappa_scale_fake = self._kappa_scale_fake_ell()

    def _kappa_scale(self):
        return (self.kappa_scale_fake * (self.rs-self.ra)/(2*self.ra*self.rs))
    

########用爱因斯坦半径直接转换的思路#########
#### Eliasdottir定义
    def _kappa_scale_fake_ell(self):

        # ell_correction = self._ell_correction()
        # kappa_scale_fake_sph = self._kappa_scale_fake_sph()
        einstein_radius_star = self.einstein_radius#这里是不做转换的
        kappa_scale_fake_ell = (
            einstein_radius_star**2
            /(np.sqrt(self.ra**2+einstein_radius_star**2)
              -np.sqrt(self.rs**2+einstein_radius_star**2)
              +self.rs-self.ra)
        )
        return (kappa_scale_fake_ell)

    def _align_to_major_axis(self, ys, xs):
        '''
        Aligns coordinates to the major axis of this halo. Returns (y', x'),
        where x' is along the major axis and y' is along the minor axis.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        _xs = (costheta * xs + sintheta * ys)
        _ys = (-sintheta * xs + costheta * ys)
        return _ys, _xs

    def _align_from_major_axis(self, _ys, _xs):
        '''
        Given _ys and _xs as offsets along the minor and major axes,
        respectively, this transforms them back to the regular coordinate
        system.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        xs = (costheta * _xs + -sintheta * _ys)
        ys = (sintheta * _xs + costheta * _ys)
        return ys, xs

    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0]**2 + self.ell_comps[1]**2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip)) #Jack定义
        # _radii = np.sqrt(_xs**2 / (1 + ellip)**2 + _ys**2 * (1 - ellip)**2) # Eliasdottir定义

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(_radii)

        # This is in axes aligned to the major/minor axis
        _defl_xs = alpha_circ * np.sqrt(1 - ellip) * (_xs / _radii)
        _defl_ys = alpha_circ * np.sqrt(1 + ellip) * (_ys / _radii)

        # # This is in axes aligned to the major/minor axis
        # _defl_xs = alpha_circ * (_xs / _radii)
        # _defl_ys = alpha_circ * (_ys / _radii)

        # And here we convert back to the real axes
        defl_ys, defl_xs = self._align_from_major_axis(_defl_ys, _defl_xs)
        return aa.Grid2DIrregular.from_yx_1d(
            defl_ys, defl_xs
        )

    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(_radii)
        alpha_circ = self._deflection_angle(_radii)

        asymm_term = (ellip * (1 - ellip) * _xs**2 - ellip * (1 + ellip) * _ys**2) / _radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / _radii) * asymm_term

    # @aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # ys, xs = grid.T
        # (ycen, xcen) = self.centre
        # xoff, yoff = xs - xcen, ys - ycen
        # _ys, _xs = self._align_to_major_axis(yoff, xoff)
        # ellip = self._ellip()
        # _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))
        # potential_grid = np.zeros(grid.shape[0])
        # # return super(dPIESph, self)._potential(_radii)
        potential_grid = np.zeros(grid.shape[0])
        return potential_grid


class dPIESph_custom2(MassProfile):
    '''
    use velocity dispersion instead of kappa_scale

    this has been tested well in real data, so I think it's correct.

    E0 = 6*\\pi* (D_{ds}*D{d})/D{s} * \\sigma^2 / c^2

    E0 have the unit same as a, s, and D, so the unit of E0 should be arcsec
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        E0: float = 1.0,
    ):
        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.E0 = E0

    def _deflection_angle(self, radii):
        '''
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        '''
        # r_ra = radii / self.ra
        # r_rs = radii / self.rs
        # # c.f. Eliasdottir '07 eq. A20
        # f = (
        #     r_ra / (1 + np.sqrt(1 + r_ra * r_ra))
        #     - r_rs / (1 + np.sqrt(1 + r_rs * r_rs))
        # )

        a, s = self.ra, self.rs
        radii = np.maximum(radii, 1e-8)
        f = (
            radii / (a + np.sqrt(a**2 + radii**2))
            - radii / (s + np.sqrt(s**2 + radii**2))
        )

        # c.f. Eliasdottir '07 eq. A23
        # magnitude of deflection
        alpha = self.E0 * (s + a) / s * f
        return alpha

    def _convergence(self, radii):
        radsq = radii * radii
        a, s = self.ra, self.rs
        # c.f. Eliasdottir '07 eqn (A3)
        return (
            self.E0 / 2 * (s + a) / s *
            (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        )
    
    def _theta_E_from_WangH(self):
        return self.E0 * (self.rs + self.ra)**2 * (self.rs - self.ra) / self.rs**3

    def _potential(self, radii):
        return 0
        # raise NotImplementedError

    # @aa.grid_dec.to_vector_yx
    # #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        alpha = self._deflection_angle(radii)

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(
            defl_y, defl_x
        )

    # #@aa.grid_dec.grid_2d_to_structure
    # @aa.grid_dec.transform
    # @aa.grid_dec.relocate_to_radial_minimum
    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        return self._convergence(np.sqrt(radsq))

    #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # already transformed to center on profile centre so this works
        # radsq = (grid[:, 0]**2 + grid[:, 1]**2)

        # return self._potential(np.sqrt(radsq))
        # return aa.Grid2DIrregular.from_yx_1d(
        #     0, 0
        # )
        potential_grid = np.zeros(grid.shape[0])
        return potential_grid


class dPIE_custom2(dPIESph_custom2):
    '''
    use velocity dispersion instead of kappa_scale

    E0 = 6*\\pi* (D_{ds}*D{d})/D{s} * \\sigma^2 / c^2

    E0 have the unit same as a, s, and D, so the unit of E0 should be arcsec
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        E0: float = 1.0,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)
        if ra > rs:
            ra, rs = rs, ra
        self.ra = ra
        self.rs = rs
        self.E0 = E0

    def _align_to_major_axis(self, ys, xs):
        '''
        Aligns coordinates to the major axis of this halo. Returns (y', x'),
        where x' is along the major axis and y' is along the minor axis.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        _xs = (costheta * xs + sintheta * ys)
        _ys = (-sintheta * xs + costheta * ys)
        return _ys, _xs

    def _align_from_major_axis(self, _ys, _xs):
        '''
        Given _ys and _xs as offsets along the minor and major axes,
        respectively, this transforms them back to the regular coordinate
        system.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        xs = (costheta * _xs + -sintheta * _ys)
        ys = (sintheta * _xs + costheta * _ys)
        return ys, xs

    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0]**2 + self.ell_comps[1]**2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    # @aa.grid_dec.to_vector_yx
    # #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(_radii)

        # This is in axes aligned to the major/minor axis
        _defl_xs = alpha_circ * np.sqrt(1 - ellip) * (_xs / _radii)
        _defl_ys = alpha_circ * np.sqrt(1 + ellip) * (_ys / _radii)

        # And here we convert back to the real axes
        defl_ys, defl_xs = self._align_from_major_axis(_defl_ys, _defl_xs)
        return aa.Grid2DIrregular.from_yx_1d(
            defl_ys, defl_xs
        )

    #@aa.grid_dec.grid_2d_to_structure
    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(_radii)
        alpha_circ = self._deflection_angle(_radii)

        asymm_term = (ellip * (1 - ellip) * _xs**2 - ellip * (1 + ellip) * _ys**2) / _radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / _radii) * asymm_term

    #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # ys, xs = grid.T
        # (ycen, xcen) = self.centre
        # xoff, yoff = xs - xcen, ys - ycen
        # _ys, _xs = self._align_to_major_axis(yoff, xoff)
        # ellip = self._ellip()
        # _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))
        # return super(dPIESph, self)._potential(_radii)
        potential_grid = np.zeros(grid.shape[0])
        return potential_grid

#test github
    
class dPIESph_custom3(MassProfile):
    '''
    use "reduced einstein radius" instead of E0

    E0 = 6*\\pi* (D_{ds}*D{d})/D{s} * \\sigma^2 / c^2
    theta_E = 2 * kappa_scale * ra * (rs + ra) / rs
            = E0 * (rs + ra)**2 * (rs -ra) / rs**3

    theta_E have the unit same as a, s, and D, so the unit of it should be arcsec
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        einstein_radius: float = 1.0,
    ):
        # Ensure rs > ra (things will probably break otherwise)
        if ra > rs:
            ra, rs = rs, ra
        super(MassProfile, self).__init__(centre, (0.0, 0.0))
        self.ra = ra
        self.rs = rs
        self.einstein_radius = einstein_radius

    def _deflection_angle(self, radii):
        '''
        For a circularly symmetric dPIE profile, computes the magnitude of the deflection at each radius.
        '''
        # r_ra = radii / self.ra
        # r_rs = radii / self.rs
        # # c.f. Eliasdottir '07 eq. A20
        # f = (
        #     r_ra / (1 + np.sqrt(1 + r_ra * r_ra))
        #     - r_rs / (1 + np.sqrt(1 + r_rs * r_rs))
        # )

        a, s = self.ra, self.rs
        radii = np.maximum(radii, 1e-8)
        f = (
            radii / (a + np.sqrt(a**2 + radii**2))
            - radii / (s + np.sqrt(s**2 + radii**2))
        )

        # c.f. WangH. '22 eq. A07
        # magnitude of deflection
        alpha = self.einstein_radius * s**2 / (s**2 - a**2) * f
        return alpha

    def _convergence(self, radii):
        radsq = radii * radii
        a, s = self.ra, self.rs
        # c.f. WangH. '22 eqn (A7)
        return (
            self.einstein_radius / 2 * s**2 / (s**2 - a**2) *
            (1/np.sqrt(a**2 + radsq) - 1/np.sqrt(s**2 + radsq))
        )

    def _potential(self, radii):
        return 0
        # raise NotImplementedError

    # @aa.grid_dec.to_vector_yx
    # #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        radii = np.sqrt(xoff**2 + yoff**2)

        alpha = self._deflection_angle(radii)

        # now we decompose the deflection into y/x components
        defl_y = alpha * yoff / radii
        defl_x = alpha * xoff / radii
        return aa.Grid2DIrregular.from_yx_1d(
            defl_y, defl_x
        )

    # #@aa.grid_dec.grid_2d_to_structure
    # @aa.grid_dec.transform
    # @aa.grid_dec.relocate_to_radial_minimum
    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        # already transformed to center on profile centre so this works
        radsq = (grid[:, 0]**2 + grid[:, 1]**2)
        return self._convergence(np.sqrt(radsq))

    #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # already transformed to center on profile centre so this works
        # radsq = (grid[:, 0]**2 + grid[:, 1]**2)

        # return self._potential(np.sqrt(radsq))
        # return aa.Grid2DIrregular.from_yx_1d(
        #     0, 0
        # )
        potential_grid = np.zeros(grid.shape[0])
        return potential_grid


class dPIE_custom3(dPIESph_custom3):
    '''
    use velocity dispersion instead of kappa_scale

    E0 = 6*\\pi* (D_{ds}*D{d})/D{s} * \\sigma^2 / c^2

    E0 have the unit same as a, s, and D, so the unit of E0 should be arcsec
    '''
    def __init__(
        self,
        centre: Tuple[float, float] = (0.0, 0.0),
        ell_comps: Tuple[float, float] = (0.0, 0.0),
        ra: float = 0.1,
        rs: float = 2.0,
        einstein_radius: float = 1.0,
    ):
        super(MassProfile, self).__init__(centre, ell_comps)
        if ra > rs:
            ra, rs = rs, ra
        self.ra = ra
        self.rs = rs
        self.einstein_radius = einstein_radius

    def _align_to_major_axis(self, ys, xs):
        '''
        Aligns coordinates to the major axis of this halo. Returns (y', x'),
        where x' is along the major axis and y' is along the minor axis.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        _xs = (costheta * xs + sintheta * ys)
        _ys = (-sintheta * xs + costheta * ys)
        return _ys, _xs

    def _align_from_major_axis(self, _ys, _xs):
        '''
        Given _ys and _xs as offsets along the minor and major axes,
        respectively, this transforms them back to the regular coordinate
        system.

        Does NOT translate, only rotates.
        '''
        costheta, sintheta = self._cos_and_sin_to_x_axis()
        xs = (costheta * _xs + -sintheta * _ys)
        ys = (sintheta * _xs + costheta * _ys)
        return ys, xs

    def _ellip(self):
        ellip = np.sqrt(self.ell_comps[0]**2 + self.ell_comps[1]**2)
        MAX_ELLIP = 0.99999
        return min(ellip, MAX_ELLIP)

    # @aa.grid_dec.to_vector_yx
    # #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.to_vector_yx
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def deflections_yx_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the deflection magnitude of a *non-elliptical* profile
        alpha_circ = self._deflection_angle(_radii)

        # This is in axes aligned to the major/minor axis
        _defl_xs = alpha_circ * np.sqrt(1 - ellip) * (_xs / _radii)
        _defl_ys = alpha_circ * np.sqrt(1 + ellip) * (_ys / _radii)

        # And here we convert back to the real axes
        defl_ys, defl_xs = self._align_from_major_axis(_defl_ys, _defl_xs)
        return aa.Grid2DIrregular.from_yx_1d(
            defl_ys, defl_xs
        )

    #@aa.grid_dec.grid_2d_to_structure
    @aa.over_sample
    @aa.grid_dec.to_array
    @aa.grid_dec.transform
    @aa.grid_dec.relocate_to_radial_minimum
    def convergence_2d_from(self, grid: aa.type.Grid2DLike, **kwargs):
        ys, xs = grid.T
        (ycen, xcen) = self.centre
        xoff, yoff = xs - xcen, ys - ycen
        _ys, _xs = self._align_to_major_axis(yoff, xoff)

        ellip = self._ellip()
        _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))

        # Compute the convergence and deflection of a *circular* profile
        kappa_circ = self._convergence(_radii)
        alpha_circ = self._deflection_angle(_radii)

        asymm_term = (ellip * (1 - ellip) * _xs**2 - ellip * (1 + ellip) * _ys**2) / _radii**2

        # convergence = 1/2 \nabla \alpha = 1/2 \nabla^2 potential
        # The "asymm_term" is asymmetric on x and y, so averages out to
        # zero over all space
        return kappa_circ * (1 - asymm_term) + (alpha_circ / _radii) * asymm_term

    #@aa.grid_dec.grid_2d_to_structure
    @aa.grid_dec.relocate_to_radial_minimum
    def potential_2d_from(self, grid: aa.type.Grid2DLike):
        # ys, xs = grid.T
        # (ycen, xcen) = self.centre
        # xoff, yoff = xs - xcen, ys - ycen
        # _ys, _xs = self._align_to_major_axis(yoff, xoff)
        # ellip = self._ellip()
        # _radii = np.sqrt(_xs**2 * (1 - ellip) + _ys**2 * (1 + ellip))
        # return super(dPIESph, self)._potential(_radii)
        potential_grid = np.zeros(grid.shape[0])
        return potential_grid
