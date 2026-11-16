"""PhotonTwin: a differentiable optical forward model of a finger lit through skin, in torch.

WHAT IT IS
A tiny renderer. Given the *physical* state of one finger on one device -- arterial saturation `s`, blood volume
fraction `fb`, melanin volume fraction `fm`, the pulsatile blood-volume swing `alpha`, and the SOURCE-TO-DETECTOR
SEPARATION `rho` (millimetres between the flash and the lens, the quantity that ruins phone oximetry) -- it renders
what each of the device's colour channels would measure, by integrating diffuse reflectance over wavelength with
published tissue optics (`spectra.py`). It is differentiable in every input, so it can be used as the DECODER of an
amortised inference network: the network looks at a window and infers the physical state, and the only way it can
explain the measurement is through real optics.

WHY THAT MATTERS HERE
A phone's flash sits 5-15 mm from its lens. Light has to make that trip inside the finger, so the photons that
arrive have taken long, multiply-scattered, oblique paths. Three things follow, all of which this model produces
from first principles rather than from a fitted curve:
  1. the DC level collapses, and collapses far faster in blue than in red (mu_eff is wavelength dependent);
  2. the mean photon path grows with rho, so the pulsatile modulation per unit blood-volume swing grows too;
  3. because a camera channel is BROAD, the surviving light inside each band shifts to the long-wavelength side as
     rho grows -- and the HbO2/Hb extinction ratio varies steeply across the red band -- so the ratio-of-ratios
     calibration curve itself MOVES. A model with one fixed calibration is biased on a phone it was not fitted on.
Item 3 is why band integration is done honestly here instead of with one effective wavelength per channel: with a
single wavelength the effect vanishes identically, and the model would be blind to the thing that actually breaks.

GEOMETRY
Semi-infinite homogeneous turbid medium, spatially resolved diffuse reflectance in the diffusion approximation
(T. J. Farrell, M. S. Patterson and B. Wilson, "A diffusion theory model of spatially resolved, steady-state diffuse
reflectance for the non-invasive determination of tissue optical properties in vivo", Med. Phys. 19(4):879-888, 1992):

    R(rho) = a' / (4 pi) * [ z0 (mu_eff + 1/r1) exp(-mu_eff r1) / r1^2
                           + (z0 + 2 zb) (mu_eff + 1/r2) exp(-mu_eff r2) / r2^2 ]

    mu_t'  = mu_a + mu_s'      a' = mu_s' / mu_t'      z0 = 1 / mu_t'      zb = 2 A / (3 mu_t')
    mu_eff = sqrt(3 mu_a mu_t')     r1 = sqrt(rho^2 + z0^2)     r2 = sqrt(rho^2 + (z0 + 2 zb)^2)

with A = (1 + Reff)/(1 - Reff) for the internal-reflection boundary (Reff = 0.493 at n = 1.4, Groenhuis/Haskell).
A contact pulse oximeter is the same expression at a small rho with two narrow LED bands, so ONE renderer serves
both the phone camera and the clinical oximeter, and the 660/940 nm arterial-blood-gas calibration of the clinical
device is inherited by the phone through shared physics rather than through shared features.

EVERYTHING IS DONE IN LOG SPACE. At rho = 10 mm the blue channel's reflectance is of order exp(-130): it underflows
float32 and even float64. So the renderer returns ln R, sums bands with logsumexp, and -- the useful part -- the
pulsatile amplitude is a DIFFERENCE of log intensities:

    ln I_c(fb)  and  ln I_c(fb (1 + alpha))   ->   d_c = ln I_c(fb (1+alpha)) - ln I_c(fb)

|d_c| is exactly the quantity the OxyDrop feature extractor measures: `features.py` band-passes ln(RGB) and takes
peak-to-trough, i.e. |Delta ln I_c|. So there is no modified-Beer-Lambert step, no effective-pathlength fudge and no
unit conversion between the model and the data: the renderer predicts the measured channel directly, and the
pathlength, the band shift and the nonlinearity in saturation all come out of the two renders.

OBSERVABLES (what `predict` returns, in the units of `features.py`)
    ror_RG = ln(|d_R| / |d_G|)        ror_RB = ln(|d_R| / |d_B|)        (log ratio-of-ratios, features.ABS)
    lac_c  = ln|d_c|                                                     (log modulation depth, features.ABS)
    dc_c   = ln I_c                                                      (log DC, features.ABS; + a device gain)
and, for the baseline-relative side vector the model is trained against (`features.SIDE`), the SAME renderer is run
twice -- once at the window's state, once at the person's first-minute baseline state -- and the difference is taken,
so every device constant (lens gain, white balance, flash brightness) cancels exactly.

The published constants are multiplied by learned scalars (`log_k_*`): the physics fixes the SHAPE of each spectrum,
and the data is allowed to correct the scale, which is where tissue heterogeneity, hematocrit and the unpublished
camera curves live. Those scalars are shared across every person and device, so no per-subject parameter is fitted.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import torch
from torch import nn

from . import spectra as SP

REFF = 0.493                      # effective internal reflectance at a tissue/air boundary, n = 1.4
A_BND = (1.0 + REFF) / (1.0 - REFF)

# latent order, used by the encoder, the export and the docs
LATENTS = ['sat', 'log_fb', 'log_fm', 'log_rho', 'log_alpha']
LAT_LO = np.array([0.50, np.log(0.002), np.log(0.002), np.log(0.4), np.log(0.0005)])
LAT_HI = np.array([1.00, np.log(0.12), np.log(0.45), np.log(25.0), np.log(0.08)])
# prior means / sds in the *unconstrained* parameterisation (sat is handled separately)
PRIOR_MU = np.array([0.0, np.log(0.02), np.log(0.03), np.log(3.0), np.log(0.008)])
PRIOR_SD = np.array([1.0, 0.8, 1.0, 0.8, 0.8])

# device band definitions: how each channel of each domain sees the spectrum
PHONE_CENTRE = SP.CAM_CENTRE                 # R, G, B
PHONE_FWHM = SP.CAM_FWHM
CONTACT_CENTRE = np.array([660.0, 940.0, 0.0])      # red, infrared, (unused slot)
CONTACT_FWHM = np.array([25.0, 40.0, 1.0])          # LED emission widths


@dataclass
class OpticsCfg:
    learn_bands: bool = True       # learn the camera band centres / widths
    learn_scales: bool = True      # learn the scalar corrections to the published spectra
    n_quad: int = 0                # 0 = use the full 5 nm grid; >0 = sub-sample for speed (about n_quad evenly spaced grid points)
    lam_max: float = 0.0           # 0 = keep the whole grid; >0 = drop wavelengths above lam_max nm (phone-only use: the IR-cut filter ends near 700 nm)
    clamp_lnR: float = -400.0

    def to_dict(self):
        return asdict(self)


def _t(x, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def lam_index(cfg):
    """Indices into spectra.LAM used by a renderer with this config (`lam_max` crop, then `n_quad` even sub-sampling)."""
    idx = np.arange(SP.N_LAM)
    if cfg.lam_max and cfg.lam_max > 0:
        idx = idx[SP.LAM[idx] <= cfg.lam_max + 1e-9]
    if cfg.n_quad and 0 < cfg.n_quad < len(idx):
        idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, int(cfg.n_quad))).astype(int))]
    return idx


class PhotonTwin(nn.Module):
    """The renderer. All public methods take and return torch tensors with a leading batch dimension."""

    def __init__(self, cfg: OpticsCfg | None = None, dtype=torch.float64):
        super().__init__()
        self.cfg = cfg = cfg or OpticsCfg()
        self.dt = dtype
        idx = lam_index(cfg)
        lam = SP.LAM[idx]
        self.register_buffer('lam', _t(lam, dtype))
        self.register_buffer('eps_hbo2', _t(SP.EPS_HBO2[idx], dtype))
        self.register_buffer('eps_hb', _t(SP.EPS_HB[idx], dtype))
        self.register_buffer('musp0', _t(SP.mu_s_prime()[idx], dtype))
        self.register_buffer('mu_mel0', _t(SP.mu_a_melanin()[idx], dtype))
        self.register_buffer('mu_base0', _t(SP.mu_a_baseline()[idx], dtype))
        self.register_buffer('spd', _t(SP.led_flash_spd()[idx], dtype))
        self.register_buffer('c_haem', torch.tensor(SP.C_HAEM_M, dtype=dtype))
        # learned scalar corrections to the published spectra (shared by every person and device)
        z = lambda: nn.Parameter(torch.zeros((), dtype=dtype), requires_grad=cfg.learn_scales)
        self.log_k_musp, self.log_k_mel, self.log_k_base, self.log_k_blood = z(), z(), z(), z()
        # learned device bands: phone R/G/B and contact red/IR
        self.band_centre = nn.Parameter(_t(np.stack([PHONE_CENTRE, CONTACT_CENTRE]), dtype), requires_grad=cfg.learn_bands)
        self.band_logw = nn.Parameter(_t(np.log(np.stack([PHONE_FWHM, CONTACT_FWHM])), dtype), requires_grad=cfg.learn_bands)
        self.register_buffer('band_centre0', _t(np.stack([PHONE_CENTRE, CONTACT_CENTRE]), dtype))
        self.register_buffer('band_logw0', _t(np.log(np.stack([PHONE_FWHM, CONTACT_FWHM])), dtype))
        # a contact oximeter illuminates with its own LEDs, a camera with the white flash
        self.register_buffer('use_flash', torch.tensor([1.0, 0.0], dtype=dtype))

    # ------------------------------------------------------------------ optical coefficients
    def mu_a(self, sat, fb, fm):
        """Absorption coefficient [1/mm], shape [B, n_lam]. sat, fb, fm are [B]."""
        eps = sat[:, None] * self.eps_hbo2[None] + (1.0 - sat[:, None]) * self.eps_hb[None]
        blood = 2.303 * eps * self.c_haem / 10.0 * torch.exp(self.log_k_blood)
        mel = self.mu_mel0[None] * torch.exp(self.log_k_mel)
        base = self.mu_base0[None] * torch.exp(self.log_k_base)
        return fb[:, None] * blood + fm[:, None] * mel + base

    def mu_sp(self):
        return self.musp0 * torch.exp(self.log_k_musp)

    # ------------------------------------------------------------------ Farrell diffuse reflectance, in log space
    def ln_reflectance(self, mua, rho):
        """ln R(rho) of Farrell et al. 1992. mua [B, n_lam] 1/mm, rho [B] mm. Returns [B, n_lam]."""
        musp = self.mu_sp()[None]
        mut = mua + musp
        a_p = musp / mut
        z0 = 1.0 / mut
        zb = 2.0 * A_BND / (3.0 * mut)
        mueff = torch.sqrt(3.0 * mua * mut)
        r = rho[:, None]
        r1 = torch.sqrt(r * r + z0 * z0)
        r2 = torch.sqrt(r * r + (z0 + 2.0 * zb) ** 2)
        ln1 = torch.log(z0) + torch.log(mueff + 1.0 / r1) - mueff * r1 - 2.0 * torch.log(r1)
        ln2 = torch.log(z0 + 2.0 * zb) + torch.log(mueff + 1.0 / r2) - mueff * r2 - 2.0 * torch.log(r2)
        # Farrell et al. 1992 (Eq. 10, the diffuse reflectance as the surface flux) ADDS the real-source and the image-source
        # terms: the image source has negative strength but its z-gradient at the surface has the same sign. The earlier
        # draft subtracted them, which made R(rho) decay 10-20 orders of magnitude too fast and could hit the 1e-30 guard.
        # logaddexp keeps the sum exact in log space down to ln R of order -700.
        ln_r = torch.log(a_p) - np.log(4.0 * np.pi) + torch.logaddexp(ln1, ln2)
        return torch.clamp(ln_r, min=self.cfg.clamp_lnR)

    # ------------------------------------------------------------------ device bands
    def bands(self, domain):
        """[B, 3, n_lam] log of (channel sensitivity x illumination), rows that do not exist are -inf-ish.
        `domain` [B] long: 0 phone RGB, 1 contact red/IR."""
        c = self.band_centre[domain]                              # [B, 3]
        w = torch.exp(self.band_logw[domain])
        sd = w / 2.3548
        q = -0.5 * ((self.lam[None, None, :] - c[:, :, None]) / sd[:, :, None]) ** 2
        q = q - torch.logsumexp(q, dim=-1, keepdim=True)          # normalise each band
        flash = self.use_flash[domain][:, None, None] * torch.log(self.spd)[None, None, :]
        return q + flash

    # ------------------------------------------------------------------ the render
    def ln_channels(self, sat, fb, fm, rho, domain):
        """ln of the per-channel measured intensity (up to a device gain), [B, 3]."""
        ln_r = self.ln_reflectance(self.mu_a(sat, fb, fm), rho)
        return torch.logsumexp(self.bands(domain) + ln_r[:, None, :], dim=-1)

    def render(self, sat, fb, fm, rho, alpha, domain):
        """One optical state -> the observables `features.py` measures.
        Returns dict with ln_dc [B,3], ln_amp [B,3] ( = ln|Delta ln I| ), ror [B,2]."""
        ln_dc = self.ln_channels(sat, fb, fm, rho, domain)
        ln_dc2 = self.ln_channels(sat, fb * (1.0 + alpha), fm, rho, domain)
        d = ln_dc2 - ln_dc                                        # negative: more blood absorbs more light
        ln_amp = torch.log(torch.clamp(d.abs(), min=1e-12))
        ror = torch.stack([ln_amp[:, 0] - ln_amp[:, 1], ln_amp[:, 0] - ln_amp[:, 2]], dim=1)
        return dict(ln_dc=ln_dc, ln_amp=ln_amp, ror=ror)

    def side(self, z, z_base, domain):
        """The baseline-relative observables of `features.SIDE[0:8]`, which is what the model is trained against.
        `z`, `z_base`: dicts of [B] tensors (sat, fb, fm, rho, alpha). Every device constant cancels in the difference.
        Returns [B, 8]: d_ror_RG, d_ror_RB, d_dc_R, d_dc_G, d_dc_B, d_lac_R, d_lac_G, d_lac_B."""
        a = self.render(z['sat'], z['fb'], z['fm'], z['rho'], z['alpha'], domain)
        b = self.render(z_base['sat'], z_base['fb'], z_base['fm'], z_base['rho'], z_base['alpha'], domain)
        return torch.cat([a['ror'] - b['ror'], a['ln_dc'] - b['ln_dc'], a['ln_amp'] - b['ln_amp']], dim=1)

    # ------------------------------------------------------------------ diagnostics the demo and the paper use
    def calibration_curve(self, sat_grid, fb=0.02, fm=0.03, rho=3.0, alpha=0.008, domain=0, pair=0):
        """The model's OWN ratio-of-ratios calibration curve at a given geometry: ror versus saturation.
        `pair` 0 = R/G (or red/IR for domain 1), 1 = R/B. Returns numpy [len(sat_grid)]."""
        n = len(sat_grid)
        k = lambda v: torch.full((n,), float(v), dtype=self.dt, device=self.lam.device)
        with torch.no_grad():
            out = self.render(_t(sat_grid, self.dt).to(self.lam.device), k(fb), k(fm), k(rho), k(alpha),
                              torch.full((n,), int(domain), dtype=torch.long, device=self.lam.device))
        return out['ror'][:, pair].cpu().numpy()

    def effective_wavelength(self, sat=0.97, fb=0.02, fm=0.03, rho=3.0, domain=0):
        """Intensity-weighted mean wavelength actually reaching the sensor in each channel [3].
        This is the quantity that drifts red as `rho` grows and moves the calibration curve."""
        k = lambda v: torch.full((1,), float(v), dtype=self.dt, device=self.lam.device)
        with torch.no_grad():
            ln_r = self.ln_reflectance(self.mu_a(k(sat), k(fb), k(fm)), k(rho))
            w = torch.softmax(self.bands(torch.zeros(1, dtype=torch.long, device=self.lam.device) + int(domain))
                              + ln_r[:, None, :], dim=-1)
            return (w * self.lam[None, None, :]).sum(-1)[0].cpu().numpy()

    def param_report(self):
        with torch.no_grad():
            return dict(k_musp=float(torch.exp(self.log_k_musp)), k_mel=float(torch.exp(self.log_k_mel)),
                        k_base=float(torch.exp(self.log_k_base)), k_blood=float(torch.exp(self.log_k_blood)),
                        band_centre=self.band_centre.cpu().numpy().tolist(),
                        band_fwhm=torch.exp(self.band_logw).cpu().numpy().tolist())


# ---------------------------------------------------------------------------------------------- latent plumbing
def unpack(raw, lat_lo=None, lat_hi=None):
    """Map an unconstrained network output [B, 5] to the physical latents, each squashed into its prior box.
    Returns dict(sat, fb, fm, rho, alpha) of [B] tensors and the raw tensor for the KL."""
    lo = _t(LAT_LO if lat_lo is None else lat_lo, raw.dtype).to(raw.device)
    hi = _t(LAT_HI if lat_hi is None else lat_hi, raw.dtype).to(raw.device)
    u = lo + (hi - lo) * torch.sigmoid(raw)
    return dict(sat=u[:, 0], fb=torch.exp(u[:, 1]), fm=torch.exp(u[:, 2]), rho=torch.exp(u[:, 3]), alpha=torch.exp(u[:, 4]))


def sat_to_raw(sat):
    """Inverse of `unpack` for the saturation coordinate only (used to initialise and to supervise in raw space)."""
    lo, hi = LAT_LO[0], LAT_HI[0]
    p = np.clip((np.asarray(sat, float) - lo) / (hi - lo), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))
