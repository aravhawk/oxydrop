"""Published optical constants for the PhotonTwin forward model, on a single wavelength grid.

Everything here is literature data or a literature formula; nothing is fitted to OxyDrop's own recordings.
The forward model (`optics.py`) multiplies each of these by a LEARNED scalar correction, so the results do not
depend on the third digit of any constant below; what the physics supplies is the *shape* of each spectrum.

Sources, all fetched 2026-11-15 and listed with their exact citation in `optics_data/SOURCES.md`:
  * haemoglobin molar extinction, 400-1000 nm in 2 nm steps: S. Prahl, "Tabulated Molar Extinction Coefficient for
    Hemoglobin in Water", Oregon Medical Laser Centre, https://omlc.org/spectra/hemoglobin/summary.html (data of
    Gratzer and Kollias). Stored verbatim in `optics_data/hemoglobin_prahl.csv` (lambda_nm, eps_hbo2, eps_hb in
    cm^-1 per mol/l). The page's conversion is mu_a [cm^-1] = 2.303 * eps * x / 64500 with x in g/l, 64500 g/mol.
  * reduced scattering, melanin absorption and bloodless-skin baseline absorption: S. L. Jacques, "Skin Optics",
    OMLC News, January 1998, https://omlc.org/news/jan98/skinoptics.html --
      mu_s' = 2e5 lam^-1.5 + 2e12 lam^-4  [cm^-1]   (Mie fibres + Rayleigh; epidermis = dermis to first order)
      mu_a,melanosome = 6.6e11 lam^-3.33  [cm^-1]   (after Jacques and McAuliffe 1991)
      mu_a,baseline   = 0.244 + 85.3 exp(-(lam - 154)/66.2)  [cm^-1]   (Huang rat-skin data)
    with lam in nm. Melanosome volume fraction from the same page: 1.3-6.3 % light, 11-16 % moderate, 18-43 % dark.
  * camera spectral sensitivity: Gaussian bands initialised to the phone-camera range of Jiang, Liu, Gu and
    Susstrunk, WACV 2013 (28 cameras, 400-720 nm); `optics.py` learns the centres, widths and a correction.
  * flash: shape of a phosphor-converted white LED (CIE 15:2018 LED-B family), mix learned.
  * water absorption is negligible below 700 nm over the 1-10 mm paths used here and is inside mu_a,baseline.

Grid: 420-1000 nm in 5 nm steps: the white-LED + silicon-CMOS band of a phone camera (420-700 nm, an IR-cut filter
kills the rest) and the 660 / 940 nm LED pair of a contact pulse oximeter, so ONE forward model serves both devices.
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, 'optics_data')

LAM_LO, LAM_HI, LAM_STEP = 420.0, 1000.0, 5.0
LAM = np.arange(LAM_LO, LAM_HI + LAM_STEP / 2, LAM_STEP)       # [n_lam] nm
N_LAM = len(LAM)

# --- haemoglobin ---------------------------------------------------------------------------------------------
HB_MW = 64500.0            # g/mol of haemoglobin (4 haems), the constant used by the OMLC conversion
HB_BLOOD_G_L = 150.0       # g/L haemoglobin in whole blood (adult reference)
C_HAEM_M = HB_BLOOD_G_L / HB_MW * 4.0   # mol/L of haem groups in whole blood (~9.3 mM)


def _load_hb():
    p = os.path.join(DATA, 'hemoglobin_prahl.csv')
    if not os.path.exists(p):
        raise FileNotFoundError(
            f'{p} is missing. It holds the published Prahl/OMLC haemoglobin extinction table '
            '(lambda_nm, eps_hbo2, eps_hb in cm^-1/M). Nothing in this package invents those numbers.')
    import pandas as pd
    t = pd.read_csv(p)
    lam = t.lambda_nm.values.astype(float)
    o = np.interp(LAM, lam, t.eps_hbo2.values.astype(float))
    d = np.interp(LAM, lam, t.eps_hb.values.astype(float))
    return o, d


EPS_HBO2, EPS_HB = _load_hb()        # [n_lam] cm^-1 / M (per haem, base-10 molar extinction)


def mu_a_blood(s):
    """Absorption coefficient of whole blood [1/mm] at saturation `s` (0..1), on LAM.
    mu_a = 2.303 * (s eps_HbO2 + (1-s) eps_Hb) * C_haem, converted cm^-1 -> mm^-1.
    `s` may be a scalar or an array; the wavelength axis is appended last."""
    s = np.asarray(s, float)[..., None]
    eps = s * EPS_HBO2 + (1.0 - s) * EPS_HB
    return 2.303 * eps * C_HAEM_M / 10.0


# --- scattering, melanin and bloodless-skin baseline (OMLC "Skin Optics", Jan 1998) ---------------------------
MUSP_MIE_A, MUSP_MIE_P = 2.0e5, 1.5          # mu_s'_Mie      = 2e5  lam^-1.5  [cm^-1]
MUSP_RAY_A, MUSP_RAY_P = 2.0e12, 4.0         # mu_s'_Rayleigh = 2e12 lam^-4    [cm^-1]
MEL_A, MEL_P = 6.6e11, 3.33                  # mu_a,melanosome = 6.6e11 lam^-3.33 [cm^-1] (Jacques and McAuliffe 1991)
BASE_C0, BASE_C1, BASE_L0, BASE_TAU = 0.244, 85.3, 154.0, 66.2   # mu_a,baseline = c0 + c1 exp(-(lam-l0)/tau) [cm^-1]


def mu_s_prime():
    """Reduced scattering coefficient of dermis [1/mm] on LAM (OMLC Mie + Rayleigh two-term form)."""
    cm = MUSP_MIE_A * LAM ** (-MUSP_MIE_P) + MUSP_RAY_A * LAM ** (-MUSP_RAY_P)
    return cm / 10.0


def mu_a_melanin():
    """Absorption coefficient of a fully melanised melanosome [1/mm] on LAM."""
    return MEL_A * LAM ** (-MEL_P) / 10.0


def mu_a_baseline():
    """Absorption of bloodless, melanin-free skin [1/mm] on LAM (Huang rat-skin fit)."""
    return (BASE_C0 + BASE_C1 * np.exp(-(LAM - BASE_L0) / BASE_TAU)) / 10.0


# --- illumination and camera --------------------------------------------------------------------------------
def led_flash_spd(blue_peak=452.0, blue_fwhm=22.0, phos_peak=565.0, phos_fwhm=115.0, blue_frac=0.30):
    """Relative spectral power distribution of a phosphor-converted white LED flash on LAM: a narrow InGaN pump
    plus a broad phosphor band (the shape of CIE 15:2018 LED-B series). Normalised to unit sum."""
    def g(mu, fwhm):
        sd = fwhm / 2.3548
        return np.exp(-0.5 * ((LAM - mu) / sd) ** 2)
    s = blue_frac * g(blue_peak, blue_fwhm) + (1.0 - blue_frac) * g(phos_peak, phos_fwhm)
    return s / s.sum()


# Nominal CMOS phone-camera RGB spectral sensitivities: Gaussian bands at the centres and widths typical of the
# Jiang et al. 2013 camera-sensitivity database (28 cameras, 400-720 nm). `optics.py` learns a correction to these.
CAM_CENTRE = np.array([607.0, 537.0, 463.0])     # R, G, B band centres [nm]
CAM_FWHM = np.array([90.0, 85.0, 70.0])          # R, G, B band FWHM [nm]


def cam_response(centre=None, fwhm=None):
    """[3, n_lam] relative RGB spectral sensitivities, each normalised to unit sum."""
    c = CAM_CENTRE if centre is None else np.asarray(centre, float)
    w = CAM_FWHM if fwhm is None else np.asarray(fwhm, float)
    sd = w / 2.3548
    q = np.exp(-0.5 * ((LAM[None, :] - c[:, None]) / sd[:, None]) ** 2)
    return q / q.sum(1, keepdims=True)


def summary():
    """Dict of the constants, for the docs generator and the claim check."""
    return dict(lam_lo=LAM_LO, lam_hi=LAM_HI, lam_step=LAM_STEP, n_lam=N_LAM, hb_mw=HB_MW,
                hb_blood_g_l=HB_BLOOD_G_L, c_haem_mM=1e3 * C_HAEM_M,
                musp_500_per_mm=float(np.interp(500.0, LAM, mu_s_prime())), musp_660_per_mm=float(np.interp(660.0, LAM, mu_s_prime())),
                mel_a=MEL_A, mel_p=MEL_P, mu_a_base_660_per_mm=float(np.interp(660.0, LAM, mu_a_baseline())),
                eps_hbo2_660=float(np.interp(660.0, LAM, EPS_HBO2)), eps_hb_660=float(np.interp(660.0, LAM, EPS_HB)),
                eps_hbo2_540=float(np.interp(540.0, LAM, EPS_HBO2)), eps_hb_540=float(np.interp(540.0, LAM, EPS_HB)),
                cam_centre=CAM_CENTRE.tolist(), cam_fwhm=CAM_FWHM.tolist())
