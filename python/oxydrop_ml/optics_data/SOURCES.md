# Optical constants used by the PhotonTwin forward model

Every number is published literature, fetched on 2026-11-15. Nothing here is fitted to OxyDrop recordings.
`python/oxydrop_ml/optics.py` multiplies each spectrum by a learned scalar correction, so the model does not
depend on the third digit of any value below; the physics supplies the *shape*.

## `hemoglobin_prahl.csv` — haemoglobin molar extinction, 400-1000 nm, 2 nm steps
Columns `lambda_nm, eps_hbo2, eps_hb`, units cm^-1/(moles/litre) per haem.

Scott Prahl, "Tabulated Molar Extinction Coefficient for Hemoglobin in Water", Oregon Medical Laser Center,
<https://omlc.org/spectra/hemoglobin/summary.html> (SAP, 4 March 1998). Underlying measurements: W. B. Gratzer
(Medical Research Council Labs, Holly Hill, London) and N. Kollias (Wellman Laboratories, Harvard Medical School).

The page's own conversion, which `spectra.mu_a_blood` implements:
  mu_a [cm^-1] = 2.303 * eps * x / 64500,  x = g haemoglobin per litre, 64500 g/mol = molecular weight of haemoglobin.
The page states x = 150 g/l as a typical value for whole blood.

## Melanin absorption
S. L. Jacques, "Skin Optics", Oregon Medical Laser Center News, January 1998,
<https://omlc.org/news/jan98/skinoptics.html>:
  mu_a,melanosome = 6.6e11 * lambda[nm]^-3.33  [cm^-1]
after S. L. Jacques and D. J. McAuliffe, "The melanosome: threshold temperature for explosive vaporization and
internal absorption coefficient during pulsed laser irradiation", Photochem. Photobiol. 53:769-775 (1991).
Melanosome volume fractions from the same page: 1.3-6.3 % lightly pigmented, 11-16 % moderately, 18-43 % darkly.
(The later OMLC melanin index page, <https://omlc.org/spectra/melanin/mua.html>, gives a different fit,
1.70e12 * lambda^-3.48 cm^-1 for skin. We use the 1998 skin-optics fit and learn a scalar correction.)

## Reduced scattering of dermis (and, to first order, epidermis)
Same "Skin Optics" page, two-term Rayleigh + Mie form:
  mu_s'_Mie      = 2e5  * lambda[nm]^-1.5  [cm^-1]
  mu_s'_Rayleigh = 2e12 * lambda[nm]^-4    [cm^-1]
  mu_s'          = mu_s'_Mie + mu_s'_Rayleigh
The page states "musp.epi = musp.derm to first approximation" and that Rayleigh dominates below about 650 nm.
(We did not use the power law of Jacques, Phys. Med. Biol. 58 R37 (2013): its fitted coefficients are behind a
paywall and we do not quote numbers we could not read.)

## Baseline absorption of bloodless skin
Same page, from Huang's rat-skin data (stated to extend 350-1100 nm):
  mu_a,baseline = 0.244 + 85.3 * exp(-(lambda[nm] - 154) / 66.2)  [cm^-1]

## Blood
Same "Skin Optics" page: typical average dermal blood volume fraction 0.2 %, with 2-5 % in the venous plexus
100-200 um below the surface; whole-blood figures assume 45 % haematocrit. A finger pressed on a camera lens is
much more vascular than average dermis, so the model's prior on blood volume fraction is 0.5-8 %.

## Flash spectrum
Shape of a phosphor-converted white LED (narrow InGaN pump plus a broad phosphor band), of the family of the
CIE 15:2018 LED-B illuminants. Parameterised in `spectra.led_flash_spd`; the pump/phosphor mix is a learned latent.

## Camera spectral sensitivity
Initialised to Gaussian R/G/B bands at centres and widths typical of the phone cameras in J. Jiang, D. Liu, J. Gu
and S. Susstrunk, "What is the space of spectral sensitivity functions for digital colour cameras?", WACV 2013
(28-camera database, 400-720 nm). `optics.py` learns the centres, widths and a low-rank correction per camera,
because the true curve of the Nexus 6P / iPhone sensors in these datasets is not published.

The table runs 400-1000 nm in the 2 nm steps the page tabulates (odd wavelengths are not on the page and are not
invented here; `spectra.py` interpolates onto its own 5 nm grid). The page states no isosbestic wavelengths, so
none are quoted anywhere in this repository as published values.
