# Subgroup note

Source of demographics: the Hoffman 2022 repository (`ubicomplab/oximetry-phone-cam-data`), `data/info/{Left,Right}/*.info`, field `subject_info` (ethnicity, age, sex, skintone hex). These are the only per-person demographics the dataset documents; the README gives none, and the repository has no Fitzpatrick or Monk score (only a skin-tone hex colour). Left and right-hand files agree for every person: True. The table below copies them verbatim and puts the TINY per-person metrics (default threshold 0.5, windows t >= 60 s, leave-one-person-out) next to them.

| person | sid | age | sex | ethnicity | skintone_hex | n_windows | base_rate_hypoxemic | TINY_acc | TINY_sens | TINY_spec | abstain_rate_band010 | acc_on_answered_band010 | clock_acc |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 100001 | 31 | Male | White | #FAF3EB | 1022 | 0.526 | 0.917 | 0.944 | 0.886 | 0.223 | 0.995 | 0.963 |
| 2 | 100002 | 34 | Male | African American/Black | #32210F | 1053 | 0.536 | 0.886 | 0.874 | 0.900 | 0.136 | 0.952 | 0.954 |
| 3 | 100003 | 23 | Female | White | #F1E1D1 | 998 | 0.571 | 0.850 | 0.884 | 0.804 | 0.124 | 0.884 | 0.818 |
| 4 | 100004 | 20 | Male | White | #F1E1D1 | 947 | 0.511 | 0.775 | 0.769 | 0.782 | 0.220 | 0.829 | 0.911 |
| 5 | 100005 | 24 | Female | White | #DEC09C | 859 | 0.633 | 0.917 | 0.925 | 0.905 | 0.068 | 0.954 | 0.739 |
| 6 | 100006 | 23 | Female | White | #F1E1D1 | 765 | 0.711 | 0.877 | 0.866 | 0.905 | 0.103 | 0.929 | 0.539 |

**Dark-skinned person.** Person 2 (sid 100002) is the only participant documented as African American/Black, with the darkest skin-tone hex (#32210F); the other five are documented as White (hex #DEC09C, #F1E1D1, #FAF3EB). TINY on this person: accuracy 0.886, sensitivity 0.874, specificity 0.900, abstain-rate 0.136 at band 0.10. For comparison the other five people: accuracy 0.78 to 0.92.

**What this does and does not show.** The groups contain 1 Black participant, 3 women and 3 men aged 20 to 34 (all adults); person 4 (the worst-performing person, accuracy 0.775) is a 20-year-old White man with a skin-tone hex of #F1E1D1. With only one Black participant and five White participants, per-person differences cannot be attributed to skin tone, sex or age, and no subgroup or fairness claim is made. Not documented anywhere in the repository: BMI, hand size, nail polish, perfusion at rest, medications, or any clinical history. Dataset-wide caveats: one phone (Nexus 6P), fingertip only, controlled FiO2 protocol.
