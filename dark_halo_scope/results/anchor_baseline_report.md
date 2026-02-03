# Anchor Baseline Evaluation Report

**Generated**: 2026-02-02 17:02:42.830074

## Summary

| Metric | Value |
|--------|-------|
| Known Lenses Total | 68 |
| Known Lenses in Footprint | 68 |
| Hard Negatives Total | 20 |
| Hard Negatives in Footprint | 14 |

## Performance at Key Thresholds

| Threshold | Recall (Known Lenses) | Contamination (Hard Neg) |
|-----------|----------------------|--------------------------|
| 0.5 | 2.9% (2/68) | 7.1% (1/14) |
| 0.7 | 1.5% (1/68) | 7.1% (1/14) |
| 0.9 | 0.0% (0/68) | 7.1% (1/14) |

## Score Distributions

### Known Lenses
- Mean p_lens: 0.232
- Median p_lens: 0.209
- Std p_lens: 0.126

### Hard Negatives
- Mean p_lens: 0.323
- Median p_lens: 0.290
- Std p_lens: 0.223

## Interpretation

**Gating Rule**: For viable real-world performance:
- Recall on known lenses should be > 50% at threshold 0.5
- Contamination on hard negatives should be < 20% at threshold 0.5

**Current Status**: ❌ FAIL - Focus on data/simulation improvements before model iteration

## Individual Results

### Top Detected Known Lenses (by p_lens)
- SDSSJ0912+0029: p_lens = 0.7488 (θ_e = 1.63")
- SDSSJ1205+4910: p_lens = 0.5824 (θ_e = 1.22")
- SDSSJ2300+0022: p_lens = 0.4806 (θ_e = 1.24")
- BELLSJ1401+3845: p_lens = 0.4404 (θ_e = 1.35")
- BELLSJ0903+4116: p_lens = 0.4158 (θ_e = 1.48")
- SDSSJ0936+0913: p_lens = 0.4097 (θ_e = 1.09")
- SDSSJ0728+3835: p_lens = 0.3940 (θ_e = 1.25")
- BELLSJ0830+5116: p_lens = 0.3784 (θ_e = 1.41")
- SDSSJ1636+4707: p_lens = 0.3602 (θ_e = 1.09")
- SDSSJ0029-0055: p_lens = 0.3505 (θ_e = 0.96")

### Missed Known Lenses (p_lens < 0.5)
- SDSSJ0029-0055: p_lens = 0.3505 (θ_e = 0.96")
- SDSSJ0037-0942: p_lens = 0.1521 (θ_e = 1.53")
- SDSSJ0216-0813: p_lens = 0.1026 (θ_e = 1.16")
- SDSSJ0252+0039: p_lens = 0.1824 (θ_e = 1.04")
- SDSSJ0330-0020: p_lens = 0.2723 (θ_e = 1.1")
- SDSSJ0728+3835: p_lens = 0.3940 (θ_e = 1.25")
- SDSSJ0737+3216: p_lens = 0.1631 (θ_e = 0.98")
- SDSSJ0822+2652: p_lens = 0.2328 (θ_e = 1.17")
- SDSSJ0936+0913: p_lens = 0.4097 (θ_e = 1.09")
- SDSSJ0956+5100: p_lens = 0.2898 (θ_e = 1.33")
- SDSSJ0959+0410: p_lens = 0.2539 (θ_e = 0.99")
- SDSSJ1016+3859: p_lens = 0.3149 (θ_e = 1.09")
- SDSSJ1020+1122: p_lens = 0.3064 (θ_e = 1.2")
- SDSSJ1023+4230: p_lens = 0.2034 (θ_e = 1.41")
- SDSSJ1029+0420: p_lens = 0.1970 (θ_e = 1.01")
- SDSSJ1106+5228: p_lens = 0.3111 (θ_e = 1.23")
- SDSSJ1112+0826: p_lens = 0.1762 (θ_e = 1.49")
- SDSSJ1134+6027: p_lens = 0.1952 (θ_e = 1.1")
- SDSSJ1142+1001: p_lens = 0.0918 (θ_e = 0.98")
- SDSSJ1143-0144: p_lens = 0.1766 (θ_e = 1.68")
- SDSSJ1153+4612: p_lens = 0.2025 (θ_e = 1.05")
- SDSSJ1204+0358: p_lens = 0.1185 (θ_e = 1.31")
- SDSSJ1213+6708: p_lens = 0.2305 (θ_e = 1.42")
- SDSSJ1218+0830: p_lens = 0.1453 (θ_e = 1.45")
- SDSSJ1250+0523: p_lens = 0.1922 (θ_e = 1.13")
- SDSSJ1402+6321: p_lens = 0.1810 (θ_e = 1.35")
- SDSSJ1403+0006: p_lens = 0.2076 (θ_e = 0.83")
- SDSSJ1416+5136: p_lens = 0.0824 (θ_e = 1.37")
- SDSSJ1420+6019: p_lens = 0.1076 (θ_e = 1.04")
- SDSSJ1430+4105: p_lens = 0.1032 (θ_e = 1.52")
- SDSSJ1432+6317: p_lens = 0.0813 (θ_e = 1.25")
- SDSSJ1436-0000: p_lens = 0.2159 (θ_e = 1.12")
- SDSSJ1443+0304: p_lens = 0.2311 (θ_e = 0.81")
- SDSSJ1451-0239: p_lens = 0.1741 (θ_e = 1.04")
- SDSSJ1525+3327: p_lens = 0.3120 (θ_e = 1.31")
- SDSSJ1531-0105: p_lens = 0.0416 (θ_e = 1.71")
- SDSSJ1538+5817: p_lens = 0.0687 (θ_e = 1.0")
- SDSSJ1621+3931: p_lens = 0.1670 (θ_e = 1.29")
- SDSSJ1627-0053: p_lens = 0.3181 (θ_e = 1.23")
- SDSSJ1630+4520: p_lens = 0.2157 (θ_e = 1.78")
- SDSSJ1636+4707: p_lens = 0.3602 (θ_e = 1.09")
- SDSSJ2238-0754: p_lens = 0.1588 (θ_e = 1.27")
- SDSSJ2300+0022: p_lens = 0.4806 (θ_e = 1.24")
- SDSSJ2303+1422: p_lens = 0.3077 (θ_e = 1.62")
- SDSSJ2321-0939: p_lens = 0.3250 (θ_e = 1.6")
- SDSSJ2341+0000: p_lens = 0.3401 (θ_e = 1.44")
- BELLSJ0747+4448: p_lens = 0.2122 (θ_e = 1.16")
- BELLSJ0801+4727: p_lens = 0.1101 (θ_e = 0.91")
- BELLSJ0830+5116: p_lens = 0.3784 (θ_e = 1.41")
- BELLSJ0847+2348: p_lens = 0.2716 (θ_e = 1.26")
- BELLSJ0903+4116: p_lens = 0.4158 (θ_e = 1.48")
- BELLSJ0918+5104: p_lens = 0.2551 (θ_e = 1.28")
- BELLSJ1014+3920: p_lens = 0.0983 (θ_e = 1.07")
- BELLSJ1110+2808: p_lens = 0.2913 (θ_e = 1.37")
- BELLSJ1159+5820: p_lens = 0.2304 (θ_e = 0.97")
- BELLSJ1221+3806: p_lens = 0.0856 (θ_e = 1.53")
- BELLSJ1226+5457: p_lens = 0.2100 (θ_e = 1.22")
- BELLSJ1318+3942: p_lens = 0.1688 (θ_e = 1.08")
- BELLSJ1349+3612: p_lens = 0.1197 (θ_e = 0.95")
- BELLSJ1401+3845: p_lens = 0.4404 (θ_e = 1.35")
- BELLSJ1522+2910: p_lens = 0.0948 (θ_e = 1.42")
- BELLSJ1541+1812: p_lens = 0.0965 (θ_e = 1.48")
- BELLSJ1545+2748: p_lens = 0.1604 (θ_e = 1.34")
- BELLSJ1601+2138: p_lens = 0.2166 (θ_e = 1.52")
- BELLSJ1611+1705: p_lens = 0.2949 (θ_e = 1.19")
- BELLSJ1631+1854: p_lens = 0.1806 (θ_e = 1.06")

### False Positive Hard Negatives (p_lens > 0.5)
- Merger002: p_lens = 0.9905
