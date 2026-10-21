"""CRISP-DM tahap 3-6: Preparation -> Modeling -> Evaluation -> keputusan.
Rantai yang dijelaskan pemateri:  DATA -> INFORMASI -> INSIGHT -> KEPUTUSAN."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from common import load_raw, clean, monthly_kpi, END

t = load_raw(); df, exp = clean(t); k = monthly_kpi(df, exp)
Y1, Y2 = k.iloc[:12], k.iloc[12:]
juta = lambda x: f"Rp{x / 1e6:,.0f} jt"
pct = lambda a, b: (b - a) / abs(a) * 100

# ===================== 1. INFORMASI (deskriptif): apa yang terjadi? =====================
print("=" * 66, "\n1. INFORMASI: kinerja Tahun 1 (Jul24-Jun25) vs Tahun 2 (Jul25-Jun26)\n" + "=" * 66)
tab = pd.DataFrame({"Tahun 1": [Y1.revenue.sum(), Y1.net_profit.sum(), Y1.op_cash_flow.sum(), Y1.ar_end.iloc[-1]],
                    "Tahun 2": [Y2.revenue.sum(), Y2.net_profit.sum(), Y2.op_cash_flow.sum(), Y2.ar_end.iloc[-1]]},
                   index=["Sales revenue", "Net profit", "Operating cash flow", "Piutang akhir periode"])
tab["Perubahan %"] = [pct(a, b) for a, b in zip(tab["Tahun 1"], tab["Tahun 2"])]
print(tab.assign(**{"Tahun 1": tab["Tahun 1"].map(juta), "Tahun 2": tab["Tahun 2"].map(juta)})
      .to_string(formatters={"Perubahan %": "{:+.1f}%".format}))

# ===================== 2. INSIGHT (diagnostik): kenapa terjadi? =====================
print("\n" + "=" * 66, "\n2. INSIGHT: mengapa penjualan naik tapi laba & kas turun?\n" + "=" * 66)
d1, d2 = df[(df.order_date >= "2024-07-01") & (df.order_date <= "2025-06-30")], df[df.order_date > "2025-06-30"]
print(f"  Gross margin   : {Y1.gross_profit.sum() / Y1.revenue.sum():.1%} -> {Y2.gross_profit.sum() / Y2.revenue.sum():.1%}")
print(f"  Rata2 diskon   : {d1.discount_pct.mean():.1%} -> {d2.discount_pct.mean():.1%}")
print(f"  Biaya operasional: {juta(Y1.opex.sum())} -> {juta(Y2.opex.sum())} ({pct(Y1.opex.sum(), Y2.opex.sum()):+.1f}%)")
mix = pd.DataFrame({"T1": d1.groupby("segment").revenue.sum() / d1.revenue.sum(), "T2": d2.groupby("segment").revenue.sum() / d2.revenue.sum()})
print("  Porsi revenue per segmen:\n" + (mix * 100).round(1).to_string(float_format="{:.1f}%".format).replace("\n", "\n    "))
open_inv = df[df.paid_date.isna()].copy()
open_inv["overdue_days"] = (END - open_inv.due_date).dt.days
open_inv["bucket"] = pd.cut(open_inv.overdue_days, [-9999, 0, 30, 60, 9999], labels=["belum jatuh tempo", "telat 1-30 hr", "telat 31-60 hr", "telat >60 hr"])
aging = open_inv.pivot_table(index="segment", columns="bucket", values="revenue", aggfunc="sum", observed=False).fillna(0)
print("\n  Umur piutang per 30-Jun-2026 (juta Rp):\n    " + (aging / 1e6).round(0).astype(int).to_string().replace("\n", "\n    "))
dso = open_inv.revenue.sum() / Y2.revenue.iloc[-3:].sum() * 90
print(f"\n  DSO (rata2 hari penagihan): {dso:.0f} hari")
big = (mix.loc["distributor_besar", "T2"] - mix.loc["distributor_besar", "T1"]) * 100
print(f"\n  >> Insight: revenue naik terutama dari distributor besar (+{big:.0f} poin porsi) yang bertempo bayar panjang,")
print("     diberi diskon lebih besar, dan makin telat membayar. Akibatnya margin & kas tertekan, piutang menumpuk.")

# ===================== 3. PREDIKTIF: invoice mana yang berisiko telat? =====================
print("\n" + "=" * 66, "\n3. PEMODELAN: prediksi risiko telat bayar (>14 hari lewat jatuh tempo)\n" + "=" * 66)
lab = df[df.order_date <= "2026-03-31"].copy()
lab["days_late"] = np.where(lab.paid_date.notna(), (lab.paid_date - lab.due_date).dt.days, (END - lab.due_date).dt.days)
lab = lab[lab.paid_date.notna() | (lab.days_late > 14)]                       # buang yang masih menggantung
lab["late"] = (lab.days_late > 14).astype(int)
feat = lambda d: pd.get_dummies(d[["segment"]], dtype=float).reindex(columns=["segment_retail", "segment_grosir", "segment_distributor_besar"], fill_value=0).assign(
    amount=d.revenue / 1e6, disc=d.discount_pct, month_idx=(d.order_date.dt.year - 2024) * 12 + d.order_date.dt.month)
Xtr, Xte, ytr, yte = train_test_split(feat(lab), lab.late, test_size=0.25, stratify=lab.late, random_state=42)
mu, sd = Xtr.mean(), Xtr.std() + 1e-9
clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
print(f"  Data berlabel: {len(lab)} invoice, proporsi telat {lab.late.mean():.1%} | ROC-AUC test = {roc_auc_score(yte, clf.predict_proba((Xte - mu) / sd)[:, 1]):.3f}")
open_inv["risk"] = clf.predict_proba((feat(open_inv) - mu) / sd)[:, 1]
top = open_inv.assign(exposure=open_inv.revenue * open_inv.risk).groupby("name").agg(
    segmen=("segment", "first"), piutang=("revenue", "sum"), risiko=("risk", "mean"), exposure=("exposure", "sum")).nlargest(5, "exposure")
print("  5 pelanggan prioritas penagihan (piutang x risiko):")
print("    " + top.assign(piutang=lambda x: (x.piutang / 1e6).round(0).astype(int), risiko=lambda x: x.risiko.round(2)).drop(columns="exposure").to_string().replace("\n", "\n    "))

# ===================== 4. KEPUTUSAN =====================
print("\n" + "=" * 66, "\n4. KEPUTUSAN yang direkomendasikan\n" + "=" * 66)
tot_over = open_inv[open_inv.overdue_days > 0].revenue.sum()
print(f"  1) Tagih prioritas: piutang lewat jatuh tempo total {juta(tot_over)}; mulai dari 5 pelanggan di atas.")
print("  2) Perketat kredit distributor besar: kurangi tempo bayar / minta uang muka / batasi limit kredit.")
print(f"  3) Tinjau kebijakan diskon (rata2 {d1.discount_pct.mean():.1%} -> {d2.discount_pct.mean():.1%}): beri diskon hanya untuk pembayaran cepat.")
print("  4) Tinjau harga jual vs kenaikan HPP; margin bruto turun karena harga naik lebih lambat dari biaya.")
print("  Catatan: ini hasil dari DATA SIMULASI. Keputusan nyata perlu informasi eksternal & pengalaman (kata pemateri).")

# ===================== Dashboard =====================
x = k.index.to_timestamp()
fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
ax[0, 0].plot(x, k.revenue / 1e6, label="Revenue"); ax[0, 0].plot(x, k.net_profit / 1e6, label="Net profit"); ax[0, 0].set_title("Revenue vs Net Profit (juta Rp)"); ax[0, 0].legend()
ax[0, 1].plot(x, k.ar_end / 1e6, color="tab:red", label="Piutang akhir bulan"); ax[0, 1].plot(x, k.op_cash_flow / 1e6, color="tab:green", label="Operating cash flow"); ax[0, 1].axhline(0, color="gray", lw=.6); ax[0, 1].set_title("Piutang & Arus Kas Operasi (juta Rp)"); ax[0, 1].legend()
(aging / 1e6).plot(kind="bar", stacked=True, ax=ax[1, 0], colormap="RdYlGn_r"); ax[1, 0].set_title("Umur piutang per segmen (juta Rp)"); ax[1, 0].tick_params(axis="x", rotation=0)
ax[1, 1].plot(x, k.gross_margin * 100, color="tab:purple"); ax[1, 1].set_title("Gross margin (%)")
for a in ax.flat: a.grid(alpha=.3)
for a in (ax[0, 0], ax[0, 1], ax[1, 1]): a.tick_params(axis="x", rotation=30)
plt.suptitle("PT Maju Bersama: dashboard analisis (data simulasi)"); plt.tight_layout(); plt.savefig("../results/dashboard.png", dpi=130)
tab.to_csv("../results/kpi_tahun1_vs_tahun2.csv"); print("\nTersimpan: results/dashboard.png, results/kpi_tahun1_vs_tahun2.csv")
