"""CRISP-DM tahap 2: Data Understanding. Kenali tabel, granularitas, rentang waktu, dan masalah kualitas data."""
import pandas as pd
from common import load_raw

t = load_raw()
print("== Ukuran tiap tabel ==")
for n, d in t.items():
    print(f"  {n:10s} {d.shape[0]:6d} baris x {d.shape[1]} kolom  | kolom: {', '.join(d.columns)}")

o = t["orders"]
print("\n== Granularitas & rentang waktu ==")
print(f"  1 baris orders = 1 order line (produk x pelanggan x tanggal). Rentang: {o.order_date.min()} s.d. {o.order_date.max()}")

print("\n== Masalah kualitas data ==")
print(f"  Duplikat order_id        : {o.order_id.duplicated().sum()}")
print(f"  Missing discount_pct     : {o.discount_pct.isna().sum()} ({o.discount_pct.isna().mean():.1%})")
seg = t["customers"].segment
print(f"  Label segmen unik (mentah): {sorted(seg.unique())}  -> tidak konsisten (spasi/huruf besar)")
print(f"  Order tanpa pembayaran   : {(~o.order_id.isin(t['payments'].order_id)).sum()} (belum dibayar / piutang)")
print("\nKeputusan: duplikat dibuang, missing discount diisi median, label segmen dinormalkan (lihat common.clean).")
