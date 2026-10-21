"""Membuat database SQLite SIMULASI 'PT Maju Bersama' (distributor barang), 24 bulan: Jul 2024 - Jun 2026 (+3 bulan pemanasan Apr-Jun 2024 agar saldo piutang awal realistis).
Sengaja dibuat berantakan seperti dunia nyata (duplikat, missing value, label tidak konsisten)."""
import os, sqlite3
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
SEG = {"retail": dict(term=30, qty=(2, 20), disc=0.02, late=3),
       "grosir": dict(term=45, qty=(10, 50), disc=0.05, late=5),
       "distributor_besar": dict(term=60, qty=(40, 160), disc=0.08, late=8)}

# --- customers & products
cust = pd.DataFrame({"customer_id": range(1, 141)})
cust["name"] = [f"Pelanggan {i:03d}" for i in cust.customer_id]
cust["segment"] = rng.choice(list(SEG), 140, p=[0.5, 0.35, 0.15])
cust["credit_term_days"] = cust.segment.map({k: v["term"] for k, v in SEG.items()})
dirty = rng.choice(140, 10, replace=False)                                   # label tidak konsisten
cust.loc[dirty, "segment"] = cust.loc[dirty, "segment"].str.title() + " "
prod = pd.DataFrame({"product_id": range(1, 41), "category": rng.choice(["makanan", "minuman", "rumah_tangga", "kesehatan"], 40)})
prod["base_cost"] = rng.uniform(20_000, 250_000, 40).round(-2)
prod["base_price"] = (prod.base_cost * rng.uniform(1.33, 1.55, 40)).round(-2)

# --- orders, payments
orders, pays, oid = [], [], 1
seg_of = cust.set_index("customer_id").segment.str.strip().str.lower()
for mi, m in enumerate(pd.period_range("2024-04", "2026-06", freq="M"), start=-3):   # 3 bulan pemanasan
    n = int(300 * 1.012 ** mi * rng.uniform(0.95, 1.05))
    w = cust.customer_id.map(seg_of).map({"retail": 1.0, "grosir": 1.0, "distributor_besar": 1 + 1.8 * mi / 23}).values
    cids = rng.choice(cust.customer_id, n, p=w / w.sum())
    for cid in cids:
        s = SEG[seg_of[cid]]; p = prod.sample(1, random_state=int(rng.integers(1e9))).iloc[0]
        d = m.to_timestamp() + pd.Timedelta(days=int(rng.integers(0, m.days_in_month)))
        disc = max(0, s["disc"] + 0.0016 * mi + rng.normal(0, 0.01))
        orders.append((oid, cid, int(p.product_id), d.date().isoformat(), int(rng.integers(*s["qty"])),
                       round(p.base_price * (1 + 0.004 * mi), -2), round(disc, 4), round(p.base_cost * (1 + 0.007 * mi), -2)))
        late = rng.gamma(2, s["late"] * (1 + 1.8 * mi / 23))                      # keterlambatan bayar makin parah
        paid = d + pd.Timedelta(days=int(SEG[seg_of[cid]]["term"] + late - s["late"] * 2))
        bad = seg_of[cid] == "distributor_besar" and rng.random() < 0.02 + 0.06 * mi / 23
        if paid <= pd.Timestamp("2026-06-30") and not bad:
            pays.append((oid, paid.date().isoformat()))
        oid += 1
orders = pd.DataFrame(orders, columns=["order_id", "customer_id", "product_id", "order_date", "qty", "unit_price", "discount_pct", "unit_cost"])
pays = pd.DataFrame(pays, columns=["order_id", "paid_date"])
pays["amount"] = pays.order_id.map((orders.set_index("order_id").eval("qty*unit_price*(1-discount_pct)")).round(0))
orders.loc[rng.choice(len(orders), int(len(orders) * 0.02), replace=False), "discount_pct"] = np.nan   # missing
orders = pd.concat([orders, orders.sample(25, random_state=1)], ignore_index=True)                    # duplikat

# --- biaya operasional per bulan
rev_m = (orders.drop_duplicates("order_id").assign(m=lambda x: x.order_date.str[:7])
         .eval("qty*unit_price*(1-discount_pct.fillna(0.05))").groupby(orders.drop_duplicates("order_id").order_date.str[:7]).sum())
exp = []
for mi, m in enumerate(pd.period_range("2024-07", "2026-06", freq="M")):
    r = rev_m[str(m)]
    for cat, amt in [("gaji", 180e6 * (1 + 0.02 * mi)), ("logistik", 0.035 * r), ("sewa", 60e6), ("pemasaran", 35e6 * (1 + 0.03 * mi))]:
        exp.append((str(m), cat, round(amt * rng.uniform(0.97, 1.03))))
exp = pd.DataFrame(exp, columns=["month", "category", "amount"])

os.makedirs("../data", exist_ok=True)
con = sqlite3.connect("../data/maju_bersama.db")
for name, d in [("customers", cust), ("products", prod.drop(columns=["base_cost", "base_price"])), ("orders", orders), ("payments", pays), ("expenses", exp)]:
    d.to_sql(name, con, if_exists="replace", index=False)
con.close()
print(f"Database dibuat: {len(cust)} pelanggan, {len(orders)} baris order, {len(pays)} pembayaran, {len(exp)} baris biaya")
