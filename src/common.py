import sqlite3
import numpy as np
import pandas as pd

DB = "../data/maju_bersama.db"
END = pd.Timestamp("2026-06-30")          # tanggal "hari ini" dalam data


def load_raw():
    con = sqlite3.connect(DB)
    t = {n: pd.read_sql(f"SELECT * FROM {n}", con) for n in ["customers", "products", "orders", "payments", "expenses"]}
    con.close()
    return t


def clean(t):
    """Data cleaning + penggabungan tabel menjadi satu tabel analisis (1 baris = 1 invoice/order line)."""
    c = t["customers"].copy()
    c["segment"] = c["segment"].str.strip().str.lower()                      # 'Grosir ' -> 'grosir'
    o = t["orders"].drop_duplicates(subset="order_id").copy()                # buang duplikat
    o["order_date"] = pd.to_datetime(o["order_date"])
    o["discount_pct"] = o["discount_pct"].fillna(o["discount_pct"].median())  # isi missing value
    p = t["payments"].copy(); p["paid_date"] = pd.to_datetime(p["paid_date"])
    df = (o.merge(c, on="customer_id", how="left").merge(t["products"], on="product_id", how="left")
           .merge(p[["order_id", "paid_date"]], on="order_id", how="left"))
    df["revenue"] = df.qty * df.unit_price * (1 - df.discount_pct)
    df["cogs"] = df.qty * df.unit_cost
    df["month"] = df.order_date.dt.to_period("M")
    df["due_date"] = df.order_date + pd.to_timedelta(df.credit_term_days, unit="D")
    return df, t["expenses"].copy()


def monthly_kpi(df, exp):
    months = pd.period_range("2024-07", "2026-06", freq="M")
    rows = []
    for m in months:
        end = m.to_timestamp(how="end").normalize()
        cur = df[df.month == m]
        cash_in = df[df.paid_date.dt.to_period("M") == m].revenue.sum()
        cogs_paid = df[df.month == m - 1].cogs.sum()                          # HPP dibayar ke pemasok 1 bulan setelah barang dibeli
        opex = exp.loc[exp.month == str(m), "amount"].sum()
        ar = df[(df.order_date <= end) & (df.paid_date.isna() | (df.paid_date > end))].revenue.sum()
        rows.append({"month": m, "revenue": cur.revenue.sum(), "cogs": cur.cogs.sum(), "cogs_paid": cogs_paid, "opex": opex,
                     "cash_in": cash_in, "ar_end": ar})
    k = pd.DataFrame(rows).set_index("month")
    k["gross_profit"] = k.revenue - k.cogs
    k["net_profit"] = k.gross_profit - k.opex
    k["op_cash_flow"] = k.cash_in - k.cogs_paid - k.opex                     # kas masuk dari pelanggan - bayar pemasok - biaya operasional
    k["gross_margin"] = k.gross_profit / k.revenue
    return k
