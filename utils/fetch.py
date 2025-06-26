import random

# --- OVII: Option Volume Imbalance Index (Mocked) ---
def get_ovii(symbol="NIFTY"):
    # Simulated volumes and imbalance
    ce = random.randint(12000, 25000)
    pe = random.randint(12000, 25000)
    ovii = (ce - pe) / (ce + pe) if (ce + pe) else 0
    return round(ovii, 4), ce, pe

# --- IVDI: Implied Volatility Divergence Index (Mocked) ---
def get_ivdi(symbol="NIFTY"):
    iv_call = round(random.uniform(14.5, 22.0), 2)
    iv_put = round(random.uniform(13.0, 21.0), 2)
    ivdi = iv_call - iv_put
    return iv_call, iv_put, round(ivdi, 4)
