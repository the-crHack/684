"""Re-derive epsilon for every calibrated noise multiplier with independent
dp_accounting accountants (finer-grained PLD, and RDP). Catches accounting
mistakes before we trust a DP claim, and shows how pessimistic jax_privacy's
default PLD discretisation (1e-3) is at very small epsilon."""
import json

import dp_accounting as dpa
import pandas as pd


def eps_pld(sigma, q, T, delta, disc=1e-4):
    acc = dpa.pld.PLDAccountant(value_discretization_interval=disc)
    acc.compose(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma)), T)
    return acc.get_epsilon(delta)


def eps_rdp(sigma, q, T, delta):
    acc = dpa.rdp.RdpAccountant()
    acc.compose(dpa.PoissonSampledDpEvent(q, dpa.GaussianDpEvent(sigma)), T)
    return acc.get_epsilon(delta)


rows = []
for key, sigma in json.load(open("results/noise_multipliers.json")).items():
    p = dict(kv.split("=") for kv in key.split("|"))
    eps, B, T, n, delta = float(p["eps"]), int(p["B"]), int(p["T"]), int(p["n"]), float(p["delta"])
    if n != 30200 or B * T != 614_400:
        continue  # only the main grid
    q = B / n
    rows.append(dict(target_eps=eps, B=B, T=T, q=q, sigma=sigma, noise_std_mean_grad=sigma / B,
                     eps_pld_fine=eps_pld(sigma, q, T, delta) if eps <= 100 else float("nan"),
                     eps_rdp=eps_rdp(sigma, q, T, delta)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).sort_values(["target_eps", "B"]).to_csv("results/accounting_check.csv", index=False)
