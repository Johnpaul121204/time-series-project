#!/usr/bin/env python3
"""
Advanced Time Series Forecasting - Single-file project

Save this script as advanced_timeseries_project.py

Requirements (install in your environment):
    pip install pandas numpy matplotlib prophet torch scikit-learn tqdm

Usage examples:
    # generate data, run both pipelines and create report
    python advanced_timeseries_project.py --out data/generated_daily.csv --years 4 --run_prophet --run_nbeats

    # run only data generation
    python advanced_timeseries_project.py --out data/generated_daily.csv --years 4

The script will create:
    - data/generated_daily.csv
    - models/prophet_fold_{i}.json
    - models/nbeats_fold_{i}.pth
    - report.md
"""
import os
import json
import argparse
from datetime import datetime
import numpy as np
import pandas as pd

# -------------------------
# Metrics
# -------------------------
def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def smape(y_true, y_pred, eps=1e-8):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    return float(100.0 * np.mean(np.abs(y_pred - y_true) / (denom + eps)))

# -------------------------
# Data generation
# -------------------------
def generate_series(start='2018-01-01', years=4, seed=42):
    np.random.seed(seed)
    n_days = int(365.25 * years)
    dates = pd.date_range(start=start, periods=n_days, freq='D')
    t = np.arange(n_days).astype(float)

    # yearly seasonality: harmonics
    yearly = 10 * np.sin(2 * np.pi * t / 365.25) + 3 * np.cos(4 * np.pi * t / 365.25)

    # weekly seasonality: small weekend bump
    weekday = np.array([0.0 if d.weekday() < 5 else 2.5 for d in dates])

    # base trend + piecewise changepoints
    base_trend = 0.02 * t
    cp1 = int(n_days * 0.35)
    cp2 = int(n_days * 0.70)
    base_trend[cp1:] += 0.5 * (t[cp1:] - cp1) / n_days
    base_trend[cp2:] -= 1.2 * (t[cp2:] - cp2) / n_days

    # holiday effects (three repeated dates per year)
    holidays = []
    start_year = pd.Timestamp(start).year
    for y in range(start_year, start_year + years + 1):
        holidays.append(f'{y}-01-26')
        holidays.append(f'{y}-08-15')
        holidays.append(f'{y}-12-25')
    hol_series = np.zeros(n_days)
    dates_str = dates.strftime('%Y-%m-%d').tolist()
    for h in holidays:
        if h in dates_str:
            idx = dates_str.index(h)
            # small 3-day bump centered on holiday
            for offset, weight in zip([-1,0,1],[0.6,1.0,0.6]):
                k = idx + offset
                if 0 <= k < n_days:
                    hol_series[k] += 8.0 * weight

    # noise + occasional outliers
    noise = np.random.normal(scale=1.5, size=n_days)
    outlier_idx = np.random.choice(n_days, size=max(1,int(n_days*0.003)), replace=False)
    noise[outlier_idx] += np.random.choice([15, -12], size=len(outlier_idx))

    y = 50 + base_trend + yearly + weekday + hol_series + noise
    df = pd.DataFrame({'ds': dates, 'y': y})
    return df

# -------------------------
# Rolling-origin CV helper
# -------------------------
def rolling_origin_folds(df_len, initial_train_days, horizon_days, step_days=None):
    if step_days is None:
        step_days = horizon_days
    folds = []
    train_end = initial_train_days
    max_train_end = df_len - horizon_days
    while train_end <= max_train_end:
        train_idx = (0, train_end)  # [0,train_end)
        test_idx = (train_end, train_end + horizon_days)
        folds.append((train_idx, test_idx))
        train_end += step_days
    return folds

# -------------------------
# Prophet pipeline
# -------------------------
def run_prophet_cv(df, initial_train_days=365*2, horizon_days=90, step_days=90, outdir='models'):
    from prophet import Prophet
    os.makedirs(outdir, exist_ok=True)
    folds = rolling_origin_folds(len(df), initial_train_days, horizon_days, step_days)
    results = []
    # prepare holiday DataFrame for Prophet
    years = sorted(df['ds'].dt.year.unique())
    rows = []
    for y in years:
        for d in ['01-26','08-15','12-25']:
            rows.append({'ds': pd.to_datetime(f'{y}-{d}'), 'holiday': 'special_day'})
    holidays = pd.DataFrame(rows)

    for i, ((t0, t1), (v0, v1)) in enumerate(folds):
        train = df.iloc[t0:t1].copy()
        test = df.iloc[v0:v1].copy()
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False,
                    changepoint_prior_scale=0.05, seasonality_mode='additive')
        # attach custom holidays
        m.holidays = holidays
        m.fit(train)
        future = m.make_future_dataframe(periods=len(test), freq='D')
        fcst = m.predict(future)
        preds = fcst[['ds','yhat']].iloc[-len(test):].set_index('ds')
        y_true = test.set_index('ds')['y'].values
        y_pred = preds['yhat'].values
        fold_metrics = {'fold': i, 'MAE': mae(y_true,y_pred), 'RMSE': rmse(y_true,y_pred), 'sMAPE': smape(y_true,y_pred)}
        results.append(fold_metrics)
        # save simple metadata (hyperparams) and Prophet forecast frame
        meta = {
            'fold': i,
            'changepoint_prior_scale': 0.05,
            'seasonality_mode': 'additive',
            'yearly_seasonality': True,
            'weekly_seasonality': True
        }
        with open(os.path.join(outdir, f'prophet_fold_{i}.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        # save predictions CSV for fold for inspection
        preds.reset_index().to_csv(os.path.join(outdir, f'prophet_fold_{i}_preds.csv'), index=False)
        print(f"[Prophet] Fold {i} metrics: {fold_metrics}")
    return results

# -------------------------
# Simple N-BEATS implementation (PyTorch)
# -------------------------
def run_nbeats_cv(df, initial_train_days=365*2, horizon_days=90, step_days=90, outdir='models',
                  input_len=90, epochs=30, lr=1e-3, device='cpu'):
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from sklearn.preprocessing import StandardScaler
    os.makedirs(outdir, exist_ok=True)

    class WindowDataset(Dataset):
        def __init__(self, series, input_len, output_len):
            self.series = series.astype(np.float32)
            self.input_len = input_len
            self.output_len = output_len
            self.indices = np.arange(len(series) - (input_len + output_len) + 1)
        def __len__(self):
            return max(0, len(self.indices))
        def __getitem__(self, idx):
            i = self.indices[idx]
            x = self.series[i:i+self.input_len]
            y = self.series[i+self.input_len:i+self.input_len+self.output_len]
            return x, y

    class Block(nn.Module):
        def __init__(self, input_dim, theta_dim, hidden_units, n_layers=4):
            super().__init__()
            layers = []
            d = input_dim
            for _ in range(n_layers):
                layers.append(nn.Linear(d, hidden_units))
                layers.append(nn.ReLU())
                d = hidden_units
            layers.append(nn.Linear(hidden_units, theta_dim))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    class NBeats(nn.Module):
        def __init__(self, input_len=90, output_len=90, n_blocks=3, hidden_units=128):
            super().__init__()
            self.blocks = nn.ModuleList([Block(input_len, output_len, hidden_units) for _ in range(n_blocks)])
            self.final = nn.Linear(output_len * n_blocks, output_len)
        def forward(self, x):
            # x shape: (B, input_len)
            outs = []
            for b in self.blocks:
                outs.append(b(x))
            concat = torch.cat(outs, dim=1)
            return self.final(concat)

    y = df['y'].values
    folds = rolling_origin_folds(len(df), initial_train_days, horizon_days, step_days)
    results = []

    for i, ((t0, t1), (v0, v1)) in enumerate(folds):
        train = y[t0:t1]
        test = y[v0:v1]
        # scaling
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train.reshape(-1,1)).flatten()
        test_scaled = scaler.transform(test.reshape(-1,1)).flatten()
        # prepare datasets: we allow windows inside train; for validation we'll build windows covering last part
        train_ds = WindowDataset(train_scaled, input_len, len(test))
        if len(train_scaled) < input_len:
            print(f"[N-BEATS] Fold {i} skipped: training length {len(train_scaled)} < input_len {input_len}")
            continue
        # validation windows: create concatenation of last input_len from train + test so at least 1 window
        val_series = np.concatenate([train_scaled[-input_len:], test_scaled])
        val_ds = WindowDataset(val_series, input_len, len(test))
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

        model = NBeats(input_len=input_len, output_len=len(test), n_blocks=3, hidden_units=128).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        # training loop
        for epoch in range(epochs):
            model.train()
            running = 0.0
            seen = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                opt.zero_grad()
                pred = model(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                running += loss.item() * xb.size(0)
                seen += xb.size(0)
            train_loss = running / max(1, seen)
            # val loss
            model.eval()
            running_v = 0.0
            seen_v = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    pred = model(xb)
                    running_v += loss_fn(pred, yb).item() * xb.size(0)
                    seen_v += xb.size(0)
            val_loss = running_v / max(1, seen_v) if seen_v>0 else None
            if epoch % 10 == 0 or epoch == epochs-1:
                print(f"[N-BEATS] Fold {i} Epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        # produce forecast: create last window from train
        model.eval()
        with torch.no_grad():
            last_window = torch.tensor(train_scaled[-input_len:], dtype=torch.float32).unsqueeze(0).to(device)
            pred_scaled = model(last_window).cpu().numpy().flatten()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()
        fold_metrics = {'fold': i, 'MAE': mae(test, pred), 'RMSE': rmse(test, pred), 'sMAPE': smape(test, pred)}
        results.append(fold_metrics)
        # save weights and scaler params
        torch.save(model.state_dict(), os.path.join(outdir, f'nbeats_fold_{i}.pth'))
        meta = {'fold': i, 'input_len': input_len, 'output_len': len(test), 'n_blocks':3, 'hidden_units':128, 'lr':lr, 'epochs':epochs}
        with open(os.path.join(outdir, f'nbeats_fold_{i}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        # save predictions CSV
        pd.DataFrame({'ds': df['ds'].iloc[v0:v1].values, 'y_true': test, 'y_pred': pred}).to_csv(os.path.join(outdir, f'nbeats_fold_{i}_preds.csv'), index=False)
        print(f"[N-BEATS] Fold {i} metrics: {fold_metrics}")

    return results

# -------------------------
# Report writing
# -------------------------
def write_report(report_path, df, prophet_results, nbeats_results, meta_info):
    lines = []
    lines.append("# Advanced Time Series Forecasting — Prophet vs N-BEATS\n")
    lines.append(f"Generated: {datetime.utcnow().isoformat()} UTC\n")
    lines.append("## Dataset\n")
    lines.append(f"- Rows: {len(df)} (from {df['ds'].iloc[0].date()} to {df['ds'].iloc[-1].date()})\n")
    lines.append("- Data generated programmatically with yearly & weekly seasonality, two trend changepoints, holidays, noise and outliers.\n")
    lines.append("## Cross-validation strategy\n")
    lines.append(f"- Rolling-origin CV: initial_train_days={meta_info['initial_train_days']}, horizon_days={meta_info['horizon_days']}, step_days={meta_info['step_days']}\n")
    lines.append("## Prophet results (per fold)\n")
    if prophet_results:
        for r in prophet_results:
            lines.append(f"- Fold {r['fold']}: MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, sMAPE={r['sMAPE']:.4f}\n")
    else:
        lines.append("- Prophet was not run.\n")
    lines.append("## N-BEATS results (per fold)\n")
    if nbeats_results:
        for r in nbeats_results:
            lines.append(f"- Fold {r['fold']}: MAE={r['MAE']:.4f}, RMSE={r['RMSE']:.4f}, sMAPE={r['sMAPE']:.4f}\n")
    else:
        lines.append("- N-BEATS was not run.\n")
    # aggregate
    def agg(rs):
        if not rs: return None
        maes = [x['MAE'] for x in rs]
        rmses = [x['RMSE'] for x in rs]
        smapes = [x['sMAPE'] for x in rs]
        return (np.mean(maes), np.mean(rmses), np.mean(smapes))
    pa = agg(prophet_results)
    na = agg(nbeats_results)
    lines.append("\n## Aggregate (mean across folds)\n")
    if pa:
        lines.append(f"- Prophet mean MAE={pa[0]:.4f}, RMSE={pa[1]:.4f}, sMAPE={pa[2]:.4f}\n")
    if na:
        lines.append(f"- N-BEATS mean MAE={na[0]:.4f}, RMSE={na[1]:.4f}, sMAPE={na[2]:.4f}\n")
    lines.append("\n## Final hyperparameters / notes\n")
    lines.append("- Prophet: changepoint_prior_scale=0.05, seasonality_mode='additive', custom holidays (Jan26, Aug15, Dec25).\n")
    lines.append("- N-BEATS: simple fully-connected blocks, input_len=90 (modifiable), epochs configurable.\n")
    lines.append("\n## Files produced\n")
    lines.append("- data/generated_daily.csv\n")
    lines.append("- models/prophet_fold_*.json and prophet_fold_*_preds.csv\n")
    lines.append("- models/nbeats_fold_*.pth and nbeats_fold_*_preds.csv\n")
    lines.append("\n## Next steps / improvements\n")
    lines.append("1. Run hyperparameter tuning (changepoint_prior_scale, N-BEATS blocks/hidden units, lr, epochs).\n")
    lines.append("2. Use early stopping and longer training for N-BEATS; implement a more faithful N-BEATS with backcast/forecast decomposition.\n")
    lines.append("3. Produce plots: Prophet components, forecast vs actual per fold, residual ACF/PACF.\n")

    with open(report_path, 'w') as f:
        f.writelines(line if line.endswith('\n') else line + '\n' for line in lines)
    print(f"Wrote report to {report_path}")

# -------------------------
# Main CLI
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/generated_daily.csv", help="output CSV path for generated data")
    parser.add_argument("--years", type=int, default=4, help="years of daily data to generate")
    parser.add_argument("--start", default="2018-01-01", help="start date for generated data")
    parser.add_argument("--run_prophet", action='store_true', help="run Prophet CV")
    parser.add_argument("--run_nbeats", action='store_true', help="run N-BEATS CV")
    parser.add_argument("--initial_train_days", type=int, default=365*2)
    parser.add_argument("--horizon_days", type=int, default=90)
    parser.add_argument("--step_days", type=int, default=90)
    parser.add_argument("--device", default="cpu", help="device for PyTorch (cpu or cuda)")
    parser.add_argument("--models_dir", default="models", help="directory to save models and preds")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    os.makedirs(args.models_dir, exist_ok=True)

    print("Generating data...")
    df = generate_series(start=args.start, years=args.years)
    df.to_csv(args.out, index=False)
    print(f"Wrote generated data to {args.out} ({len(df)} rows)")

    prophet_results = []
    nbeats_results = []
    if args.run_prophet:
        print("Running Prophet cross-validation...")
        prophet_results = run_prophet_cv(df, initial_train_days=args.initial_train_days,
                                         horizon_days=args.horizon_days, step_days=args.step_days,
                                         outdir=args.models_dir)
    if args.run_nbeats:
        print("Running N-BEATS cross-validation...")
        nbeats_results = run_nbeats_cv(df, initial_train_days=args.initial_train_days,
                                       horizon_days=args.horizon_days, step_days=args.step_days,
                                       outdir=args.models_dir, input_len=90, epochs=30,
                                       device=args.device)

    meta_info = {'initial_train_days': args.initial_train_days, 'horizon_days': args.horizon_days, 'step_days': args.step_days}
    write_report('report.md', df, prophet_results, nbeats_results, meta_info)
    print("Done.")

if __name__ == "__main__":
    main()
