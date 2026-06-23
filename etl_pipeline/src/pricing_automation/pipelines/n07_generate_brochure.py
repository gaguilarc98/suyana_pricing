"""
generate_brochure.py
Kedro-compatible node that returns a raw HTML brochure string.
The node output is written to disk via catalog.yml (text.TextDataset).
"""

import base64
import io
import textwrap
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gamma as gamma_dist


# ── provider / field metadata lookup ─────────────────────────────────────────

_PROVIDER_META = {
    ('ERA5', 'swc'): {
        'full_name':    'ERA5 Reanalysis (ECMWF)',
        'variable':     'Soil Water Content (Volumetric)',
        'spatial_res':  '0.25° (~28 km)',
        'temporal_res': 'Daily',
        'update_freq':  'Monthly (with ~5-day latency for near-real-time)',
        'latency':      '~5 days (ERA5T near-real-time)',
    },
    ('ERA5', 'prcp'): {
        'full_name':    'ERA5 Reanalysis (ECMWF)',
        'variable':     'Total Precipitation',
        'spatial_res':  '0.25° (~28 km)',
        'temporal_res': 'Daily',
        'update_freq':  'Monthly (with ~5-day latency for near-real-time)',
        'latency':      '~5 days (ERA5T near-real-time)',
    },
    ('CHIRPS', 'prcp'): {
        'full_name':    'CHIRPS (Climate Hazards Group InfraRed Precipitation with Station data)',
        'variable':     'Precipitation',
        'spatial_res':  '0.05° (~5.5 km)',
        'temporal_res': 'Daily / Pentadal',
        'update_freq':  'Monthly (with ~3-week latency)',
        'latency':      '~3 weeks',
    },
    ('MODIS', 'swc'): {
        'full_name':    'MODIS Terra/Aqua (NASA)',
        'variable':     'Soil Moisture Index',
        'spatial_res':  '0.05° (~5 km)',
        'temporal_res': '8-day composite',
        'update_freq':  'Every 8 days',
        'latency':      '~2 days',
    },
}

_FIELD_EXPLANATION = {
    'en': {
        'swc': (
            "The index is based on Soil Water Content (SWC), which measures the volumetric "
            "fraction of water present in the top soil layers. To build the insurance index, "
            "we first compute the daily climatology (long-term average) for each calendar day. "
            "The anomaly is then defined as the difference between the observed SWC and its "
            "climatological mean. These anomalies are accumulated over the relevant seasonal "
            "window for the insured crop, producing a single cumulative value per season that "
            "captures the overall water deficit or surplus experienced during the growing period."
        ),
        'prcp': (
            "The index is based on cumulative Precipitation over the relevant seasonal window. "
            "Total rainfall is accumulated across each seasonal period, and the resulting "
            "cumulative value is compared against the historical distribution to assign an "
            "empirical percentile. Extreme low values indicate drought conditions; extreme "
            "high values indicate excess rainfall."
        ),
    },
    'es': {
        'swc': (
            "El índice se basa en el Contenido de Agua en el Suelo (SWC), que mide la fracción "
            "volumétrica de agua presente en las capas superficiales del suelo. Para construir "
            "el índice de seguro, primero se calcula la climatología diaria (promedio histórico) "
            "para cada día del calendario. La anomalía se define como la diferencia entre el SWC "
            "observado y su media climatológica. Estas anomalías se acumulan a lo largo de la "
            "ventana estacional relevante para el cultivo asegurado, produciendo un valor "
            "acumulado por temporada que captura el déficit o excedente hídrico total."
        ),
        'prcp': (
            "El índice se basa en la precipitación acumulada durante la ventana estacional "
            "relevante. La lluvia total se acumula en cada período estacional y el valor "
            "resultante se compara con la distribución histórica para asignar un percentil "
            "empírico. Valores extremadamente bajos indican condiciones de sequía; valores "
            "extremadamente altos indican exceso de lluvia."
        ),
    },
}

_LABELS = {
    'en': {
        'title':            'Parametric Insurance Product Brief',
        'section_source':   '1. Data Source',
        'section_aoi':      '2. Area of Interest',
        'section_index':    '3. Index Construction',
        'section_policy':   '4. Policy Payout Mechanism',
        'section_rates':    '5. Insurance Rates',
        'provider':         'Provider',
        'variable':         'Variable',
        'spatial_res':      'Spatial Resolution',
        'temporal_res':     'Temporal Resolution',
        'update_freq':      'Update Frequency',
        'latency':          'Latency',
        'unknown':          '[UNKNOWN — please fill manually]',
        'aoi_caption':      'Geographic coverage of the insured portfolio.',
        'index_title':      'Index Construction',
        'hist_title':       'Historical distribution of the seasonal index',
        'ts_title':         'Example: index accumulation during a single season',
        'policy_intro': (
            "This product uses a <strong>parametric</strong> trigger mechanism. "
            "Rather than assessing individual losses, an automated claim is generated "
            "whenever the seasonal index crosses a pre-agreed threshold derived from "
            "the historical distribution of the index. A parametric distribution is "
            "fitted to the historical index values, and trigger thresholds correspond "
            "to specific percentiles of that distribution."
        ),
        'table_crop':       'Crop',
        'table_window':     'Window',
        'table_tail':       'Tail',
        'table_design':     'Design',
        'table_triggers':   'Triggers',
        'table_rate_limit': 'Rate (% of limit)',
        'table_rate_iv':    'Rate (% of insured value)',
        'chart_title':      'Annual Total Premium and Total Payout (Portfolio)',
        'chart_premium':    'Total Premium',
        'chart_payout':     'Total Payout',
        'footer':           'Suyana. All rights reserved.',
        'generated':        'Generated',
        'logo_placeholder': '[LOGO]',
        'continuous_desc':  (
            "Continuous ramp: payout increases linearly from {min_p:.0%} at the "
            "trigger (p{start}) to {max_p:.0%} at the exit (p{exit})."
        ),
        'layered_desc':     "Layered step: payout activates at discrete thresholds.",
    },
    'es': {
        'title':            'Ficha Técnica de Seguro Paramétrico',
        'section_source':   '1. Fuente de Datos',
        'section_aoi':      '2. Área de Interés',
        'section_index':    '3. Construcción del Índice',
        'section_policy':   '4. Mecanismo de Pago de la Póliza',
        'section_rates':    '5. Tasas del Seguro',
        'provider':         'Proveedor',
        'variable':         'Variable',
        'spatial_res':      'Resolución Espacial',
        'temporal_res':     'Resolución Temporal',
        'update_freq':      'Frecuencia de Actualización',
        'latency':          'Latencia',
        'unknown':          '[DESCONOCIDO — completar manualmente]',
        'aoi_caption':      'Cobertura geográfica del portafolio asegurado.',
        'index_title':      'Construcción del Índice',
        'hist_title':       'Distribución histórica del índice estacional',
        'ts_title':         'Ejemplo: acumulación del índice en una temporada',
        'policy_intro': (
            "Este producto utiliza un mecanismo de activación <strong>paramétrico</strong>. "
            "En lugar de evaluar pérdidas individuales, se genera un reclamo automático "
            "cuando el índice estacional supera un umbral preacordado derivado de la "
            "distribución histórica del índice. Se ajusta una distribución paramétrica "
            "a los valores históricos del índice y los umbrales de activación corresponden "
            "a percentiles específicos de esa distribución."
        ),
        'table_crop':       'Cultivo',
        'table_window':     'Ventana',
        'table_tail':       'Cola',
        'table_design':     'Diseño',
        'table_triggers':   'Disparadores',
        'table_rate_limit': 'Tasa (% del límite)',
        'table_rate_iv':    'Tasa (% del valor asegurado)',
        'chart_title':      'Prima y Pago Total Anual (Portafolio)',
        'chart_premium':    'Prima Total',
        'chart_payout':     'Pago Total',
        'footer':           'Suyana. Todos los derechos reservados.',
        'generated':        'Generado',
        'logo_placeholder': '[LOGOTIPO]',
        'continuous_desc':  (
            "Rampa continua: el pago aumenta linealmente desde {min_p:.0%} en el "
            "disparador (p{start}) hasta {max_p:.0%} en la salida (p{exit})."
        ),
        'layered_desc':     "Escalonado: el pago se activa en umbrales discretos.",
    },
}


# ── helper: figure → base64 string ───────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── helper: AOI map ───────────────────────────────────────────────────────────

def _make_aoi_map(gdf_aoi) -> str:
    fig, ax = plt.subplots(figsize=(6, 5))
    gdf_aoi.to_crs(epsg=4326).plot(
        ax=ax, color='#c8e6c9', edgecolor='#333333', linewidth=0.4
    )
    ax.set_axis_off()
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ── helper: index histogram + fitted gamma ────────────────────────────────────

def _make_hist(df_cum, field, lang, L) -> str:
    vals = df_cum['index_value'].dropna().values
    if len(vals) == 0:
        # Simulate illustrative data
        vals = gamma_dist.rvs(a=3, scale=100, size=200, random_state=42)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.hist(vals, bins=25, density=True, color='steelblue',
            alpha=0.6, edgecolor='white', label='Historical')

    # Fit gamma for overlay
    try:
        shape, loc, scale = gamma_dist.fit(vals, floc=0)
        x = np.linspace(vals.min(), vals.max(), 300)
        ax.plot(x, gamma_dist.pdf(x, shape, loc, scale),
                color='darkred', lw=2, label='Fitted distribution')
    except Exception:
        pass

    ax.set_xlabel('Index value', fontsize=9)
    ax.set_ylabel('Density', fontsize=9)
    ax.set_title(L['hist_title'], fontsize=10)
    ax.legend(fontsize=8)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ── helper: single-season accumulation example ───────────────────────────────

def _make_ts(df_cum, lang, L) -> str:
    # Use most recent complete year, fallback to simulation
    if df_cum is not None and not df_cum.empty:
        complete = df_cum[df_cum.get('flag_complete_window', pd.Series(1,
                   index=df_cum.index)) == 1]
        if not complete.empty:
            yr  = complete['window_year'].max()
            row = complete[complete['window_year'] == yr].iloc[0]
            n   = int(row.get('n_days', 180))
            mu  = row['index_value'] / max(n, 1)
            # Simulate daily values consistent with the season total
            np.random.seed(int(yr))
            daily = np.random.gamma(shape=2, scale=mu / 2, size=n)
            daily = daily / daily.sum() * row['index_value']
        else:
            n, daily = 180, np.random.gamma(2, 2, 180)
    else:
        n, daily = 180, np.random.gamma(2, 2, 180)

    cum = np.cumsum(daily)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(np.arange(n), cum, color='steelblue', lw=1.5)
    ax.fill_between(np.arange(n), cum, alpha=0.2, color='steelblue')
    ax.set_xlabel('Day within season', fontsize=9)
    ax.set_ylabel('Cumulated index', fontsize=9)
    ax.set_title(L['ts_title'], fontsize=10)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ── helper: annual premium / payout chart ─────────────────────────────────────

def _make_annual_chart(df_payouts, lang, L) -> str:
    df = df_payouts.copy()
    annual = (
        df.groupby('window_year')[['total_premium', 'total_payout']]
        .sum()
        .reset_index()
        .sort_values('window_year')
    )

    fig, ax = plt.subplots(figsize=(8, 3.5))
    x = annual['window_year'].astype(int)
    w = 0.35
    ax.bar(x - w / 2, annual['total_premium'], width=w,
           color='steelblue', alpha=0.8, label=L['chart_premium'])
    ax.bar(x + w / 2, annual['total_payout'], width=w,
           color='crimson', alpha=0.7, label=L['chart_payout'])
    ax.set_xlabel('Year', fontsize=9)
    ax.set_ylabel('Amount', fontsize=9)
    ax.set_title(L['chart_title'], fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45, labelsize=7)
    plt.tight_layout()
    b64 = _fig_to_b64(fig)
    plt.close(fig)
    return b64


# ── helper: policy table rows ────────────────────────────────────────────────

def _build_table_rows(df_payouts, df_policy, params_contract, L) -> str:
    crops = params_contract.get('crops', {})
    rows  = []

    seen = set()
    for crop_name, crop_spec in crops.items():
        for window_key, window_spec in crop_spec.get('windows', {}).items():
            for tail_label in ('upper', 'lower'):
                if tail_label not in window_spec:
                    continue
                key = (crop_name, window_key, tail_label)
                if key in seen:
                    continue
                seen.add(key)

                tail_spec = window_spec[tail_label]
                design    = tail_spec['design']
                iv        = tail_spec.get('insured_value', 1)
                coverage  = tail_spec.get('coverage', 1.0)

                # Triggers string
                if design == 'layered':
                    levels = tail_spec.get('levels', {})
                    trig_str = ', '.join(
                        f"p{int(k)}: {v:.0%}" for k, v in sorted(levels.items())
                    )
                else:
                    lo = tail_spec.get('start_pct') or min(tail_spec.get('levels', [0, 0]))
                    hi = tail_spec.get('exit_pct')  or max(tail_spec.get('levels', [0, 0]))
                    trig_str = f"p{lo} → p{hi}"

                # Rates from df_payouts
                mask = (
                    (df_payouts['crop']   == crop_name) &
                    (df_payouts['window'] == window_key) &
                    (df_payouts['tail']   == tail_label)
                )
                subset = df_payouts[mask]
                if subset.empty:
                    rate_limit = '—'
                    rate_iv    = '—'
                else:
                    gpp        = subset['gross_premium_pct'].iloc[0]
                    rate_limit = f"{gpp:.2%}"
                    rate_iv    = f"{gpp * coverage:.2%}"

                rows.append(f"""
                <tr>
                  <td>{crop_name}</td>
                  <td>{window_key}</td>
                  <td>{tail_label}</td>
                  <td>{design}</td>
                  <td style="font-size:0.85em">{trig_str}</td>
                  <td style="text-align:center">{rate_limit}</td>
                  <td style="text-align:center">{rate_iv}</td>
                </tr>""")

    return '\n'.join(rows)


# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', Arial, sans-serif;
  font-size: 14px;
  color: #222;
  background: #f5f6f7;
}
.page {
  max-width: 900px;
  margin: 0 auto;
  background: #fff;
  padding: 40px 50px 60px;
}
header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 3px solid #1a6b3c;
  padding-bottom: 16px;
  margin-bottom: 28px;
}
.logo-box {
  width: 90px; height: 50px;
  border: 2px dashed #aaa;
  display: flex; align-items: center; justify-content: center;
  color: #aaa; font-size: 11px; border-radius: 4px;
}
h1 { font-size: 1.5em; color: #1a6b3c; }
h2 {
  font-size: 1.1em; color: #1a6b3c;
  border-left: 4px solid #1a6b3c;
  padding-left: 10px;
  margin: 32px 0 14px;
}
.meta-table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
.meta-table td { padding: 6px 10px; border-bottom: 1px solid #e8e8e8; }
.meta-table td:first-child { font-weight: 600; width: 38%; color: #444; }
.policy-table {
  border-collapse: collapse; width: 100%;
  font-size: 0.88em; margin-top: 10px;
}
.policy-table th {
  background: #1a6b3c; color: #fff;
  padding: 7px 10px; text-align: left;
}
.policy-table td { padding: 6px 10px; border-bottom: 1px solid #e8e8e8; }
.policy-table tr:nth-child(even) td { background: #f9f9f9; }
p { line-height: 1.65; margin-bottom: 10px; }
.fig-row {
  display: flex; gap: 20px; margin: 16px 0;
  flex-wrap: wrap;
}
.fig-row img { flex: 1 1 45%; max-width: 100%; border-radius: 4px; }
img.full { width: 100%; border-radius: 4px; margin: 12px 0; }
img.half { width: 48%; border-radius: 4px; margin: 12px 0; }
footer {
  margin-top: 48px;
  border-top: 1px solid #ddd;
  padding-top: 10px;
  font-size: 0.75em;
  color: #888;
  display: flex;
  justify-content: space-between;
}
"""


# ── main function ─────────────────────────────────────────────────────────────

def generate_brochure(
    params_request:  dict,
    params_trigger:  dict,
    params_contract: dict,
    df_cum:          pd.DataFrame,
    df_payouts:      pd.DataFrame,
    df_policy:       pd.DataFrame,
    gdf_aoi,
    params_brochure: dict = {},
) -> str:
    """
    Build and return a raw HTML brochure string.
    Write to disk via Kedro catalog (text.TextDataset).

    params_brochure keys:
        language : 'en' | 'es'  (default 'en')
    """
    language = params_brochure.get('language', 'en')
    L        = _LABELS[language]
    field    = params_request.get('field', '').lower()
    provider = params_request.get('provider', '').upper()
    meta_key = (provider, field)
    meta     = _PROVIDER_META.get(meta_key, {})

    def m(key):
        return meta.get(key, L['unknown'])

    field_text = (
        _FIELD_EXPLANATION.get(language, _FIELD_EXPLANATION['en'])
        .get(field, L['unknown'])
    )

    # ── figures ───────────────────────────────────────────────────────────────
    aoi_b64    = _make_aoi_map(gdf_aoi)
    hist_b64   = _make_hist(df_cum, field, language, L)
    ts_b64     = _make_ts(df_cum, language, L)
    chart_b64  = _make_annual_chart(df_payouts, language, L)
    table_rows = _build_table_rows(df_payouts, df_policy, params_contract, L)

    now = datetime.now().strftime('%Y-%m-%d %H:%M')

    html = f"""<!DOCTYPE html>
<html lang="{language}">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{L['title']}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="page">

  <!-- HEADER -->
  <header>
    <div class="logo-box">{L['logo_placeholder']}</div>
    <h1>{L['title']}</h1>
    <div style="font-size:0.8em;color:#666;text-align:right">
      {L['generated']}<br>{now}
    </div>
  </header>

  <!-- SECTION 1: DATA SOURCE -->
  <h2>{L['section_source']}</h2>
  <table class="meta-table">
    <tr><td>{L['provider']}</td><td>{m('full_name')}</td></tr>
    <tr><td>{L['variable']}</td><td>{m('variable')}</td></tr>
    <tr><td>{L['spatial_res']}</td><td>{m('spatial_res')}</td></tr>
    <tr><td>{L['temporal_res']}</td><td>{m('temporal_res')}</td></tr>
    <tr><td>{L['update_freq']}</td><td>{m('update_freq')}</td></tr>
    <tr><td>{L['latency']}</td><td>{m('latency')}</td></tr>
  </table>

  <!-- SECTION 2: AOI -->
  <h2>{L['section_aoi']}</h2>
  <img class="full" src="data:image/png;base64,{aoi_b64}"
       alt="Area of Interest">
  <p style="font-size:0.85em;color:#555;text-align:center">
    {L['aoi_caption']}
  </p>

  <!-- SECTION 3: INDEX CONSTRUCTION -->
  <h2>{L['section_index']}</h2>
  <p>{field_text}</p>
  <div class="fig-row">
    <img src="data:image/png;base64,{hist_b64}" alt="Index histogram">
    <img src="data:image/png;base64,{ts_b64}"   alt="Season accumulation">
  </div>

  <!-- SECTION 4: POLICY PAYOUT -->
  <h2>{L['section_policy']}</h2>
  <p>{L['policy_intro']}</p>
  <img class="full" src="data:image/png;base64,{chart_b64}"
       alt="Annual premium and payout">

  <!-- SECTION 5: INSURANCE RATES -->
  <h2>{L['section_rates']}</h2>
  <table class="policy-table">
    <thead>
      <tr>
        <th>{L['table_crop']}</th>
        <th>{L['table_window']}</th>
        <th>{L['table_tail']}</th>
        <th>{L['table_design']}</th>
        <th>{L['table_triggers']}</th>
        <th>{L['table_rate_limit']}</th>
        <th>{L['table_rate_iv']}</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>

  <!-- FOOTER -->
  <footer>
    <span>{L['footer']}</span>
    <span>{L['generated']}: {now}</span>
  </footer>

</div>
</body>
</html>"""

    return html


# ── convenience: open in browser or notebook ─────────────────────────────────

def open_brochure(path: str):
    """Open a saved brochure HTML file in the default browser or Jupyter."""
    import os
    try:
        from IPython.display import IFrame, display
        display(IFrame(src=path, width='100%', height=850))
    except ImportError:
        import webbrowser
        webbrowser.open(f'file://{os.path.abspath(path)}')