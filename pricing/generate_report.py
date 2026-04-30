"""
Generate a Suyana-styled LaTeX report (article) and Beamer presentation
from the pricing pipeline outputs, using Claude with vision to write a
coherent technical narrative across the plots and tabular results.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python generate_report.py \\
        --outputs-dir outputs \\
        --pricing-config config.yaml \\
        --etl-conf ../etl_pipeline/conf/base \\
        --plots-dir outputs/plots \\
        --out outputs/report_<region>_<YYYYMMDD> \\
        --region "Mendoza, Argentina" \\
        --model claude-opus-4-7

Outputs
-------
    <out>/report.tex         Suyana-styled article (full document)
    <out>/report.pdf         compiled
    <out>/presentation.tex   Suyana-styled Beamer deck
    <out>/presentation.pdf   compiled
    <out>/presentation.pptx  optional, via --emit-pptx
    <out>/figures/*.png      copies of every input plot
    <out>/brand/             logo files (if present)
    <out>/prompt_inputs.json structured snapshot of what was sent to Claude
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from anthropic import Anthropic

# ---------------------------------------------------------------------------
# API key handling — placeholder fails loudly, never silently
# ---------------------------------------------------------------------------

API_KEY_PLACEHOLDER = "<<PUT_YOUR_ANTHROPIC_API_KEY_HERE>>"


def load_api_key() -> str:
    key = os.environ.get("ANTHROPIC_API_KEY", API_KEY_PLACEHOLDER)
    env_path = Path(__file__).parent / ".env"
    if key.startswith("<<") and env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
                break
    if key.startswith("<<") or not key:
        sys.exit(
            "ANTHROPIC_API_KEY is not set. Export it or put it in pricing/.env "
            "before running. Do not hardcode it in this file."
        )
    return key


# ---------------------------------------------------------------------------
# Suyana brand tokens — single source of truth, mirrors the suyana-presentation skill
# ---------------------------------------------------------------------------

SUYANA_PALETTE = {
    "sgreen": "43A047",
    "sblack": "141414",
    "sdark": "111111",
    "sgray": "555555",
    "smidgray": "9E9E9E",
}

ARTICLE_PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[__BABEL__]{babel}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[a4paper,margin=2.4cm,headsep=0.7cm,footskip=1.2cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{xcolor}
\usepackage{fancyhdr}
\usepackage{caption}
\usepackage{hyperref}

\definecolor{sgreen}{HTML}{43A047}
\definecolor{sblack}{HTML}{141414}
\definecolor{sdark}{HTML}{111111}
\definecolor{sgray}{HTML}{555555}
\definecolor{smidgray}{HTML}{9E9E9E}

\hypersetup{colorlinks=true, linkcolor=sgreen, urlcolor=sgreen, citecolor=sgreen}

\graphicspath{{figures/}{brand/}}

% Section numbers in Suyana green — uses base LaTeX only, no titlesec dependency.
\renewcommand{\thesection}{\textcolor{sgreen}{\arabic{section}}}
\renewcommand{\thesubsection}{\textcolor{sgreen}{\arabic{section}.\arabic{subsection}}}
\renewcommand{\thesubsubsection}{\textcolor{sgreen}{\arabic{section}.\arabic{subsection}.\arabic{subsubsection}}}

% Green bullets without the enumitem dependency.
\renewcommand{\labelitemi}{\textcolor{sgreen}{\textbullet}}
\renewcommand{\labelitemii}{\textcolor{sgreen}{\textendash}}

\captionsetup{font=small, labelfont={bf,color=sgreen}, labelsep=period}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\renewcommand{\footrulewidth}{0pt}
\fancyhead[L]{{\color{smidgray}\small \@title}}
\fancyhead[R]{{\color{smidgray}\small __CONFIDENTIAL__}}
\fancyfoot[L]{{\color{smidgray}\small suyana.io}}
\fancyfoot[R]{{\color{smidgray}\small \thepage}}
\renewcommand{\headrule}{{\color{sgreen}\hrule height 1.5pt}\vspace{2pt}{\color{smidgray}\hrule height 0.4pt}}
\renewcommand{\footrule}{{\color{smidgray}\hrule height 0.4pt}}
"""

BRIEF_PREAMBLE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[__BABEL__]{babel}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage[a4paper,margin=1.15cm,top=0.8cm,bottom=0.9cm]{geometry}
\usepackage{graphicx}
\usepackage{xcolor}
\usepackage{tikz}
\usepackage{booktabs}
\usepackage{parskip}
\usetikzlibrary{positioning}

\definecolor{sgreen}{HTML}{43A047}
\definecolor{sdarkgreen}{HTML}{2E7D32}
\definecolor{sblack}{HTML}{141414}
\definecolor{sdark}{HTML}{111111}
\definecolor{sgray}{HTML}{555555}
\definecolor{smidgray}{HTML}{9E9E9E}
\definecolor{slightgray}{HTML}{F2F2F2}

\graphicspath{{figures/}{brand/}}

\pagestyle{empty}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}

% ---------------------------------------------------------------------------
% Suyana brief macros — match the dashboard-style product-sheet layout.
% ---------------------------------------------------------------------------

% Dark header band: brand wordmark + product title + subtitle line.
\newcommand{\suyanaheader}[2]{%
  \noindent\begin{tikzpicture}[baseline=(current bounding box.north)]
    \node[fill=sdark, rounded corners=10pt, minimum width=\textwidth,
          inner xsep=18pt, inner ysep=11pt, anchor=north west] (h) at (0,0) {%
      \parbox{\dimexpr\textwidth-40pt}{%
        \color{white}%
        {\huge\textbf{Suyana}}\hfill{\Large\textbf{#1}}\\[3pt]%
        {\color{smidgray}\small\textsc{Hoja de producto}}\hfill{\color{smidgray}\small #2}%
      }%
    };
  \end{tikzpicture}\par
}

% A single stat pill — green uppercase label + white value. Used inside \suyanastats.
\newcommand{\statpill}[2]{%
  {\color{sgreen}\scriptsize\textbf{\MakeUppercase{#1}}}\\[3pt]%
  {\color{white}\normalsize #2}%
}

% Stats strip: 6 pills inside a continuation of the dark band.
\newcommand{\suyanastats}[6]{%
  \par\vspace{5pt}\noindent\begin{tikzpicture}[baseline=(current bounding box.north)]
    \node[fill=sdark, rounded corners=10pt, minimum width=\textwidth,
          inner xsep=12pt, inner ysep=8pt, anchor=north west] {%
      \parbox{\dimexpr\textwidth-28pt}{%
        \begin{minipage}[t]{0.16\linewidth}\centering #1\end{minipage}\hfill
        \begin{minipage}[t]{0.16\linewidth}\centering #2\end{minipage}\hfill
        \begin{minipage}[t]{0.16\linewidth}\centering #3\end{minipage}\hfill
        \begin{minipage}[t]{0.16\linewidth}\centering #4\end{minipage}\hfill
        \begin{minipage}[t]{0.16\linewidth}\centering #5\end{minipage}\hfill
        \begin{minipage}[t]{0.16\linewidth}\centering #6\end{minipage}%
      }%
    };
  \end{tikzpicture}\par\vspace{6pt}
}

% Big-number tile for a headline figure (AAL, prima, etc.). Use in a row.
\newcommand{\bignumtile}[2]{%
  \begin{tikzpicture}[baseline=(current bounding box.north)]
    \node[fill=white, draw=slightgray, line width=0.8pt, rounded corners=8pt,
          inner xsep=10pt, inner ysep=8pt, text width=\linewidth-22pt,
          minimum height=48pt, anchor=north west, align=center] {%
      {\fontsize{19}{22}\selectfont\bfseries\color{sgreen} #1}\\[2pt]%
      {\scriptsize\color{sgray} #2}%
    };
  \end{tikzpicture}%
}

% Content panel: green uppercase title + slightgray rounded body.
\newcommand{\suyanapanel}[2]{%
  \begin{tikzpicture}[baseline=(current bounding box.north)]
    \node[fill=slightgray, rounded corners=8pt,
          inner xsep=9pt, inner ysep=7pt, text width=\linewidth-20pt,
          anchor=north west, align=left] {%
      {\color{sgreen}\bfseries\scriptsize\MakeUppercase{#1}}\\[3pt]%
      {\color{sblack}\footnotesize #2}%
    };
  \end{tikzpicture}%
}

% Fixed-height plot block. Keeps column bottoms aligned even when captions vary.
\newcommand{\briefplot}[2]{%
  \begin{center}
    \includegraphics[width=\linewidth,height=0.115\textheight,keepaspectratio]{#1}%
  \end{center}
  \vspace{-4pt}%
  {\scriptsize\color{sgray}#2\par}%
}

% Numbered FAQ item — green circle with the number, then a content panel.
\newcommand{\faqitem}[3]{%
  \noindent\begin{minipage}{\linewidth}
    \tikz[baseline=(n.base)]\node[circle, fill=sgreen, text=white,
      font=\bfseries\scriptsize, minimum size=15pt, inner sep=0pt] (n) {#1};%
    \hspace{5pt}%
    \parbox[t]{\dimexpr\linewidth-28pt\relax}{\bfseries\scriptsize #2}%
    \par\vspace{2pt}%
    \begin{tikzpicture}[baseline=(current bounding box.north)]
      \node[fill=slightgray, rounded corners=8pt,
          inner xsep=7pt, inner ysep=5pt, text width=\linewidth-16pt,
          anchor=north west] {%
        {\color{sblack}\scriptsize #3}%
      };
    \end{tikzpicture}%
  \end{minipage}%
}

% Bottom call-to-action strip: green band, centered headline + contact.
\newcommand{\suyanacta}[2]{%
  \noindent\begin{tikzpicture}[baseline=(current bounding box.north)]
    \node[fill=sgreen, text=white, rounded corners=10pt,
          inner xsep=18pt, inner ysep=10pt,
          minimum width=\textwidth, anchor=north west] {%
      \parbox{\dimexpr\textwidth-40pt}{%
        \centering\bfseries\large #1\\[3pt]%
        \normalfont\small #2%
      }%
    };
  \end{tikzpicture}\par
}

% Footer disclaimer — small, gray, centered.
\newcommand{\suyanafooter}[1]{%
  \par\vspace{4pt}{\centering\color{smidgray}\tiny #1\par}%
}
"""

BEAMER_PREAMBLE = r"""\documentclass[aspectratio=169,11pt]{beamer}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage[__BABEL__]{babel}
\usepackage{lmodern}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{etoolbox}
\usepackage{tikz}
\usetikzlibrary{calc,positioning}

\definecolor{sgreen}{HTML}{43A047}
\definecolor{sblack}{HTML}{141414}
\definecolor{sdark}{HTML}{111111}
\definecolor{sgray}{HTML}{555555}
\definecolor{smidgray}{HTML}{9E9E9E}

\usetheme{default}
\usecolortheme{default}
\setbeamertemplate{navigation symbols}{}
\setbeamercolor{frametitle}{fg=sblack, bg=white}
\setbeamercolor{structure}{fg=sgreen}
\setbeamercolor{normal text}{fg=sblack}
\setbeamercolor{itemize item}{fg=sgreen}
\setbeamercolor{itemize subitem}{fg=sgreen}
\setbeamercolor{enumerate item}{fg=sgreen}

\setbeamerfont{itemize/enumerate body}{size=\footnotesize}
\setbeamerfont{itemize/enumerate subbody}{size=\scriptsize}
\setbeamerfont{block body}{size=\footnotesize}

\AtBeginEnvironment{itemize}{\setlength{\itemsep}{3pt}\setlength{\parskip}{0pt}\setlength{\topsep}{2pt}}
\AtBeginEnvironment{enumerate}{\setlength{\itemsep}{3pt}\setlength{\parskip}{0pt}\setlength{\topsep}{2pt}}

\setbeamertemplate{headline}{%
  \color{sgreen}\rule{\paperwidth}{1.5pt}\par
  \vspace{4pt}%
}
\setbeamertemplate{frametitle}{%
  \vspace{4pt}{\large\textbf{\insertframetitle}}%
  \ifx\insertframesubtitle\empty\else\\[1pt]{\small\color{sgray}\insertframesubtitle}\fi%
  \par\vspace{4pt}{\color{smidgray}\hrule height 0.4pt}\vspace{4pt}%
}
\setbeamertemplate{footline}{%
  {\color{smidgray}\hrule height 0.4pt}\vspace{2pt}%
  \hspace{6pt}{\tiny\color{smidgray}__PRODUCT_NAME__\ \ |\ \ __CONFIDENTIAL__\ \ |\ \ __MONTH_YEAR__}%
  \hfill{\tiny\color{smidgray}\insertframenumber\,/\,\inserttotalframenumber}\hspace{6pt}\vspace{3pt}%
}

\graphicspath{{figures/}{brand/}}

\newcommand{\secslide}[2]{{%
  \setbeamertemplate{headline}{}\setbeamertemplate{footline}{}%
  \setbeamercolor{background canvas}{bg=sdark}%
  \begin{frame}[plain]%
    \vspace{1.0cm}\hspace{0.4cm}{\color{sgreen}\fontsize{40}{40}\selectfont\textbf{#1}}%
    \par\vspace{0.3cm}\hspace{0.4cm}{\color{white}\Large\textbf{\MakeUppercase{#2}}}%
    \vfill\vspace{6pt}%
  \end{frame}}}
"""

# ---------------------------------------------------------------------------
# Input gathering
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    region: str
    pricing_config: dict
    etl_parameters: dict | None
    etl_globals: dict | None
    parquet_summaries: dict
    qa_report_md: str
    contract_slip_md: str
    pricing_quote_csv: str
    plot_files: list[Path]


def _read_yaml(path: Path) -> dict | None:
    if not path.exists():
        return None
    with path.open() as fh:
        return yaml.safe_load(fh)


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _summarize_parquet(path: Path) -> dict[str, Any]:
    df = pd.read_parquet(path)
    summary: dict[str, Any] = {
        "file": path.name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "head": df.head(8).to_dict(orient="records"),
    }
    numeric = df.select_dtypes("number")
    if not numeric.empty:
        desc = numeric.describe(percentiles=[0.05, 0.5, 0.95]).round(4)
        summary["describe"] = desc.to_dict()
    return summary


def gather_inputs(
    outputs_dir: Path,
    pricing_config_path: Path,
    etl_conf_dir: Path | None,
    plots_dir: Path,
    region: str,
) -> Inputs:
    parquet_summaries = {
        p.stem: _summarize_parquet(p) for p in sorted(outputs_dir.glob("*.parquet"))
    }
    return Inputs(
        region=region,
        pricing_config=_read_yaml(pricing_config_path) or {},
        etl_parameters=_read_yaml(etl_conf_dir / "parameters.yml") if etl_conf_dir else None,
        etl_globals=_read_yaml(etl_conf_dir / "globals.yml") if etl_conf_dir else None,
        parquet_summaries=parquet_summaries,
        qa_report_md=_read_text(outputs_dir / "qa_report.md"),
        contract_slip_md=_read_text(outputs_dir / "contract_slip.md"),
        pricing_quote_csv=_read_text(outputs_dir / "pricing_quote.csv"),
        plot_files=sorted(plots_dir.glob("*.png")),
    )


# ---------------------------------------------------------------------------
# Language packs — every translatable string lives here
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Language:
    code: str
    babel: str
    confidential: str
    system_prompt: str
    style_rules: str
    article_instruction: str
    beamer_instruction: str
    brief_instruction: str


_STYLE_RULES_ES = r"""
Eres un consultor técnico de Suyana redactando entregables de seguros
paramétricos. Aplicás de forma estricta las convenciones de marca de Suyana:

PALETA (única permitida):
  sgreen   #43A047   acento principal — números de sección, viñetas, reglas
  sblack   #141414   texto principal
  sdark    #111111   fondo oscuro de portada y separadores
  sgray    #555555   texto secundario, subtítulos
  smidgray #9E9E9E   texto silenciado, reglas finas, pie de página

TERMINOLOGÍA en español (NUNCA uses los acrónimos en cuerpo de texto):
  GPD  -> distribución de Pareto generalizada
  EVT  -> teoría de valores extremos
  POT  -> máximos sobre umbral
  AEP  -> probabilidad anual de excedencia
  AAL  -> pérdida media anual
  RoL  -> tasa de prima
  RP   -> período de retorno
  IC   -> intervalo de confianza
  ENSO -> El Niño
  NDVI -> índice de vegetación
  profundidad (de sequía) -> intensidad

REGLAS DE CONTENIDO:
  - Toda fórmula matemática debe ir acompañada de una leyenda inmediata
    (`\textbf{donde:}` ...) que defina cada símbolo, incluyendo valores numéricos
    de las constantes calibradas.
  - Toda fórmula debe ir seguida de al menos un ejemplo numérico paso a paso
    usando datos históricos reales (un año concreto), exponiendo los pasos
    intermedios y terminando en un valor en USD cuando aplique.
  - Nunca presentes un valor calculado sin su aritmética: muestra la cadena
    fracción -> exceso -> escalado a [0;1].
  - Cada constante no obvia debe tener una línea "de dónde viene"
    (ej. 0,20 = 1 - S_min, "la distancia entre el piso y el techo").
  - Ningún acrónimo en cuerpo de texto. Solo se permiten como símbolos en
    fórmulas matemáticas.
  - Las tablas usan booktabs (\toprule, \midrule, \bottomrule) con
    \renewcommand{\arraystretch}{1.3}.
  - Toda figura debe llevar caption con lectura analítica concreta —
    no descripciones genéricas.

ESTILO:
  - Tono profesional, técnico, en español rioplatense neutro.
  - Prosa densa y directa; sin frases de relleno tipo "es importante notar que".
  - Conectá narrativamente: cada sección debe construir sobre la anterior y
    cerrar apuntando a la siguiente.
  - No inventes números. Si un dato no está en los insumos provistos, no lo
    menciones.
"""

_ARTICLE_INSTRUCTION_ES = r"""
Escribí el CUERPO de un informe técnico de Suyana en LaTeX. Suyana es una
startup de seguros paramétricos — el tono debe ser corporativo, directo y
comercial, NO académico. NO incluyas el preámbulo (\documentclass,
\usepackage, definiciones de color, \begin{document}, \end{document}) — esos
los agrego yo. Empezá con \title{...}, \author{Suyana}, \date{...},
\maketitle, y luego las secciones.

Estructura obligatoria (en este orden):
  1. \section{Producto y cobertura} — Qué cubre el producto y para quién, en
     2–3 párrafos máximo. Líder con el valor: qué problema resuelve, en qué
     región, sobre qué cultivos, ventana de cobertura. Sin fórmulas.
  2. \section{Cómo funciona} — Mecánica del gatillo y la estructura de pago
     en lenguaje claro. Una figura (la curva de probabilidad anual de
     excedencia funciona bien aquí) con caption analítico. Sin fórmulas
     todavía.
  3. \section{Datos y validación} — Fuentes (ERA5, CHIRPS), resolución,
     ventana climatológica, controles de calidad. Una tabla booktabs con las
     fuentes. Tono breve, factual.
  4. \section{Resultados} — Los números primero. Pérdida media anual, prima
     técnica, prima comercial, tasa de prima. Una tabla booktabs con los
     valores a períodos de retorno clave (5, 10, 25, 50, 100 años) si están
     en los insumos. Insertá las figuras de pérdidas históricas y AEP per-cultivo
     con captions analíticos que extraigan la lectura concreta.
  5. \section{Anexo técnico} — Acá viven las fórmulas. Una subsección por
     paso del pipeline (climatología → anomalías acumuladas → gatillos →
     bootstrap → cotización). Cada fórmula con leyenda inmediata y un
     ejemplo numérico real paso a paso usando un año concreto del dataset.
  6. \section{Anexo: slip de contrato} — Reproducí verbatim el slip de
     contrato provisto en los insumos.

Reglas de tono:
  - Voz activa, oraciones cortas, sin frases de relleno
    ("es importante notar que", "cabe destacar", "se observa que").
  - Lideramos con el cliente, no con la metodología. La fórmula explica
    por qué confiar en el producto, no es el producto.
  - Sin pasados rebuscados, sin condicionales innecesarios.
  - Números antes que adjetivos. "La prima es USD 1,8M sobre un notional
    de USD 60M" es mejor que "la prima resulta competitiva".

Reglas técnicas:
  - Insertá figuras con \begin{figure}[ht]...\includegraphics[width=0.85\textwidth]{<archivo>}...\end{figure}
    y \caption{} analítico — la lectura concreta del gráfico, no
    "se muestra la curva".
  - Tablas booktabs (\toprule/\midrule/\bottomrule) con
    \renewcommand{\arraystretch}{1.3}.
  - Cada sección cierra apuntando a la siguiente para mantener narrativa.
  - No inventes números. Si un dato no está en los insumos, no lo menciones.

Devolvé EXCLUSIVAMENTE el LaTeX del cuerpo, sin bloque de código markdown,
sin comentarios fuera del LaTeX, sin texto introductorio.
"""

_BEAMER_INSTRUCTION_ES = r"""
Escribí el CUERPO de una presentación Beamer de Suyana en LaTeX. La
audiencia es un cliente o reasegurador — claridad, no profundidad académica.
NO incluyas el preámbulo (yo lo agrego). Empezá con
\begin{document} ... \end{document}.

Objetivo: 20–25 diapositivas totales (incluyendo separadores).

Estructura obligatoria:
  1. Portada oscura (fondo sdark, título blanco, subtítulo en smidgray,
     metadatos en sgray pequeños).
  2. Agenda — máximo 4 secciones con numerales verdes.
  3. \secslide{01}{Producto y cobertura} — 1 separador + 2 frames.
     • Frame 1: qué cubre, en qué región, sobre qué cultivos. Sin fórmulas.
     • Frame 2: ventana de cobertura + propuesta de valor en 2 viñetas.
  4. \secslide{02}{Datos} — 1 separador + 2 frames de credibilidad
     únicamente. Una validación cuantitativa (scatter o mapa) y una tabla
     resumen de fuentes. Nada de profundizar en estaciones meteorológicas.
  5. \secslide{03}{Diseño del índice} — 1 separador + 3 frames.
     • Frame 1: pipeline visual (climatología → anomalías → gatillo).
     • Frame 2: fórmula del índice con leyenda \scriptsize.
     • Frame 3: ejemplo numérico paso a paso usando un año real del dataset
       — exponé fracción → exceso → escalado a [0;1] → cierre en USD.
  6. \secslide{04}{Diseño del contrato} — 1 separador + 4 frames.
     • Frame 1: serie temporal del índice anual (figura del pipeline).
     • Frame 2: curva de período de retorno con caption analítico.
     • Frame 3: estructura de capas — capa atacante, capa media, capa de
       cola. Layout de dos columnas: diagrama izquierda, tabla
       parámetros derecha.
     • Frame 4: marco de cuatro lentes para K (parámetro disparador):
       Mecánica / En el dato / En el tiempo / En la economía. Cerrar la
       diapositiva con la frase puente: con el reasegurador hablamos en
       período de retorno; con el cedente hablamos en extensión de cartera.
  7. \secslide{05}{Resultados} — 1 separador + 2 frames.
     • Frame 1: tabla de cotización (pérdida media anual, prima técnica,
       prima comercial, tasa de prima) en booktabs.
     • Frame 2: figura de pérdidas anuales o AEP per-cultivo con caption
       analítico.
  8. \secslide{06}{Próximos pasos} — 1 frame con 3 acciones concretas.
  9. Diapositiva de cierre oscura "Gracias" con metadatos.

Reglas estrictas (cliente-facing):
  - Mostrá la aritmética siempre. "Índice $= 0{,}648$" sin pasos previos
    está prohibido. La cadena fracción → exceso → escalado debe verse.
  - Toda constante no obvia lleva una línea "de dónde viene". Por ejemplo,
    "0,20 = 1 − S_min, la distancia entre el piso y el techo".
  - Toda métrica ponderada lleva su ratio explícito. "Captura ponderada por
    hectáreas = Σ ha detectadas / Σ ha reclamadas".
  - Sin frases como "el diseño anterior", "antes hacíamos X", "en la
    versión previa". Solo el producto actual.
  - Sin acrónimos en texto corrido (ver terminología).
  - Captions de figuras analíticos, no genéricos.
  - Usá los nombres de archivo exactos de las figuras provistas.
  - Máximo 3 viñetas por columna en dos columnas, 4 en una columna; cada
    viñeta ≤ 1 línea a \footnotesize.
  - Imágenes con
    \includegraphics[width=\textwidth,height=0.68\textheight,keepaspectratio].

Devolvé EXCLUSIVAMENTE el LaTeX desde \begin{document} hasta \end{document},
sin bloque markdown, sin texto fuera del LaTeX.
"""

_STYLE_RULES_EN = r"""
You are a senior technical consultant at Suyana drafting parametric insurance
deliverables. Apply Suyana's brand conventions strictly:

PALETTE (the only colours allowed):
  sgreen   #43A047   primary accent — section numbers, bullets, rules
  sblack   #141414   body text
  sdark    #111111   dark background for cover and section dividers
  sgray    #555555   secondary text, subtitles
  smidgray #9E9E9E   muted text, fine rules, footer

TERMINOLOGY (never use the acronyms in body text — spell them out):
  GPD  -> generalised Pareto distribution
  EVT  -> extreme value theory
  POT  -> peaks over threshold
  AEP  -> annual exceedance probability
  AAL  -> average annual loss
  RoL  -> rate on line
  RP   -> return period
  CI   -> confidence interval
  ENSO -> El Niño / La Niña
  NDVI -> vegetation index
  drought "depth" -> drought intensity

CONTENT RULES:
  - Every mathematical formula must be followed immediately by a legend
    (`\textbf{where:}` ...) defining every symbol, including the numerical
    values of calibrated constants.
  - Every formula must be followed by at least one step-by-step worked
    example using a real historical year, exposing intermediate steps and
    ending in a USD figure when applicable.
  - Never present a calculated value without its arithmetic: show the chain
    fraction -> excess -> rescaled to [0,1].
  - Every non-obvious constant must carry a "where it comes from" line
    (e.g. 0.20 = 1 - S_min, "the distance between floor and ceiling").
  - No acronyms in body text. Acronyms are only allowed as formal symbols
    inside math expressions.
  - Tables use booktabs (\toprule, \midrule, \bottomrule) with
    \renewcommand{\arraystretch}{1.3}.
  - Every figure must carry an analytical caption that extracts the concrete
    reading — never generic descriptions.

VOICE:
  - Professional, technical, neutral international English.
  - Dense and direct prose; no filler such as "it is important to note that".
  - Connect narratively: each section must build on the previous one and
    close by pointing to the next.
  - Do not invent numbers. If a value is not in the provided inputs, do not
    mention it.
"""

_ARTICLE_INSTRUCTION_EN = r"""
Write the BODY of a Suyana technical report in LaTeX. Suyana is a parametric
insurance startup — the tone is corporate, direct, and commercial, NOT
academic. Do NOT include the preamble (\documentclass, \usepackage, colour
definitions, \begin{document}, \end{document}) — I add those. Start with
\title{...}, \author{Suyana}, \date{...}, \maketitle, then the sections.

Required structure (in this order):
  1. \section{Product and coverage} — What the product covers and for whom,
     two to three paragraphs maximum. Lead with the value: what problem it
     solves, region, crops, coverage window. No formulas.
  2. \section{How it works} — Trigger mechanics and payout structure in
     plain language. One figure (the annual exceedance probability curve
     fits well here) with an analytical caption. No formulas yet.
  3. \section{Data and validation} — Sources (ERA5, CHIRPS), resolution,
     climatological window, quality controls. A booktabs table of sources.
     Brief, factual tone.
  4. \section{Results} — Numbers first. Average annual loss, technical
     premium, commercial premium, rate on line. A booktabs table of losses
     at key return periods (5, 10, 25, 50, 100 years) when present in the
     inputs. Insert the historical-loss and per-crop AEP figures with
     analytical captions that extract the concrete reading.
  5. \section{Technical appendix} — Formulas live here. One subsection per
     pipeline step (climatology → cumulative anomalies → triggers →
     bootstrap → quote). Every formula with its immediate legend and a
     step-by-step worked numerical example using a real year from the dataset.
  6. \section{Appendix: contract slip} — Reproduce the contract slip from
     the inputs verbatim.

Voice rules:
  - Active voice, short sentences, no filler
    ("it is important to note", "it should be highlighted", "one observes that").
  - Lead with the client, not the methodology. The formula explains why to
    trust the product; it is not the product itself.
  - Numbers before adjectives. "The premium is USD 1.8M against a notional
    of USD 60M" is better than "the premium is competitive".

Technical rules:
  - Insert figures with \begin{figure}[ht]...\includegraphics[width=0.85\textwidth]{<file>}...\end{figure}
    and an analytical \caption{} — the concrete reading of the chart, never
    "the curve is shown".
  - Booktabs tables (\toprule/\midrule/\bottomrule) with
    \renewcommand{\arraystretch}{1.3}.
  - Each section closes by pointing to the next to maintain narrative flow.
  - Do not invent numbers. If a value is not in the inputs, do not mention it.

Return ONLY the LaTeX body, with no markdown code fence, no commentary
outside the LaTeX, no introductory text.
"""

_BEAMER_INSTRUCTION_EN = r"""
Write the BODY of a Suyana Beamer presentation in LaTeX. The audience is a
client or reinsurer — clarity, not academic depth. Do NOT include the
preamble (I add it). Start with \begin{document} ... \end{document}.

Target: 20–25 slides total (including dividers).

Required structure:
  1. Dark cover slide (sdark background, white title, smidgray subtitle,
     small sgray metadata).
  2. Agenda — at most 4 sections with green numerals.
  3. \secslide{01}{Product and coverage} — 1 divider + 2 frames.
     • Frame 1: what is covered, region, crops. No formulas.
     • Frame 2: coverage window + value proposition in 2 bullets.
  4. \secslide{02}{Data} — 1 divider + 2 frames for credibility only.
     One quantitative validation (scatter or map) and a source-summary
     table. Do not deep-dive into station networks.
  5. \secslide{03}{Index design} — 1 divider + 3 frames.
     • Frame 1: visual pipeline (climatology → anomalies → trigger).
     • Frame 2: index formula with \scriptsize legend.
     • Frame 3: step-by-step worked example using a real year — show the
       chain fraction → excess → rescaled to [0,1] → close in USD.
  6. \secslide{04}{Contract design} — 1 divider + 4 frames.
     • Frame 1: annual index time series (pipeline figure).
     • Frame 2: return-period curve with analytical caption.
     • Frame 3: layer structure — attaching, middle, tail layer. Two-column
       layout: diagram on the left, parameter table on the right.
     • Frame 4: four-lens framework for K (the trigger parameter):
       Mechanics / In the data / In time / In the economics. Close the
       slide with the bridge sentence: with the reinsurer talk in return
       period; with the cedent talk in portfolio extent.
  7. \secslide{05}{Results} — 1 divider + 2 frames.
     • Frame 1: pricing table (average annual loss, technical premium,
       commercial premium, rate on line) as booktabs.
     • Frame 2: annual losses or per-crop AEP figure with analytical caption.
  8. \secslide{06}{Next steps} — 1 frame with 3 concrete actions.
  9. Dark closing slide "Thank you" with metadata.

Strict rules (client-facing):
  - Always show the arithmetic. "Index = 0.648" without prior steps is
    forbidden. The chain fraction → excess → rescaled must be visible.
  - Every non-obvious constant carries a "where it comes from" line. For
    example, "0.20 = 1 − S_min, the distance between floor and ceiling".
  - Every weighted metric carries its explicit ratio. "Hectare-weighted
    capture = Σ ha detected / Σ ha claimed".
  - No phrases like "the previous design", "we used to do X", "in the
    earlier version". Only the current product.
  - No acronyms in running text (see terminology).
  - Analytical figure captions, never generic.
  - Use the exact figure filenames provided.
  - At most 3 bullets per column in two-column layouts, 4 in single column;
    each bullet ≤ 1 line at \footnotesize.
  - Images with
    \includegraphics[width=\textwidth,height=0.68\textheight,keepaspectratio].

Return ONLY the LaTeX from \begin{document} to \end{document}, with no
markdown fence, no text outside the LaTeX.
"""

_BRIEF_INSTRUCTION_ES = r"""
Escribí el CUERPO de un BRIEF DE CLIENTE Suyana en LaTeX, en formato
DASHBOARD VERTICAL COMPACTO (A4 portrait), de UNA PÁGINA. Audiencia: productor
agropecuario sudamericano. Tono: claro, comercial, didáctico, cero jerga
técnica. Sin fórmulas. Sin acrónimos. Oraciones cortas (≤ 14 palabras).

NO incluyas el preámbulo (yo lo agrego). Empezá directamente después de
\begin{document}, terminá antes de \end{document}. Usá los siguientes
macros pre-definidos en el preámbulo: \suyanaheader, \suyanastats,
\statpill, \bignumtile, \suyanapanel, \briefplot, \faqitem, \suyanacta,
\suyanafooter.

══════════════════════════════════════════════════════════════════════════
PÁGINA ÚNICA — PRODUCTO, MECÁNICA Y FAQ
══════════════════════════════════════════════════════════════════════════

  1. Header oscuro:
     \suyanaheader{Producto Paramétrico de Sequía}{<región> · <mes año>}

  2. Strip de stats (6 pills, todas \statpill{LABEL}{Valor}):
     \suyanastats
       {\statpill{Región}{<provincia/país de los insumos>}}
       {\statpill{Cultivos}{<cultivos del config>}}
       {\statpill{Ventana}{<rango de la temporada>}}
       {\statpill{Notional}{USD <monto del slip>}}
       {\statpill{Climatología}{<año_inicio>–<año_fin>}}
       {\statpill{Tipo}{Multi-cultivo}}

  3. Fila de TRES big-number tiles (las cifras estrella) — extraelas de
     pricing_quote.csv y del slip:
     \begin{minipage}[t]{0.32\textwidth}\bignumtile{USD X,XM}{Pérdida media anual histórica}\end{minipage}\hfill
     \begin{minipage}[t]{0.32\textwidth}\bignumtile{USD X,XM}{Prima anual técnica}\end{minipage}\hfill
     \begin{minipage}[t]{0.32\textwidth}\bignumtile{USD XXM}{Pago máximo posible}\end{minipage}
     \par\vspace{8pt}

  4. Cuerpo en DOS COLUMNAS (~48\% cada una). Cada columna lleva 2
     paneles + 1 plot embebido al fondo. NO uses plots full-width — eso
     genera huérfanos y rompe el layout.

     \begin{minipage}[t]{0.48\textwidth}
       \suyanapanel{Qué cubre este producto}{<2 oraciones: protección
         frente a sequía severa durante la ventana de cultivo. Pago
         automático sin peritaje, dinero rápido.>}
       \par\vspace{6pt}
       \suyanapanel{Cuándo se activa el pago}{<2-3 oraciones explicando el
         gatillo en lenguaje del productor. Tres niveles: severo, muy
         severo, catastrófico. Sin mencionar P1/P5/P10.>}
       \par\vspace{6pt}
       \briefplot{loss_trends.png}{Pérdidas anuales históricas. Los años más
       severos se destacan en rojo.}
     \end{minipage}\hfill
     \begin{minipage}[t]{0.48\textwidth}
       \suyanapanel{Estructura de pagos}{<Tabla booktabs con tres filas —
       severo / muy severo / catastrófico — con el % del notional y el
         monto en USD. Usá \scriptsize\setlength{\tabcolsep}{3pt} antes
         de \begin{tabular}. Convertí P1/P5/P10 a USD usando el notional.>}
       \par\vspace{6pt}
       \suyanapanel{Ejemplo concreto}{<Tomá UN AÑO real (idealmente 2022
         si está). 1-2 oraciones: "En 2022 el índice cayó a X. Pago: USD
         Y." Sin matemática expuesta.>}
       \par\vspace{6pt}
       \briefplot{aep_portfolio.png}{Frecuencia histórica de pérdidas
       anuales. La línea vertical marca el promedio.}
     \end{minipage}

  5. Debajo del cuerpo, agregá una franja compacta de preguntas frecuentes:
     \par\vspace{6pt}
     {\color{sgreen}\bfseries\scriptsize PREGUNTAS FRECUENTES}\par\vspace{3pt}

     Cuerpo en DOS COLUMNAS de FAQs, exactamente 3 por columna (6 en
     total). Cada respuesta debe caber en UNA sola oración
     corta de ≤ 22 palabras. Si no cabe, simplificá hasta que quepa.
     La página única no debe desbordar bajo ninguna circunstancia. Mantené
     ambas columnas alineadas arriba. Usá \vspace{3pt} entre preguntas.

     COLUMNA IZQUIERDA:
       \faqitem{1}{¿Qué es un seguro paramétrico?}{<1 oración, ≤22 palabras>}
       \faqitem{2}{¿Cómo medimos la sequía?}{<1 oración, ≤22 palabras>}
       \faqitem{3}{¿Cómo se calcula el pago?}{<1 oración, ≤22 palabras>}

     COLUMNA DERECHA:
       \faqitem{4}{¿De dónde vienen los datos?}{<1 oración, ≤22 palabras>}
       \faqitem{5}{¿Cuándo recibo el dinero?}{<1 oración, ≤22 palabras>}
       \faqitem{6}{¿Qué cubre exactamente?}{<1 oración, ≤22 palabras>}

  6. CTA inferior:
     \vfill
     \suyanacta{¿Querés una cotización?}{Contactá a nuestro equipo en suyana.io}

  7. Footer disclaimer:
     \suyanafooter{Este material fue preparado por Suyana y es información
     confidencial. La información presentada no constituye una oferta de
     seguro ni asesoramiento. Toda cobertura paramétrica conlleva el
     riesgo de que el índice gatillo no esté perfectamente correlacionado
     con la pérdida subyacente.}

REGLAS DE TONO:
  - Tuteo rioplatense neutro consistente.
  - Oraciones ≤ 14 palabras.
  - Nada de "probabilidad anual de excedencia", "pérdida media anual",
    "tasa de prima". Hablá en dólares, hectáreas, años.
  - Si mencionás período de retorno, decí "aproximadamente 1 vez cada X
    años".
  - Cero referencias a metodología, fórmulas, ni cadenas matemáticas.
  - No inventes números — extraelos de los insumos. Si un dato no está,
    omitilo o usá un placeholder claro tipo "<por confirmar>".

Devolvé EXCLUSIVAMENTE el LaTeX del cuerpo, sin bloque markdown, sin
\\documentclass, sin texto fuera del LaTeX.
"""

_BRIEF_INSTRUCTION_EN = r"""
Write the BODY of a Suyana CLIENT BRIEF in LaTeX, as a COMPACT VERTICAL
DASHBOARD (A4 portrait), ONE PAGE. Audience: South American agricultural producer.
Tone: clear, commercial, didactic, zero technical jargon. No formulas.
No acronyms. Short sentences (≤ 14 words).

Do NOT include the preamble (I add it). Start directly after
\begin{document}, end before \end{document}. Use these pre-defined macros:
\suyanaheader, \suyanastats, \statpill, \bignumtile, \suyanapanel,
\briefplot, \faqitem, \suyanacta, \suyanafooter.

══════════════════════════════════════════════════════════════════════════
SINGLE PAGE — PRODUCT, MECHANICS AND FAQ
══════════════════════════════════════════════════════════════════════════

  1. Dark header:
     \suyanaheader{Parametric Drought Cover}{<region> · <month year>}

  2. Stats strip (6 pills, all \statpill{LABEL}{Value}):
     \suyanastats
       {\statpill{Region}{<province/country>}}
       {\statpill{Crops}{<crops from config>}}
       {\statpill{Window}{<season range>}}
       {\statpill{Notional}{USD <slip amount>}}
       {\statpill{Climatology}{<start>–<end>}}
       {\statpill{Type}{Multi-crop}}

  3. Row of THREE big-number tiles — extract from pricing_quote.csv and slip:
     \begin{minipage}[t]{0.32\textwidth}\bignumtile{USD X.XM}{Historical average annual loss}\end{minipage}\hfill
     \begin{minipage}[t]{0.32\textwidth}\bignumtile{USD X.XM}{Annual technical premium}\end{minipage}\hfill
     \begin{minipage}[t]{0.32\textwidth}\bignumtile{USD XXM}{Maximum possible payout}\end{minipage}
     \par\vspace{8pt}

  4. Body in TWO COLUMNS (~48\% each). Each column has 2 panels + 1
     embedded plot at the bottom. Do NOT use full-width plots — they
     create orphans and break the layout.

     LEFT COLUMN: \suyanapanel{What this cover protects} (2 sentences),
       \suyanapanel{When the payout triggers} (2–3 sentences, three levels:
       severe, very severe, catastrophic — no P1/P5/P10),
       \briefplot{loss_trends.png}{Historical annual losses. The most severe
       years are highlighted in red.}
     RIGHT COLUMN: \suyanapanel{Payout structure} (booktabs table converting
       % of notional to USD per level; use \scriptsize\setlength{\tabcolsep}{3pt}
       before \begin{tabular}), \suyanapanel{A real example} (1–2
       sentences, one real year, narrative-only result, no math),
       \briefplot{aep_portfolio.png}{Historical loss frequency. The vertical
       line marks the average.}

  5. Below the main body, add a compact FAQ band:
     \par\vspace{6pt}
     {\color{sgreen}\bfseries\scriptsize FREQUENTLY ASKED QUESTIONS}\par\vspace{3pt}

     Two columns of FAQs, exactly 3 per column (6 total).
     Each answer must fit in ONE short sentence ≤ 22 words. Simplify
     until it fits. The single page must not overflow. Keep both columns top-aligned.
     Use \faqitem{N}{Question}{Answer}.
     1. What is parametric insurance?
     2. How do we measure drought?
     3. How is the payout calculated?
     4. Where does the data come from?
     5. When do I receive the money?
     6. What exactly is covered?

  6. CTA: add \vfill, then
     \suyanacta{Want a quote?}{Reach our team at suyana.io}

  7. Footer disclaimer using \suyanafooter{...} — standard parametric
     coverage disclaimer (basis risk, not advice, confidential).

VOICE RULES:
  - Sentences ≤ 14 words.
  - No "annual exceedance probability", "average annual loss", "rate on line".
    Talk in dollars, hectares, years.
  - If you must mention return period, say "roughly once every X years".
  - No methodology, no formulas, no math chains.
  - Do not invent numbers — pull them from inputs. If missing, use a clear
    placeholder like "<TBC>".

Return ONLY the LaTeX body, no markdown fence, no \\documentclass, no text
outside the LaTeX.
"""

_PT_NOTE = (
    "Portuguese (pt-BR) follows the same content rules as English; produce "
    "all output in Portuguese using parametric-insurance Brazilian terminology."
)

LANGUAGES: dict[str, Language] = {
    "es": Language(
        code="es",
        babel="spanish",
        confidential="Confidencial",
        system_prompt=(
            "Sos un redactor senior de Suyana, una startup de seguros paramétricos. "
            "Producís entregables LaTeX listos para compilar siguiendo las "
            "convenciones de marca y el tono comercial de Suyana."
        ),
        style_rules=_STYLE_RULES_ES,
        article_instruction=_ARTICLE_INSTRUCTION_ES,
        beamer_instruction=_BEAMER_INSTRUCTION_ES,
        brief_instruction=_BRIEF_INSTRUCTION_ES,
    ),
    "en": Language(
        code="en",
        babel="english",
        confidential="Confidential",
        system_prompt=(
            "You are a senior writer at Suyana, a parametric insurance startup. "
            "You produce compile-ready LaTeX deliverables that follow Suyana "
            "brand conventions and commercial tone."
        ),
        style_rules=_STYLE_RULES_EN,
        article_instruction=_ARTICLE_INSTRUCTION_EN,
        beamer_instruction=_BEAMER_INSTRUCTION_EN,
        brief_instruction=_BRIEF_INSTRUCTION_EN,
    ),
    "pt": Language(
        code="pt",
        babel="portuguese",
        confidential="Confidencial",
        system_prompt=(
            "You are a senior writer at Suyana, a parametric insurance startup. "
            "Produce compile-ready LaTeX deliverables following Suyana brand "
            "conventions, writing all body text in Brazilian Portuguese."
        ),
        style_rules=_STYLE_RULES_EN + "\n\n" + _PT_NOTE,
        article_instruction=_ARTICLE_INSTRUCTION_EN + "\n\n" + _PT_NOTE,
        beamer_instruction=_BEAMER_INSTRUCTION_EN + "\n\n" + _PT_NOTE,
        brief_instruction=_BRIEF_INSTRUCTION_EN + "\n\n" + _PT_NOTE,
    ),
}


# ---------------------------------------------------------------------------
# Build Claude content blocks — text + images (multimodal)
# ---------------------------------------------------------------------------


def _b64(path: Path) -> str:
    return base64.standard_b64encode(path.read_bytes()).decode("ascii")


def build_context_blocks(inputs: Inputs) -> list[dict]:
    """Assemble the shared context (configs, summaries, plots) sent to Claude."""
    blocks: list[dict] = []

    # 1. Structured numerical / config snapshot as a single text block.
    structured = {
        "region": inputs.region,
        "pricing_config": inputs.pricing_config,
        "etl_parameters": inputs.etl_parameters,
        "etl_globals": inputs.etl_globals,
        "parquet_summaries": inputs.parquet_summaries,
    }
    blocks.append(
        {
            "type": "text",
            "text": (
                "INSUMOS ESTRUCTURADOS DEL PIPELINE (JSON):\n```json\n"
                + json.dumps(structured, indent=2, default=str, ensure_ascii=False)
                + "\n```"
            ),
        }
    )

    # 2. Narrative artifacts verbatim.
    blocks.append(
        {
            "type": "text",
            "text": (
                "REPORTE DE ASEGURAMIENTO DE CALIDAD (markdown):\n"
                f"{inputs.qa_report_md or '(no disponible)'}\n\n"
                "SLIP DE CONTRATO (markdown):\n"
                f"{inputs.contract_slip_md or '(no disponible)'}\n\n"
                "COTIZACIÓN (CSV):\n"
                f"{inputs.pricing_quote_csv or '(no disponible)'}"
            ),
        }
    )

    # 3. Plot index — names first, so Claude can reference them by filename.
    plot_index = "\n".join(f"- {p.name}" for p in inputs.plot_files)
    blocks.append(
        {
            "type": "text",
            "text": (
                "GRÁFICOS DISPONIBLES (los verás como imágenes a continuación). "
                "Usalos todos en el entregable, referenciándolos por su nombre "
                "exacto en \\includegraphics{<nombre>}:\n" + plot_index
            ),
        }
    )

    # 4. Each plot as a vision input with a label.
    for p in inputs.plot_files:
        blocks.append({"type": "text", "text": f"[Figura: {p.name}]"})
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": _b64(p),
                },
            }
        )
    return blocks


# ---------------------------------------------------------------------------
# Claude call
# ---------------------------------------------------------------------------


def call_claude(
    client: Anthropic,
    model: str,
    context_blocks: list[dict],
    instruction: str,
    lang: Language,
) -> str:
    cached_blocks = context_blocks + [
        {"type": "text", "text": lang.style_rules, "cache_control": {"type": "ephemeral"}}
    ]
    message = client.messages.create(
        model=model,
        max_tokens=16000,
        system=lang.system_prompt,
        messages=[
            {
                "role": "user",
                "content": cached_blocks + [{"type": "text", "text": instruction}],
            }
        ],
    )
    return "".join(b.text for b in message.content if b.type == "text")


# ---------------------------------------------------------------------------
# Output assembly
# ---------------------------------------------------------------------------


def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        s = s[first_nl + 1 :] if first_nl != -1 else s
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s.strip()


def _extract_body(s: str) -> str:
    """Strip a markdown fence and any wrapping LaTeX preamble Claude included.

    Claude is told not to emit \\documentclass or \\begin{document} in the body,
    but sometimes does anyway. Defensive parsing: if \\begin{document} appears,
    take only the content between begin/end.
    """
    s = _strip_code_fence(s)
    begin_match = re.search(r"\\begin\{document\}", s)
    if begin_match:
        s = s[begin_match.end():]
    end_match = re.search(r"\\end\{document\}", s)
    if end_match:
        s = s[: end_match.start()]
    return s.strip()


def assemble_article(body: str, lang: Language) -> str:
    body = _extract_body(body)
    preamble = (
        ARTICLE_PREAMBLE
        .replace("__BABEL__", lang.babel)
        .replace("__CONFIDENTIAL__", lang.confidential)
    )
    return preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n"


def assemble_brief(body: str, lang: Language) -> str:
    body = _extract_body(body)
    preamble = BRIEF_PREAMBLE.replace("__BABEL__", lang.babel)
    return preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n"


def assemble_beamer(body: str, product_name: str, month_year: str, lang: Language) -> str:
    body = _extract_body(body)
    body = body.replace(r"\newcommand{\secslide}", r"\renewcommand{\secslide}")
    preamble = (
        BEAMER_PREAMBLE
        .replace("__BABEL__", lang.babel)
        .replace("__CONFIDENTIAL__", lang.confidential)
        .replace("__PRODUCT_NAME__", product_name)
        .replace("__MONTH_YEAR__", month_year)
    )
    return preamble + "\n\\begin{document}\n" + body + "\n\\end{document}\n"


def stage_output_folder(out_dir: Path, plots: list[Path], brand_dir: Path | None) -> None:
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "brand").mkdir(parents=True, exist_ok=True)
    for p in plots:
        shutil.copy2(p, out_dir / "figures" / p.name)
    if brand_dir and brand_dir.exists():
        for logo in brand_dir.glob("logo_*.png"):
            shutil.copy2(logo, out_dir / "brand" / logo.name)


def compile_pdf(tex_path: Path) -> bool:
    """Run pdflatex twice. Return True on success.

    pdflatex emits some Latin-1 in font dictionary lines, so we capture as
    bytes and decode with errors='replace' to avoid UnicodeDecodeError.
    """
    cwd = tex_path.parent
    last_stdout = b""
    for _ in range(2):
        proc = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", tex_path.name],
            cwd=cwd,
            capture_output=True,
        )
        last_stdout = proc.stdout
        if proc.returncode != 0:
            print(f"  ✗ pdflatex failed for {tex_path.name}; see {tex_path.with_suffix('.log')}")
            tail = last_stdout.decode("utf-8", errors="replace").splitlines()[-15:]
            for line in tail:
                print(f"    {line}")
            return False

    # Surface page count + any overfull warnings — caller cares about both.
    log = (cwd / (tex_path.stem + ".log")).read_text(errors="replace")
    pages_match = re.search(r"Output written on .* \((\d+) pages?", log)
    pages = pages_match.group(1) if pages_match else "?"
    overfull = sum(1 for line in log.splitlines() if "Overfull" in line)
    print(f"  ✓ {tex_path.with_suffix('.pdf').name}: {pages} pages, {overfull} overfull warnings")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    here = Path(__file__).parent.resolve()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--outputs-dir", type=Path, default=here / "outputs")
    parser.add_argument("--pricing-config", type=Path, default=here / "config.yaml")
    parser.add_argument("--etl-conf", type=Path, default=here.parent / "etl_pipeline" / "conf" / "base")
    parser.add_argument("--plots-dir", type=Path, default=here / "outputs" / "plots")
    parser.add_argument("--brand-dir", type=Path, default=here.parent / "brand")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    parser.add_argument("--region", default="Argentina")
    parser.add_argument("--product-name", default="Producto Paramétrico de Sequía")
    parser.add_argument("--month-year", default="Abril 2026")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--skip-compile", action="store_true", help="Generate .tex but don't run pdflatex")
    parser.add_argument(
        "--emit-pptx",
        action="store_true",
        help=(
            "Also create an editable native presentation.pptx based on the Beamer deck. "
            "Requires python-pptx."
        ),
    )
    parser.add_argument(
        "--only",
        choices=["report", "brief", "deck", "all"],
        default="all",
        help="Which deliverable(s) to produce.",
    )
    parser.add_argument(
        "--lang",
        choices=sorted(LANGUAGES),
        default="es",
        help="Output language (es, en, pt). Switches babel, footer labels, and the prompt to Claude.",
    )
    args = parser.parse_args()
    lang = LANGUAGES[args.lang]

    api_key = load_api_key()
    client = Anthropic(api_key=api_key)

    print(f"Gathering inputs from {args.outputs_dir} ...")
    inputs = gather_inputs(
        outputs_dir=args.outputs_dir,
        pricing_config_path=args.pricing_config,
        etl_conf_dir=args.etl_conf if args.etl_conf.exists() else None,
        plots_dir=args.plots_dir,
        region=args.region,
    )
    print(f"  {len(inputs.parquet_summaries)} parquets, {len(inputs.plot_files)} plots.")

    args.out.mkdir(parents=True, exist_ok=True)
    stage_output_folder(args.out, inputs.plot_files, args.brand_dir)

    snapshot = {
        "region": inputs.region,
        "pricing_config": inputs.pricing_config,
        "parquet_summaries": inputs.parquet_summaries,
        "plot_files": [p.name for p in inputs.plot_files],
        "model": args.model,
    }
    (args.out / "prompt_inputs.json").write_text(
        json.dumps(snapshot, indent=2, default=str, ensure_ascii=False)
    )

    print("Building Claude context (text + images) ...")
    context_blocks = build_context_blocks(inputs)

    want = {"report", "brief", "deck"} if args.only == "all" else {args.only}

    if "report" in want:
        print(f"Calling Claude ({args.model}) for the technical report [{lang.code}] ...")
        body = call_claude(
            client, args.model, context_blocks, lang.article_instruction, lang
        )
        path = args.out / "report.tex"
        path.write_text(assemble_article(body, lang))
        print(f"  wrote {path}")
        if not args.skip_compile:
            compile_pdf(path)

    if "brief" in want:
        print(f"Calling Claude ({args.model}) for the client brief [{lang.code}] ...")
        body = call_claude(
            client, args.model, context_blocks, lang.brief_instruction, lang
        )
        path = args.out / "brief.tex"
        path.write_text(assemble_brief(body, lang))
        print(f"  wrote {path}")
        if not args.skip_compile:
            compile_pdf(path)

    if "deck" in want:
        print(f"Calling Claude ({args.model}) for the deck [{lang.code}] ...")
        body = call_claude(
            client, args.model, context_blocks, lang.beamer_instruction, lang
        )
        path = args.out / "presentation.tex"
        path.write_text(assemble_beamer(body, args.product_name, args.month_year, lang))
        print(f"  wrote {path}")
        if not args.skip_compile:
            compile_pdf(path)
        if args.emit_pptx:
            from beamer_to_pptx import create_editable_deck

            pptx_path = create_editable_deck(args.out)
            print(f"  ✓ {pptx_path.name}: editable native PPTX")

    print(f"Done. See {args.out}/")


if __name__ == "__main__":
    main()
