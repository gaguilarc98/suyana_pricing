# Pricing Automation

End-to-end pipeline for climate risk pricing.

## Pipeline

```
Parameters (Rodrigo)
      |
      v
Data Request Service (Daniela)   ← data-request/
      |
      v
ETL (Gabriel)                    ← etl/
      |
      v
PostgreSQL DB
      |
      v
Pricing Analyst                  ← pricing/
      |
      v
Outputs
```

## Components

| Folder | Owner | Description |
|--------|-------|-------------|
| `data-request/` | Daniela | Downloads climate data (ERA5, CHIRPS) for a given peril and AOI |
| `etl/` | Gabriel | Transforms .nc files and loads into PostgreSQL |
| `pricing/` | Pricing team | Pricing models and outputs |
| `docs/` | All | Shared documentation |

## Setup

Each component has its own `requirements.txt`. Install only what you need:

```bash
pip install -r data-request/requirements.txt
```

Credentials go in a `.env` file (see `.env.example` in each component folder).

## Workflow

- `main` is always stable — do not push directly
- Create a branch for your work: `git checkout -b yourname/description`
- Open a Pull Request when ready → get one review → merge
