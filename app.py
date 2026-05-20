"""
Suyana Pricing Tool — simple desktop UI
Run with: python app.py
"""

import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import tempfile
import os
from pathlib import Path
from datetime import date

# ── Config ────────────────────────────────────────────────────────────────────

REPO_ROOT   = Path(__file__).parent
PIPELINE_DIR = REPO_ROOT / "etl_pipeline"

PIPELINES  = ["planet", "swc", "prcp", "temp"]
PROVIDERS  = ["ERA5", "UCSB", "Planet"]
VARIABLES  = ["swc", "prcp", "tmax", "tmin"]
DIRECTIONS = ["lower (drought / frost)", "upper (excess rain / heat)"]
COUNTRIES  = ["bolivia", "colombia"]
DATA_LOCS  = ["S3 (remote)", "Local"]

LOCATIONS = {
    "bolivia": ["Mizque", "Quillacollo", "Colomi", "Tarija", "Sacaba", "Cercado"],
    "colombia": ["Monteria", "Tierralta", "Valencia", "Lorica"],
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def get_aws_env(key: str = "", secret: str = "", token: str = "") -> dict:
    """
    Build env vars for the subprocess.
    Priority: fields typed in the UI > ~/.aws/credentials > current environment.
    """
    env = os.environ.copy()

    # Start from ~/.aws/credentials as baseline
    creds_path = Path.home() / ".aws" / "credentials"
    if creds_path.exists():
        import configparser
        cfg = configparser.ConfigParser()
        cfg.read(creds_path)
        if cfg.has_section("default"):
            env["AWS_ACCESS_KEY_ID"]     = cfg.get("default", "aws_access_key_id", fallback="")
            env["AWS_SECRET_ACCESS_KEY"] = cfg.get("default", "aws_secret_access_key", fallback="")
            env["AWS_SESSION_TOKEN"]     = cfg.get("default", "aws_session_token", fallback="")

    # UI fields override everything (and update ~/.aws/credentials for future runs)
    if key.strip():
        env["AWS_ACCESS_KEY_ID"]     = key.strip()
        env["AWS_SECRET_ACCESS_KEY"] = secret.strip()
        env["AWS_SESSION_TOKEN"]     = token.strip()
        _save_credentials(key.strip(), secret.strip(), token.strip())

    return env


LOCAL_DATA_PATH = Path.home() / "suyana_data"

def s3_outputs(country: str, lead_id: str = "valles") -> dict:
    """Return a dict of label → S3 path for all pipeline outputs."""
    base = f"s3://suyana-pricing/{country}/{lead_id}"
    return {
        "pricing":    f"{base}/outputs/pricing_quote.parquet",
        "triggers":   f"{base}/outputs/ERA5_swc_window_triggers.parquet",
        "aep":        f"{base}/displays/aep_portfolio.png",
        "aep_crop":   f"{base}/displays/aep_per_crop.png",
        "trigger_map":f"{base}/displays/trigger_frequency_map.png",
        "anomaly":    f"{base}/displays/anomaly_timeseries.png",
    }


def local_outputs(country: str, lead_id: str) -> dict:
    """Return a dict of label → local Path for all pipeline outputs."""
    base = LOCAL_DATA_PATH / country / lead_id
    return {
        "pricing":    base / "outputs" / "pricing_quote.parquet",
        "triggers":   base / "outputs" / "ERA5_swc_window_triggers.parquet",
        "aep":        base / "displays" / "aep_portfolio.png",
        "aep_crop":   base / "displays" / "aep_per_crop.png",
        "trigger_map":base / "displays" / "trigger_frequency_map.png",
        "anomaly":    base / "displays" / "anomaly_timeseries.png",
    }


def download_s3_file(s3_path: str, env: dict) -> Path:
    """Download a file from S3 to a temp dir and return the local path."""
    import s3fs
    fs = s3fs.S3FileSystem(
        key=env.get("AWS_ACCESS_KEY_ID"),
        secret=env.get("AWS_SECRET_ACCESS_KEY"),
        token=env.get("AWS_SESSION_TOKEN"),
    )
    suffix = Path(s3_path).suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.close()
    fs.get(s3_path.replace("s3://", ""), tmp.name)
    return Path(tmp.name)


def _save_credentials(key: str, secret: str, token: str):
    """Persist credentials to ~/.aws/credentials so they survive app restarts."""
    creds_path = Path.home() / ".aws" / "credentials"
    creds_path.parent.mkdir(exist_ok=True)
    creds_path.write_text(
        f"[default]\n"
        f"aws_access_key_id = {key}\n"
        f"aws_secret_access_key = {secret}\n"
        f"aws_session_token = {token}\n"
    )


def write_params(pipeline: str, country: str, locations: list[str],
                 date_start: str, date_end: str,
                 variable: str = "swc", provider: str = "ERA5",
                 trigger_side: str = "lower", lead_id: str = "valles",
                 use_local: bool = False):
    """Overwrite globals.yml and relevant sections of parameters.yml for the run."""
    if use_local:
        data_path = str(LOCAL_DATA_PATH)
    else:
        data_path = "s3://suyana-pricing"

    # globals.yml
    globals_path = PIPELINE_DIR / "conf" / "base" / "globals.yml"
    globals_path.write_text(
        f"data_path: {data_path}\n"
        f"country: {country}\n"
        f"provider: {provider}\n"
        f"field: {variable}\n"
        f"version: ''\n"
    )

    # parameters.yml — update locations and trigger side
    params_path = PIPELINE_DIR / "conf" / "base" / "parameters.yml"
    lines = params_path.read_text().splitlines()
    new_lines = []
    inside_include = False
    for line in lines:
        # Update location list
        if line.strip().startswith("include:"):
            inside_include = True
            loc_str = ", ".join(f"'{l}'" for l in locations)
            new_lines.append(f"  include:")
            new_lines.append(f"    municipio: [{loc_str}]")
            continue
        if inside_include and line.strip().startswith("municipio:"):
            continue
        if inside_include and line.startswith("  ") and not line.startswith("    "):
            inside_include = False
        # Update trigger side
        if line.strip().startswith("'side':"):
            indent = len(line) - len(line.lstrip())
            new_lines.append(" " * indent + f"'side': '{trigger_side}'")
            continue
        new_lines.append(line)
    params_path.write_text("\n".join(new_lines))


# ── Main window ───────────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Suyana Pricing Tool")
        self.resizable(True, True)
        self.minsize(700, 600)
        self._process = None
        self._build_ui()

    # ── UI layout ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        pad = dict(padx=12, pady=6)

        # ── Title ─────────────────────────────────────────────────────────────
        tk.Label(self, text="Suyana Pricing Tool",
                 font=("Helvetica", 18, "bold")).pack(pady=(16, 4))
        tk.Label(self, text="Configure and run the pricing pipeline",
                 font=("Helvetica", 11), fg="#555").pack(pady=(0, 12))

        # ── AWS Credentials ───────────────────────────────────────────────────
        aws = tk.LabelFrame(self, text="AWS Credentials  (paste fresh SSO tokens here)",
                            padx=12, pady=8)
        aws.pack(fill="x", padx=16, pady=4)
        self._aws_frame = aws

        # Status indicator
        self._cred_status = tk.Label(aws, text="", font=("Helvetica", 10))
        self._cred_status.grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self._refresh_cred_status()

        tk.Label(aws, text="Access Key ID:").grid(row=1, column=0, sticky="w", padx=6)
        self.aws_key = tk.StringVar()
        tk.Entry(aws, textvariable=self.aws_key, width=28
                 ).grid(row=1, column=1, sticky="w", padx=6)

        tk.Label(aws, text="Secret Access Key:").grid(row=2, column=0, sticky="w", padx=6, pady=4)
        self.aws_secret = tk.StringVar()
        tk.Entry(aws, textvariable=self.aws_secret, width=28, show="*"
                 ).grid(row=2, column=1, sticky="w", padx=6)

        tk.Label(aws, text="Session Token:").grid(row=3, column=0, sticky="w", padx=6)
        self.aws_token = tk.StringVar()
        token_entry = tk.Text(aws, height=3, width=60, wrap="none",
                              font=("Courier", 9))
        token_entry.grid(row=3, column=1, columnspan=3, sticky="w", padx=6, pady=2)
        self._token_entry = token_entry  # keep reference

        tk.Button(aws, text="💾  Save credentials", padx=8, pady=3,
                  command=self._save_creds).grid(row=4, column=1, sticky="w",
                                                  padx=6, pady=(6, 2))

        # ── Config frame ──────────────────────────────────────────────────────
        cfg = tk.LabelFrame(self, text="Configuration", padx=12, pady=8)
        cfg.pack(fill="x", padx=16, pady=4)
        self._cfg_frame = cfg

        # Pipeline
        tk.Label(cfg, text="Pipeline:").grid(row=0, column=0, sticky="w", **pad)
        self.pipeline_var = tk.StringVar(value="planet")
        ttk.Combobox(cfg, textvariable=self.pipeline_var,
                     values=PIPELINES, state="readonly", width=14
                     ).grid(row=0, column=1, sticky="w", **pad)

        # Country
        tk.Label(cfg, text="Country:").grid(row=1, column=0, sticky="w", **pad)
        self.country_var = tk.StringVar(value="bolivia")
        country_cb = ttk.Combobox(cfg, textvariable=self.country_var,
                                   values=COUNTRIES, state="readonly", width=14)
        country_cb.grid(row=1, column=1, sticky="w", **pad)
        country_cb.bind("<<ComboboxSelected>>", self._on_country_change)

        # Lead ID
        tk.Label(cfg, text="Lead ID:").grid(row=2, column=0, sticky="w", **pad)
        self.lead_id_var = tk.StringVar(value="valles")
        tk.Entry(cfg, textvariable=self.lead_id_var, width=14
                 ).grid(row=2, column=1, sticky="w", **pad)

        # Data location
        tk.Label(cfg, text="Data:").grid(row=2, column=2, sticky="w", **pad)
        self.data_loc_var = tk.StringVar(value="S3 (remote)")
        data_cb = ttk.Combobox(cfg, textvariable=self.data_loc_var,
                               values=DATA_LOCS, state="readonly", width=12)
        data_cb.grid(row=2, column=3, sticky="w", **pad)
        data_cb.bind("<<ComboboxSelected>>", self._on_data_loc_change)

        # Variable (peril)
        tk.Label(cfg, text="Variable:").grid(row=3, column=0, sticky="w", **pad)
        self.variable_var = tk.StringVar(value="swc")
        ttk.Combobox(cfg, textvariable=self.variable_var,
                     values=VARIABLES, state="readonly", width=10
                     ).grid(row=3, column=1, sticky="w", **pad)

        # Provider
        tk.Label(cfg, text="Provider:").grid(row=3, column=2, sticky="w", **pad)
        self.provider_var = tk.StringVar(value="ERA5")
        ttk.Combobox(cfg, textvariable=self.provider_var,
                     values=PROVIDERS, state="readonly", width=10
                     ).grid(row=3, column=3, sticky="w", **pad)

        # Trigger direction
        tk.Label(cfg, text="Trigger:").grid(row=4, column=0, sticky="w", **pad)
        self.direction_var = tk.StringVar(value=DIRECTIONS[0])
        ttk.Combobox(cfg, textvariable=self.direction_var,
                     values=DIRECTIONS, state="readonly", width=28
                     ).grid(row=4, column=1, columnspan=3, sticky="w", **pad)

        # Date range
        tk.Label(cfg, text="Start date:").grid(row=0, column=2, sticky="w", **pad)
        self.date_start = tk.StringVar(value="2002-06-19")
        tk.Entry(cfg, textvariable=self.date_start, width=14
                 ).grid(row=0, column=3, sticky="w", **pad)

        tk.Label(cfg, text="End date:").grid(row=1, column=2, sticky="w", **pad)
        self.date_end = tk.StringVar(value=str(date.today()))
        tk.Entry(cfg, textvariable=self.date_end, width=14
                 ).grid(row=1, column=3, sticky="w", **pad)

        # Locations
        loc_frame = tk.LabelFrame(self, text="Locations", padx=12, pady=8)
        loc_frame.pack(fill="x", padx=16, pady=4)

        self._loc_vars = {}
        self._loc_frame = loc_frame
        self._build_location_checkboxes("bolivia")

        # ── Results ───────────────────────────────────────────────────────────
        res = tk.LabelFrame(self, text="Results", padx=12, pady=8)
        res.pack(fill="x", padx=16, pady=4)

        tk.Label(res, text="Available after a successful run:",
                 font=("Helvetica", 10), fg="#888").grid(
                 row=0, column=0, columnspan=4, sticky="w")

        result_buttons = [
            ("📊 Pricing Table",     self._show_pricing),
            ("📈 AEP Curve",         lambda: self._show_image("aep")),
            ("🗺️  Trigger Map",      lambda: self._show_image("trigger_map")),
            ("📉 Anomaly Timeseries",lambda: self._show_image("anomaly")),
        ]
        self._result_btns = []
        for i, (label, cmd) in enumerate(result_buttons):
            b = tk.Button(res, text=label, padx=10, pady=4,
                          state="disabled", command=cmd)
            b.grid(row=1, column=i, padx=6, pady=6, sticky="w")
            self._result_btns.append(b)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_frame = tk.Frame(self)
        btn_frame.pack(pady=8)

        self.run_btn = tk.Button(btn_frame, text="▶  Run Pipeline",
                                 font=("Helvetica", 12, "bold"),
                                 bg="#2e7d32", fg="white", padx=16, pady=6,
                                 command=self._run)
        self.run_btn.pack(side="left", padx=8)

        self.stop_btn = tk.Button(btn_frame, text="⏹  Stop",
                                  font=("Helvetica", 12), padx=16, pady=6,
                                  state="disabled", command=self._stop)
        self.stop_btn.pack(side="left", padx=8)

        tk.Button(btn_frame, text="🗑  Clear logs",
                  font=("Helvetica", 12), padx=16, pady=6,
                  command=self._clear_logs).pack(side="left", padx=8)

        # ── Logs ──────────────────────────────────────────────────────────────
        tk.Label(self, text="Pipeline logs", font=("Helvetica", 11, "bold")
                 ).pack(anchor="w", padx=16)
        self.log_box = scrolledtext.ScrolledText(
            self, font=("Courier", 10), bg="#1e1e1e", fg="#d4d4d4",
            wrap="word", state="disabled", height=20
        )
        self.log_box.pack(fill="both", expand=True, padx=16, pady=(4, 16))

        # colour tags for log levels
        self.log_box.tag_config("INFO",    foreground="#4ec9b0")
        self.log_box.tag_config("WARNING", foreground="#ce9178")
        self.log_box.tag_config("ERROR",   foreground="#f44747")
        self.log_box.tag_config("DONE",    foreground="#b5cea8")
        self.log_box.tag_config("plain",   foreground="#d4d4d4")

    def _build_location_checkboxes(self, country: str):
        for w in self._loc_frame.winfo_children():
            w.destroy()
        self._loc_vars = {}
        locs = LOCATIONS.get(country, [])
        for i, loc in enumerate(locs):
            var = tk.BooleanVar(value=True)
            tk.Checkbutton(self._loc_frame, text=loc, variable=var
                           ).grid(row=i // 4, column=i % 4, sticky="w", padx=8, pady=2)
            self._loc_vars[loc] = var

    def _on_country_change(self, _event=None):
        self._build_location_checkboxes(self.country_var.get())

    def _on_data_loc_change(self, _event=None):
        is_local = self.data_loc_var.get() == "Local"
        # Hide/show AWS section
        if is_local:
            self._aws_frame.pack_forget()
            # Enable result buttons immediately — local files may already exist
            self._enable_results()
        else:
            self._aws_frame.pack(fill="x", padx=16, pady=4,
                                  before=self._cfg_frame)
            # Disable result buttons until a run completes (S3 mode)
            for b in self._result_btns:
                b.config(state="disabled")

    # ── Credentials ───────────────────────────────────────────────────────────

    def _refresh_cred_status(self):
        """Show whether credentials are already stored."""
        creds_path = Path.home() / ".aws" / "credentials"
        if creds_path.exists():
            import configparser
            cfg = configparser.ConfigParser()
            cfg.read(creds_path)
            key = cfg.get("default", "aws_access_key_id", fallback="")
            if key:
                short = key[:8] + "..." + key[-4:]
                self._cred_status.config(
                    text=f"✅  Stored credentials: {short}  (paste new ones below if expired)",
                    fg="#2e7d32")
                return
        self._cred_status.config(
            text="⚠  No credentials found — paste them below before running.",
            fg="#b71c1c")

    def _save_creds(self):
        key    = self.aws_key.get().strip()
        secret = self.aws_secret.get().strip()
        token  = self._token_entry.get("1.0", "end").strip()
        if not key or not secret or not token:
            self._log("⚠  Please fill in all three credential fields.", "WARNING")
            return
        _save_credentials(key, secret, token)
        self._log("💾  Credentials saved.", "DONE")
        self._refresh_cred_status()
        # Clear the fields (token is sensitive)
        self.aws_key.set("")
        self.aws_secret.set("")
        self._token_entry.delete("1.0", "end")

    # ── Logging ───────────────────────────────────────────────────────────────

    def _log(self, text: str, tag: str = "plain"):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text + "\n", tag)
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    def _clear_logs(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _classify(self, line: str) -> str:
        if "ERROR" in line or "Traceback" in line or "Exception" in line:
            return "ERROR"
        if "WARNING" in line or "WARN" in line:
            return "WARNING"
        if "Completed" in line or "successfully" in line or "done" in line.lower():
            return "DONE"
        if "INFO" in line:
            return "INFO"
        return "plain"

    # ── Run / Stop ────────────────────────────────────────────────────────────

    def _run(self):
        selected = [loc for loc, var in self._loc_vars.items() if var.get()]
        if not selected:
            self._log("⚠ Please select at least one location.", "WARNING")
            return

        pipeline = self.pipeline_var.get()
        country  = self.country_var.get()

        side = "lower" if self.direction_var.get().startswith("lower") else "upper"
        self._log(f"─── Starting {pipeline} / {country} ───", "DONE")
        self._log(f"    Locations : {', '.join(selected)}", "plain")
        self._log(f"    Variable  : {self.variable_var.get()} ({self.provider_var.get()})", "plain")
        self._log(f"    Trigger   : {side}", "plain")
        self._log(f"    Dates     : {self.date_start.get()} → {self.date_end.get()}", "plain")

        # Parse trigger direction
        side = "lower" if self.direction_var.get().startswith("lower") else "upper"

        # Update config files
        use_local = self.data_loc_var.get() == "Local"
        write_params(pipeline, country, selected,
                     self.date_start.get(), self.date_end.get(),
                     variable=self.variable_var.get(),
                     provider=self.provider_var.get(),
                     trigger_side=side,
                     lead_id=self.lead_id_var.get(),
                     use_local=use_local)

        # Build command
        lead_id = self.lead_id_var.get().strip() or "valles"
        cmd = ["/opt/anaconda3/bin/kedro", "run", "--pipeline", pipeline,
               "--params", f"lead_id={lead_id}"]
        env = get_aws_env(
            key=self.aws_key.get(),
            secret=self.aws_secret.get(),
            token=self._token_entry.get("1.0", "end"),
        )

        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")

        def _stream():
            self._process = subprocess.Popen(
                cmd, cwd=str(PIPELINE_DIR),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, env=env, bufsize=1
            )
            for line in self._process.stdout:
                line = line.rstrip()
                if line:
                    tag = self._classify(line)
                    self.after(0, self._log, line, tag)

            self._process.wait()
            rc = self._process.returncode
            if rc == 0:
                self.after(0, self._log, "✅ Pipeline completed successfully!", "DONE")
                self.after(0, self._enable_results)
            else:
                self.after(0, self._log, f"❌ Pipeline failed (exit code {rc})", "ERROR")
            self.after(0, self._reset_buttons)

        threading.Thread(target=_stream, daemon=True).start()

    def _stop(self):
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._log("⏹ Pipeline stopped by user.", "WARNING")
        self._reset_buttons()

    def _reset_buttons(self):
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def _enable_results(self):
        for b in self._result_btns:
            b.config(state="normal")

    def _get_env(self) -> dict:
        return get_aws_env(
            key=self.aws_key.get(),
            secret=self.aws_secret.get(),
            token=self._token_entry.get("1.0", "end"),
        )

    # ── Results popups ────────────────────────────────────────────────────────

    def _show_image(self, key: str):
        country  = self.country_var.get()
        lead_id  = self.lead_id_var.get().strip() or "valles"
        use_local = self.data_loc_var.get() == "Local"

        if use_local:
            local = local_outputs(country, lead_id)[key]
            if not local.exists():
                self._log(f"❌ File not found: {local}", "ERROR")
                return
            self._open_image_window(local, key)
            return

        paths   = s3_outputs(country, lead_id)
        s3_path = paths[key]

        def _load():
            try:
                self.after(0, self._log, f"⬇️  Downloading from {s3_path}…", "INFO")
                local = download_s3_file(s3_path, self._get_env())
                self.after(0, self._log, f"✅ Downloaded to {local}", "DONE")
                self.after(0, self._open_image_window, local, key)
            except Exception as e:
                self.after(0, self._log, f"❌ Could not load {key}: {e}", "ERROR")

        threading.Thread(target=_load, daemon=True).start()

    def _open_image_window(self, path: Path, title: str):
        try:
            from PIL import Image, ImageTk
        except ImportError:
            subprocess.run(["open", str(path)])
            return

        img = Image.open(path)
        img.thumbnail((900, 700), Image.LANCZOS)

        win = tk.Toplevel(self)
        win.title(title.replace("_", " ").title())

        # Create PhotoImage ONCE and keep reference on the label
        photo = ImageTk.PhotoImage(img)
        lbl = tk.Label(win, image=photo)
        lbl.photo = photo          # prevents garbage collection
        lbl.pack(padx=8, pady=8)
        win.update_idletasks()
        win.lift()

    def _show_pricing(self):
        country   = self.country_var.get()
        lead_id   = self.lead_id_var.get().strip() or "valles"
        use_local = self.data_loc_var.get() == "Local"

        if use_local:
            local = local_outputs(country, lead_id)["pricing"]
            if not local.exists():
                self._log(f"❌ File not found: {local}", "ERROR")
                return
            import pandas as pd
            df = pd.read_parquet(local)
            self._open_table_window(df, "Pricing Quote")
            return

        paths = s3_outputs(country, lead_id)

        def _load():
            try:
                import pandas as pd
                local = download_s3_file(paths["pricing"], self._get_env())
                df = pd.read_parquet(local)
                self.after(0, self._open_table_window, df, "Pricing Quote")
            except Exception as e:
                self.after(0, self._log, f"❌ Could not load pricing table: {e}", "ERROR")

        threading.Thread(target=_load, daemon=True).start()
        self._log("⬇️  Downloading pricing table…", "INFO")

    def _open_table_window(self, df, title: str):
        win = tk.Toplevel(self)
        win.title(title)
        win.resizable(True, True)

        cols = list(df.columns)
        tree = ttk.Treeview(win, columns=cols, show="headings", height=min(len(df)+1, 20))

        for col in cols:
            tree.heading(col, text=col)
            tree.column(col, width=160, anchor="center")

        for _, row in df.iterrows():
            vals = []
            for v in row:
                try:
                    vals.append(f"{v:,.2f}" if isinstance(v, float) else str(v))
                except Exception:
                    vals.append(str(v))
            tree.insert("", "end", values=vals)

        vsb = ttk.Scrollbar(win, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(win, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        win.grid_rowconfigure(0, weight=1)
        win.grid_columnconfigure(0, weight=1)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = App()
    app.mainloop()
