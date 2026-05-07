"""Project settings. There is no need to edit this file unless you want to change values
from the Kedro defaults. For further information, including these default values, see
https://docs.kedro.org/en/stable/configure/configuration_basics/#configuration"""

# Instantiated project hooks.
# For example, after creating a hooks.py and defining a ProjectHooks class there, do
from pricing_automation.hooks import ProjectHooks
# Hooks are executed in a Last-In-First-Out (LIFO) order.
HOOKS = (ProjectHooks(),)

# Installed plugins for which to disable hook auto-registration.
# DISABLE_HOOKS_FOR_PLUGINS = ("kedro-viz",)

# Class that manages storing KedroSession data.
# from kedro.framework.session.store import BaseSessionStore
# SESSION_STORE_CLASS = BaseSessionStore
# Keyword arguments to pass to the `SESSION_STORE_CLASS` constructor.
# SESSION_STORE_ARGS = {
#     "path": "./sessions"
# }

# Directory that holds configuration.
# CONF_SOURCE = "conf"

# Class that manages how configuration is loaded.
from kedro.config import OmegaConfigLoader
from omegaconf.resolvers import oc

from dotenv import load_dotenv
from pathlib import Path

# Get project root (one level above src/)
BASE_DIR = Path(__file__).resolve().parents[2]
# Load .env from project root explicitly
load_dotenv(BASE_DIR / ".env", override=True) #overrride otherwise it picks any credentials saved in the cache

# Keyword arguments to pass to the `CONFIG_LOADER_CLASS` constructor.
CONFIG_LOADER_ARGS = {
    "base_env": "base",
    "default_run_env": "local",
    "config_patterns": {
        "spark" : ["spark*/"],
        "parameters": ["parameters*", "parameters*/**", "**/parameters*"],
        "credentials": ["credentials*", "credentials*/**", "**/credentials*"]
    },
    "custom_resolvers": {
        "oc.env": oc.env,
    }
}

# Class that manages Kedro's library components.
# from kedro.framework.context import KedroContext
# CONTEXT_CLASS = KedroContext

# Class that manages the Data Catalog.
# from kedro.io import DataCatalog
# DATA_CATALOG_CLASS = DataCatalog
