import os
import shutil
import logging
import boto3
from botocore.credentials import InstanceMetadataProvider, InstanceMetadataFetcher
from pathlib import Path
from kedro.framework.hooks import hook_impl

logger = logging.getLogger(__name__)


class ProjectHooks:

    @hook_impl
    def before_pipeline_run(self, run_params, pipeline, catalog):
        self._setup_aws()
        self._ensure_local_paths(pipeline, catalog)

    def _setup_aws(self):
        # Clear SSO profile vars
        for var in ["AWS_PROFILE", "AWS_DEFAULT_PROFILE"]:
            os.environ.pop(var, None)

        # Block .aws config files
        os.environ["AWS_CONFIG_FILE"] = "/dev/null"
        os.environ["AWS_SHARED_CREDENTIALS_FILE"] = "/dev/null"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

        # Disable CRC32 checksum enforcement from botocore 1.37+
        os.environ["AWS_REQUEST_CHECKSUM_CALCULATION"] = "when_required"
        os.environ["AWS_RESPONSE_CHECKSUM_VALIDATION"] = "when_required"

        # Clear all AWS credential caches
        for cache_dir in ["sso/cache", "cli/cache"]:
            path = Path.home() / ".aws" / cache_dir
            if path.exists():
                shutil.rmtree(path)
                logger.info(f"AWS: cleared {cache_dir}")

        # Inject instance metadata credentials
        creds = self._try_instance_metadata()
        if creds is not None:
            logger.info("AWS: using EC2 instance metadata credentials")
            os.environ["AWS_ACCESS_KEY_ID"] = creds.access_key
            os.environ["AWS_SECRET_ACCESS_KEY"] = creds.secret_key
            os.environ["AWS_SESSION_TOKEN"] = creds.token
        else:
            logger.info("AWS: EC2 metadata not available, falling back to boto3 credential chain")

    def _try_instance_metadata(self):
        try:
            provider = InstanceMetadataProvider(
                iam_role_fetcher=InstanceMetadataFetcher(timeout=500, num_attempts=2)
            )
            return provider.load()
        except Exception:
            return None

    def _ensure_local_paths(self, pipeline, catalog):
        for dataset_name in pipeline.datasets():
            try:
                dataset = catalog._datasets[dataset_name]
                filepath = getattr(dataset, '_filepath', None)
                if filepath and not str(filepath).startswith("s3://"):
                    path = Path(str(filepath)).expanduser().resolve().parent
                    path.mkdir(parents=True, exist_ok=True)
            except (KeyError, AttributeError):
                pass