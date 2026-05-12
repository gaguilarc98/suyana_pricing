import re
import pandas as pd
import xarray as xr
import os
import fsspec
from pathlib import Path, PurePosixPath
from typing import Any, Dict
from kedro.io import AbstractDataset, AbstractVersionedDataset
from kedro.io.core import get_protocol_and_path, get_filepath_str, Version

from deltalake import DeltaTable
from deltalake.writer import write_deltalake

PROTOCOL_DELIMITER = "://"


class XarrayMultiFileDataset(AbstractDataset):
    """Custom Kedro dataset to handle multiple xarray files (just for reading)"""
    def __init__(
        self, 
        filepath: str, 
        load_args: dict = None,

    ):
        self._filepath = filepath
        self._load_args = load_args or {}

    def _describe(self):
        return dict(filepath=self._filepath, load_args=self._load_args)

    def _load(self) -> xr.Dataset:
        return xr.open_mfdataset(
            self._filepath,
            combine="by_coords",
            **self._load_args,
        )

    def _save(self, data: xr.Dataset) -> None:
        raise NotImplementedError("Saving not supported")
    
    
class NetCDFPartitionedDataset(AbstractDataset):
    """
    Single catalog entry for partitioned NetCDF files.
    Uses <partition> as wildcard in filepath template.
    
    Saves: dict[str, xr.Dataset] -> one file per key
    Loads: all matching files merged via open_mfdataset
    """
    def __init__(self, filepath, load_args=None, save_args=None):
        self._filepath = filepath
        self._load_args = load_args or {}
        self._save_args = save_args or {}

        # Split into directory and filename template
        p = Path(filepath)
        self._dir = p.parent
        self._filename_template = p.name  # e.g. 'swc_<partition>.nc'

    def _describe(self) -> dict:
        return dict(
            filepath = self._filepath,
            load_args = self._load_args,
            save_args = self._save_args,
        )
    
    def _resolve(self, partition_id: str) -> Path:
        filename = self._filename_template.replace('<partition>', str(partition_id))
        return self._dir / filename

    def _glob_pattern(self) -> str:
        return re.sub(r'<[^>]+>', '*', self._filename_template)

    def _load(self) -> xr.Dataset:
        files = sorted(self._dir.glob(self._glob_pattern()))
        if not files:
            raise FileNotFoundError(f"No files matching {self._glob_pattern()} in {self._dir}")
        return xr.open_mfdataset(files, **self._load_args)

    def _save(self, data: dict[str, xr.Dataset]) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        for partition_id, ds in data.items():
            ds.to_netcdf(self._resolve(partition_id), **self._save_args)


class XarrayZarrDataset(AbstractVersionedDataset[xr.Dataset, xr.Dataset]):
    """Custom Kedro dataset to handle xarray <-> Zarr I/O on local or S3."""

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] | None = None,
        save_args: Dict[str, Any] | None = None,
        version: Version | None = None,
        credentials: Dict[str, Any] | None = None,
        fs_args: Dict[str, Any] | None = None,
    ):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol

        _fs_args = fs_args or {}
        _credentials = credentials or {}

        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._fs = fsspec.filesystem(protocol, **_credentials, **_fs_args)

        super().__init__(
            filepath=PurePosixPath(path),
            version=version,
            exists_function=self._fs.exists,
            glob_function=self._fs.glob,
        )

        self._load_args = load_args or {"consolidated": True}
        self._save_args = save_args or {"mode": "w"}

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath, 
            protocol=self._protocol,
            load_args=self._load_args, 
            save_args=self._save_args
        )

    def _full_path(self) -> str:
        """Reconstruct full path with protocol prefix for xarray/fsspec."""
        path = self._fs.unstrip_protocol(str(self._filepath))
        return path

    def _load(self) -> xr.Dataset:
        load_path = self._full_path()
        return xr.open_zarr(load_path, **self._load_args)
        

    def _save(self, data: xr.Dataset) -> None:
        save_args = self._save_args.copy()
        save_path = self._full_path()

        chunking = save_args.pop("chunks", None)

        # Remove append_dim if writing fresh
        if not self._exists() or save_args.get("mode") == "w":
            save_args.pop("append_dim", None)

        if chunking:
            valid_chunks = {k: v for k, v in chunking.items() if k in data.dims}
            data = data.chunk(valid_chunks)

        data.to_zarr(save_path, **save_args)

    def _exists(self) -> bool:
        # Works for both local (directory) and S3 (object prefix)
        return self._fs.exists(str(self._filepath))


class ZarrPartitionedDataset(AbstractDataset):
    """
    Single catalog entry for partitioned Zarr stores on S3 or local filesystem.
    Uses <partition> as wildcard in filepath template.

    Saves: dict[str, xr.Dataset] -> one zarr store per key
    Loads: all matching stores merged via open_mfdataset(engine='zarr')

    Example filepaths:
        s3://my-bucket/data/swc_<partition>.zarr
        data/03_primary/swc_<partition>.zarr
    """

    def __init__(
            self, 
            filepath: str, 
            load_args: Dict[str, Any] | None = None,
            save_args: Dict[str, Any] | None = None,
            credentials: Dict[str, Any] | None = None,
            fs_args: Dict[str, Any] | None = None,
        ):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)

        _fs_args = fs_args or {}
        _credentials = credentials or {}

        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)
        
        self._fs = fsspec.filesystem(protocol, **_credentials, **_fs_args)
        self._load_args = load_args or {"consolidated": True}
        self._save_args = save_args or {"mode": "w"}

    def _describe(self) -> Dict[str, Any]:
        return dict(
            filepath=self._filepath, 
            protocol=self._protocol,
            load_args=self._load_args, 
            save_args=self._save_args
        )

    def _resolve(self, partition_id: str) -> str:
        path = str(self._filepath).replace("<partition>", str(partition_id))
        return self._fs.unstrip_protocol(path)

    def _glob_pattern(self) -> str:
        return re.sub(r"<[^>]+>", "*", str(self._filepath))

    def _load(self) -> xr.Dataset:
        matches = sorted(self._fs.glob(self._glob_pattern()))
        if not matches:
            raise FileNotFoundError(f"No Zarr stores matching {self._glob_pattern()}")
        stores = [self._fs.unstrip_protocol(p) for p in matches]
        return xr.open_mfdataset(stores, engine="zarr", **self._load_args)

    def _save(self, data: Dict[str, xr.Dataset]) -> None:
        for partition_id, ds in data.items():
            store = self._resolve(partition_id)
            ds.to_zarr(store, **self._save_args)

    def _exists(self) -> bool:
        return len(self._fs.glob(self._glob_pattern())) > 0


class DeltaLakeDataset(AbstractDataset):
    """
    Kedro dataset for DeltaLake tables on S3 or local filesystem.

    Saves: pd.DataFrame -> DeltaLake table
    Loads: DeltaLake table -> pd.DataFrame

    Example filepaths:
        s3://my-bucket/tables/my_table
        data/03_primary/my_table
    """

    def __init__(
        self,
        filepath: str,
        load_args: Dict[str, Any] | None = None,
        save_args: Dict[str, Any] | None = None,
        credentials: Dict[str, Any] | None = None,
        fs_args: Dict[str, Any] | None = None,
    ):
        protocol, path = get_protocol_and_path(filepath)
        self._protocol = protocol
        self._filepath = PurePosixPath(path)

        _fs_args = fs_args or {}
        _credentials = credentials or {}

        if protocol == "file":
            _fs_args.setdefault("auto_mkdir", True)

        self._fs = fsspec.filesystem(protocol, **_credentials, **_fs_args)
        self._load_args = load_args or {}
        self._save_args = save_args or {"mode": "append"}

    def _describe(self) -> Dict[str, Any]:
        return {
            "filepath": self._filepath,
            "protocol": self._protocol,
            "load_args": self._load_args,
            "save_args": self._save_args,
        }

    def _full_path(self) -> str:
        return self._fs.unstrip_protocol(str(self._filepath))

    def _load(self) -> pd.DataFrame:
        df = DeltaTable(self._full_path()).to_pandas(**self._load_args)
        #for col in df.select_dtypes(include=["datetime64[ns, UTC]"]).columns:
        #    df[col] = df[col].dt.tz_convert(None)
        return df
    
    #def _prepare_data(self, data: pd.DataFrame) -> pd.DataFrame:
    #    for col in data.select_dtypes(include=["datetime64[ns]"]).columns:
    #        data[col] = data[col].dt.tz_localize("UTC")
    #    return data

    def _save(self, data: pd.DataFrame) -> None:
        write_deltalake(self._full_path(), data, **self._save_args)
    '''
    def _upsert(self, data: pd.DataFrame) -> None:
        dt = DeltaTable(self._full_path())
        merge_condition = " AND ".join(
            [f"source.{k} = target.{k}" for k in self._merge_keys]
        )
        (
            dt.merge(
                source=data,
                predicate=merge_condition,
                source_alias="source",
                target_alias="target",
            )
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute()
        )
    '''

    def _exists(self) -> bool:
        try:
            DeltaTable(self._full_path())
            return True
        except Exception:
            return False