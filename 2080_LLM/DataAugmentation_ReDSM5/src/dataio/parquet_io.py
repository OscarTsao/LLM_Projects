"""
Parquet I/O utilities for efficient reading and writing of augmented data.

Provides chunked reading/writing, compression, and metadata management.
"""

from typing import Dict, List, Optional, Iterator, Any
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import json


class ParquetIO:
    """
    Utilities for reading and writing Parquet files with compression.
    
    Attributes:
        compression: Compression algorithm (default: "zstd")
        compression_level: Compression level (default: 3)
        row_group_size: Number of rows per row group (default: 10000)
    """
    
    def __init__(
        self,
        compression: str = "zstd",
        compression_level: int = 3,
        row_group_size: int = 10000,
    ):
        """
        Initialize ParquetIO.
        
        Args:
            compression: Compression algorithm ("zstd", "snappy", "gzip", "brotli", None)
            compression_level: Compression level (1-22 for zstd)
            row_group_size: Number of rows per row group
        """
        self.compression = compression
        self.compression_level = compression_level
        self.row_group_size = row_group_size
    
    def write_dataframe(
        self,
        df: pd.DataFrame,
        path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write DataFrame to Parquet with compression and optional metadata.
        
        Args:
            df: DataFrame to write
            path: Output path
            metadata: Optional metadata dictionary to store in file
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to PyArrow Table
        table = pa.Table.from_pandas(df)
        
        # Add metadata if provided
        if metadata:
            metadata_json = json.dumps(metadata)
            existing_metadata = table.schema.metadata or {}
            combined_metadata = {
                **existing_metadata,
                b"user_metadata": metadata_json.encode("utf-8"),
            }
            table = table.replace_schema_metadata(combined_metadata)
        
        # Write with compression
        pq.write_table(
            table,
            path,
            compression=self.compression,
            compression_level=self.compression_level,
            row_group_size=self.row_group_size,
        )
    
    def read_dataframe(
        self,
        path: Path,
        columns: Optional[List[str]] = None,
        filters: Optional[List[tuple]] = None,
    ) -> pd.DataFrame:
        """
        Read DataFrame from Parquet.
        
        Args:
            path: Path to Parquet file
            columns: Optional list of columns to read
            filters: Optional filters (e.g., [("column", "==", value)])
            
        Returns:
            DataFrame
        """
        return pd.read_parquet(
            path,
            columns=columns,
            filters=filters,
        )
    
    def read_metadata(self, path: Path) -> Optional[Dict[str, Any]]:
        """
        Read metadata from Parquet file.
        
        Args:
            path: Path to Parquet file
            
        Returns:
            Metadata dictionary or None
        """
        parquet_file = pq.ParquetFile(path)
        schema_metadata = parquet_file.schema_pandas.metadata
        
        if schema_metadata and b"user_metadata" in schema_metadata:
            metadata_json = schema_metadata[b"user_metadata"].decode("utf-8")
            return json.loads(metadata_json)
        
        return None
    
    def read_chunked(
        self,
        path: Path,
        chunk_size: int = 10000,
        columns: Optional[List[str]] = None,
    ) -> Iterator[pd.DataFrame]:
        """
        Read Parquet file in chunks.
        
        Args:
            path: Path to Parquet file
            chunk_size: Number of rows per chunk
            columns: Optional list of columns to read
            
        Yields:
            DataFrame chunks
        """
        parquet_file = pq.ParquetFile(path)
        
        for batch in parquet_file.iter_batches(batch_size=chunk_size, columns=columns):
            yield batch.to_pandas()
    
    def write_chunked(
        self,
        chunks: Iterator[pd.DataFrame],
        path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Write chunks to a single Parquet file.
        
        Args:
            chunks: Iterator of DataFrame chunks
            path: Output path
            metadata: Optional metadata dictionary
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        writer = None
        schema = None
        
        try:
            for chunk in chunks:
                table = pa.Table.from_pandas(chunk)
                
                if writer is None:
                    # Initialize writer with first chunk's schema
                    schema = table.schema
                    
                    # Add metadata if provided
                    if metadata:
                        metadata_json = json.dumps(metadata)
                        combined_metadata = {
                            b"user_metadata": metadata_json.encode("utf-8"),
                        }
                        schema = schema.with_metadata(combined_metadata)
                    
                    writer = pq.ParquetWriter(
                        path,
                        schema,
                        compression=self.compression,
                        compression_level=self.compression_level,
                    )
                
                writer.write_table(table)
        
        finally:
            if writer:
                writer.close()
    
    def get_file_info(self, path: Path) -> Dict[str, Any]:
        """
        Get information about a Parquet file.
        
        Args:
            path: Path to Parquet file
            
        Returns:
            Dictionary with file information
        """
        parquet_file = pq.ParquetFile(path)
        
        return {
            "num_rows": parquet_file.metadata.num_rows,
            "num_columns": parquet_file.metadata.num_columns,
            "num_row_groups": parquet_file.metadata.num_row_groups,
            "serialized_size": parquet_file.metadata.serialized_size,
            "schema": str(parquet_file.schema),
            "metadata": self.read_metadata(path),
        }
    
    def merge_parquet_files(
        self,
        input_paths: List[Path],
        output_path: Path,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Merge multiple Parquet files into one.
        
        Args:
            input_paths: List of input Parquet file paths
            output_path: Output path
            metadata: Optional metadata for merged file
        """
        # Read all files as chunks and write to output
        def chunk_generator():
            for path in input_paths:
                df = self.read_dataframe(path)
                yield df
        
        self.write_chunked(chunk_generator(), output_path, metadata)
