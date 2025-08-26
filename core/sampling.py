"""
Deterministic Sampling Policy System for EnMapper

This module provides advanced sampling strategies for creating representative
Sample Packs while maintaining determinism and stratification.
"""

import random
import hashlib
from typing import Dict, List, Optional, Any, Tuple, Union
from enum import Enum
from dataclasses import dataclass

import polars as pl
import numpy as np


class SamplingStrategy(str, Enum):
    """Available sampling strategies."""
    SIMPLE_RANDOM = "simple_random"
    STRATIFIED = "stratified"
    SYSTEMATIC = "systematic"
    CLUSTER = "cluster"
    RESERVOIR = "reservoir"


class StratificationMethod(str, Enum):
    """Methods for stratifying data."""
    BY_COLUMN_TYPE = "by_column_type"
    BY_VALUE_DISTRIBUTION = "by_value_distribution"
    BY_NULL_PATTERN = "by_null_pattern"
    BY_CUSTOM_GROUPS = "by_custom_groups"
    AUTOMATIC = "automatic"


@dataclass
class SamplingConfig:
    """Configuration for sampling operations."""
    strategy: SamplingStrategy
    sample_size: int
    seed: int = 42
    stratification_method: Optional[StratificationMethod] = None
    stratification_columns: Optional[List[str]] = None
    min_stratum_size: int = 1
    max_stratum_size: Optional[int] = None
    preserve_proportions: bool = True
    
    # Advanced options
    replacement: bool = False
    weight_column: Optional[str] = None
    balance_nulls: bool = True


@dataclass
class StratumInfo:
    """Information about a single stratum."""
    stratum_id: str
    description: str
    population_size: int
    sample_size: int
    selection_probability: float
    rows_indices: List[int]
    metadata: Dict[str, Any]


@dataclass
class SamplingResult:
    """Result of a sampling operation."""
    sampled_df: pl.DataFrame
    original_size: int
    sample_size: int
    sampling_config: SamplingConfig
    strata_info: List[StratumInfo]
    selection_probabilities: Dict[int, float]  # row_index -> probability
    metadata: Dict[str, Any]


class DeterministicSampler:
    """Deterministic sampling engine with reproducible results."""
    
    def __init__(self, seed: int = 42):
        self.base_seed = seed
    
    def _get_deterministic_seed(self, data_signature: str, operation_id: str) -> int:
        """Generate deterministic seed based on data signature and operation."""
        combined = f"{data_signature}_{operation_id}_{self.base_seed}"
        hash_digest = hashlib.sha256(combined.encode()).hexdigest()
        return int(hash_digest[:8], 16) % (2**31)
    
    def _calculate_data_signature(self, df: pl.DataFrame) -> str:
        """Calculate a signature for the dataframe to ensure determinism."""
        # Use column names, types, and row count as signature
        col_info = [(col, str(df[col].dtype)) for col in df.columns]
        signature_data = f"{col_info}_{len(df)}"
        return hashlib.sha256(signature_data.encode()).hexdigest()[:16]
    
    def simple_random_sample(self, df: pl.DataFrame, config: SamplingConfig) -> SamplingResult:
        """Perform simple random sampling."""
        data_sig = self._calculate_data_signature(df)
        seed = self._get_deterministic_seed(data_sig, "simple_random")
        
        sample_size = min(config.sample_size, len(df))
        
        # Generate deterministic random indices
        random.seed(seed)
        indices = random.sample(range(len(df)), sample_size)
        indices.sort()  # Sort for determinism
        
        sampled_df = df[indices]
        
        # Calculate selection probabilities
        selection_prob = sample_size / len(df)
        selection_probabilities = {idx: selection_prob for idx in indices}
        
        # Create stratum info (single stratum for simple random)
        stratum = StratumInfo(
            stratum_id="all_data",
            description="Simple random sample from entire dataset",
            population_size=len(df),
            sample_size=sample_size,
            selection_probability=selection_prob,
            rows_indices=indices,
            metadata={"sampling_method": "simple_random"}
        )
        
        return SamplingResult(
            sampled_df=sampled_df,
            original_size=len(df),
            sample_size=sample_size,
            sampling_config=config,
            strata_info=[stratum],
            selection_probabilities=selection_probabilities,
            metadata={
                "data_signature": data_sig,
                "seed_used": seed,
                "selection_probability": selection_prob
            }
        )
    
    def stratified_sample(self, df: pl.DataFrame, config: SamplingConfig) -> SamplingResult:
        """Perform stratified sampling."""
        data_sig = self._calculate_data_signature(df)
        seed = self._get_deterministic_seed(data_sig, "stratified")
        
        # Determine stratification approach
        if config.stratification_method == StratificationMethod.AUTOMATIC:
            strata_df = self._auto_stratify(df)
        elif config.stratification_method == StratificationMethod.BY_COLUMN_TYPE:
            strata_df = self._stratify_by_column_types(df)
        elif config.stratification_method == StratificationMethod.BY_VALUE_DISTRIBUTION:
            strata_df = self._stratify_by_value_distribution(df, config.stratification_columns)
        elif config.stratification_method == StratificationMethod.BY_NULL_PATTERN:
            strata_df = self._stratify_by_null_pattern(df)
        else:
            # Fallback to simple random
            return self.simple_random_sample(df, config)
        
        # Perform stratified sampling
        strata_info = []
        all_sampled_indices = []
        selection_probabilities = {}
        
        random.seed(seed)
        
        for stratum_id in strata_df["stratum_id"].unique():
            stratum_rows = strata_df.filter(pl.col("stratum_id") == stratum_id)
            stratum_indices = stratum_rows["original_index"].to_list()
            
            population_size = len(stratum_indices)
            
            # Calculate sample size for this stratum
            if config.preserve_proportions:
                stratum_sample_size = max(
                    config.min_stratum_size,
                    int((population_size / len(df)) * config.sample_size)
                )
            else:
                stratum_sample_size = min(
                    config.sample_size // len(strata_df["stratum_id"].unique()),
                    population_size
                )
            
            # Apply max stratum size limit
            if config.max_stratum_size:
                stratum_sample_size = min(stratum_sample_size, config.max_stratum_size)
            
            stratum_sample_size = min(stratum_sample_size, population_size)
            
            # Sample from this stratum
            if stratum_sample_size > 0:
                sampled_indices = random.sample(stratum_indices, stratum_sample_size)
                all_sampled_indices.extend(sampled_indices)
                
                # Calculate selection probability for this stratum
                selection_prob = stratum_sample_size / population_size
                for idx in sampled_indices:
                    selection_probabilities[idx] = selection_prob
                
                # Create stratum info
                stratum_info = StratumInfo(
                    stratum_id=str(stratum_id),
                    description=f"Stratum {stratum_id}",
                    population_size=population_size,
                    sample_size=stratum_sample_size,
                    selection_probability=selection_prob,
                    rows_indices=sampled_indices,
                    metadata={"stratification_method": config.stratification_method}
                )
                strata_info.append(stratum_info)
        
        # Create final sample
        all_sampled_indices.sort()  # Sort for determinism
        sampled_df = df[all_sampled_indices]
        
        return SamplingResult(
            sampled_df=sampled_df,
            original_size=len(df),
            sample_size=len(all_sampled_indices),
            sampling_config=config,
            strata_info=strata_info,
            selection_probabilities=selection_probabilities,
            metadata={
                "data_signature": data_sig,
                "seed_used": seed,
                "num_strata": len(strata_info),
                "stratification_method": config.stratification_method
            }
        )
    
    def _auto_stratify(self, df: pl.DataFrame) -> pl.DataFrame:
        """Automatically determine stratification based on data characteristics."""
        # Start with a copy that includes original index
        stratified = df.with_row_count("original_index")
        
        # Simple auto-stratification: by data types and null patterns
        strata = []
        
        for i, row in enumerate(stratified.iter_rows(named=True)):
            # Create stratum ID based on null pattern and dominant data types
            null_pattern = sum(1 for v in row.values() if v is None)
            
            # Categorize by null density
            if null_pattern == 0:
                stratum_id = "complete_rows"
            elif null_pattern < len(df.columns) * 0.3:
                stratum_id = "mostly_complete"
            elif null_pattern < len(df.columns) * 0.7:
                stratum_id = "mostly_null"
            else:
                stratum_id = "sparse_rows"
            
            strata.append(stratum_id)
        
        return stratified.with_columns(pl.Series("stratum_id", strata))
    
    def _stratify_by_column_types(self, df: pl.DataFrame) -> pl.DataFrame:
        """Stratify based on column data types."""
        stratified = df.with_row_count("original_index")
        
        # Count columns by type for each row
        strata = []
        for i in range(len(df)):
            numeric_cols = sum(1 for col in df.columns if df[col].dtype.is_numeric())
            string_cols = sum(1 for col in df.columns if df[col].dtype == pl.Utf8)
            
            if numeric_cols > string_cols:
                stratum_id = "numeric_heavy"
            elif string_cols > numeric_cols:
                stratum_id = "text_heavy"
            else:
                stratum_id = "mixed_types"
            
            strata.append(stratum_id)
        
        return stratified.with_columns(pl.Series("stratum_id", strata))
    
    def _stratify_by_value_distribution(self, df: pl.DataFrame, columns: Optional[List[str]]) -> pl.DataFrame:
        """Stratify based on value distributions in specified columns."""
        if not columns:
            return self._auto_stratify(df)
        
        stratified = df.with_row_count("original_index")
        
        # Use first stratification column for simplicity
        strat_col = columns[0]
        if strat_col not in df.columns:
            return self._auto_stratify(df)
        
        # Create quantile-based strata for numeric columns
        if df[strat_col].dtype.is_numeric():
            # Use quantiles to create strata
            non_null_values = df[strat_col].drop_nulls()
            if len(non_null_values) > 0:
                quantiles = non_null_values.quantile([0.25, 0.5, 0.75])
                q25, q50, q75 = quantiles.to_list()
                
                strata = []
                for value in df[strat_col]:
                    if value is None:
                        stratum_id = "null_values"
                    elif value <= q25:
                        stratum_id = "q1_low"
                    elif value <= q50:
                        stratum_id = "q2_medium_low"
                    elif value <= q75:
                        stratum_id = "q3_medium_high"
                    else:
                        stratum_id = "q4_high"
                    strata.append(stratum_id)
            else:
                strata = ["all_null"] * len(df)
        else:
            # For categorical columns, use value frequency
            value_counts = df[strat_col].value_counts()
            common_values = set(value_counts.head(5)[strat_col].to_list())
            
            strata = []
            for value in df[strat_col]:
                if value is None:
                    stratum_id = "null_values"
                elif value in common_values:
                    stratum_id = f"common_{str(value)[:10]}"
                else:
                    stratum_id = "rare_values"
                strata.append(stratum_id)
        
        return stratified.with_columns(pl.Series("stratum_id", strata))
    
    def _stratify_by_null_pattern(self, df: pl.DataFrame) -> pl.DataFrame:
        """Stratify based on null value patterns."""
        stratified = df.with_row_count("original_index")
        
        strata = []
        for i, row in enumerate(stratified.iter_rows(named=True)):
            null_count = sum(1 for v in row.values() if v is None)
            total_cols = len(df.columns)
            
            if null_count == 0:
                stratum_id = "no_nulls"
            elif null_count < total_cols * 0.25:
                stratum_id = "few_nulls"
            elif null_count < total_cols * 0.5:
                stratum_id = "some_nulls"
            elif null_count < total_cols * 0.75:
                stratum_id = "many_nulls"
            else:
                stratum_id = "mostly_nulls"
            
            strata.append(stratum_id)
        
        return stratified.with_columns(pl.Series("stratum_id", strata))
    
    def systematic_sample(self, df: pl.DataFrame, config: SamplingConfig) -> SamplingResult:
        """Perform systematic sampling with deterministic intervals."""
        data_sig = self._calculate_data_signature(df)
        seed = self._get_deterministic_seed(data_sig, "systematic")
        
        sample_size = min(config.sample_size, len(df))
        population_size = len(df)
        
        # Calculate sampling interval
        interval = population_size / sample_size
        
        # Deterministic random start point
        random.seed(seed)
        start = random.uniform(0, interval)
        
        # Generate systematic indices
        indices = []
        current = start
        while len(indices) < sample_size and current < population_size:
            indices.append(int(current))
            current += interval
        
        # Ensure we don't exceed bounds and have exact sample size
        indices = [min(idx, population_size - 1) for idx in indices]
        indices = list(dict.fromkeys(indices))  # Remove duplicates while preserving order
        indices = indices[:sample_size]  # Ensure exact sample size
        
        sampled_df = df[indices]
        
        # Calculate selection probabilities
        selection_prob = sample_size / population_size
        selection_probabilities = {idx: selection_prob for idx in indices}
        
        # Create stratum info
        stratum = StratumInfo(
            stratum_id="systematic_sample",
            description=f"Systematic sample with interval {interval:.2f}",
            population_size=population_size,
            sample_size=len(indices),
            selection_probability=selection_prob,
            rows_indices=indices,
            metadata={
                "sampling_method": "systematic",
                "interval": interval,
                "start_point": start
            }
        )
        
        return SamplingResult(
            sampled_df=sampled_df,
            original_size=population_size,
            sample_size=len(indices),
            sampling_config=config,
            strata_info=[stratum],
            selection_probabilities=selection_probabilities,
            metadata={
                "data_signature": data_sig,
                "seed_used": seed,
                "interval": interval,
                "start_point": start,
                "selection_probability": selection_prob
            }
        )
    
    def cluster_sample(self, df: pl.DataFrame, config: SamplingConfig) -> SamplingResult:
        """Perform cluster sampling by grouping rows into clusters."""
        data_sig = self._calculate_data_signature(df)
        seed = self._get_deterministic_seed(data_sig, "cluster")
        
        # Determine cluster size (aim for 5-20 clusters)
        target_clusters = min(20, max(5, len(df) // 50))
        cluster_size = max(1, len(df) // target_clusters)
        
        # Create clusters by row position
        clusters = {}
        for i in range(len(df)):
            cluster_id = i // cluster_size
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(i)
        
        # Calculate how many clusters to sample
        total_clusters = len(clusters)
        clusters_needed = min(total_clusters, max(1, config.sample_size // cluster_size))
        
        # Randomly select clusters
        random.seed(seed)
        selected_cluster_ids = random.sample(list(clusters.keys()), clusters_needed)
        
        # Collect all indices from selected clusters
        all_indices = []
        strata_info = []
        selection_probabilities = {}
        
        cluster_selection_prob = clusters_needed / total_clusters
        
        for cluster_id in selected_cluster_ids:
            cluster_indices = clusters[cluster_id]
            
            # Limit cluster contribution to sample size
            max_from_cluster = min(len(cluster_indices), config.sample_size - len(all_indices))
            if max_from_cluster <= 0:
                break
            
            # Sample from cluster if it's too large
            if len(cluster_indices) > max_from_cluster:
                cluster_indices = random.sample(cluster_indices, max_from_cluster)
            
            all_indices.extend(cluster_indices)
            
            # Calculate selection probability (cluster prob * within-cluster prob)
            within_cluster_prob = len(cluster_indices) / len(clusters[cluster_id])
            final_prob = cluster_selection_prob * within_cluster_prob
            
            for idx in cluster_indices:
                selection_probabilities[idx] = final_prob
            
            # Create stratum info for this cluster
            stratum = StratumInfo(
                stratum_id=f"cluster_{cluster_id}",
                description=f"Cluster {cluster_id} (rows {min(clusters[cluster_id])}-{max(clusters[cluster_id])})",
                population_size=len(clusters[cluster_id]),
                sample_size=len(cluster_indices),
                selection_probability=final_prob,
                rows_indices=cluster_indices,
                metadata={
                    "cluster_id": cluster_id,
                    "cluster_size": cluster_size,
                    "original_cluster_size": len(clusters[cluster_id])
                }
            )
            strata_info.append(stratum)
        
        # Sort indices for determinism
        all_indices.sort()
        sampled_df = df[all_indices]
        
        return SamplingResult(
            sampled_df=sampled_df,
            original_size=len(df),
            sample_size=len(all_indices),
            sampling_config=config,
            strata_info=strata_info,
            selection_probabilities=selection_probabilities,
            metadata={
                "data_signature": data_sig,
                "seed_used": seed,
                "total_clusters": total_clusters,
                "clusters_selected": clusters_needed,
                "cluster_size": cluster_size,
                "cluster_selection_probability": cluster_selection_prob
            }
        )
    
    def reservoir_sample(self, df: pl.DataFrame, config: SamplingConfig) -> SamplingResult:
        """Perform reservoir sampling (useful for streaming data simulation)."""
        data_sig = self._calculate_data_signature(df)
        seed = self._get_deterministic_seed(data_sig, "reservoir")
        
        sample_size = min(config.sample_size, len(df))
        
        # Initialize reservoir with first k elements
        random.seed(seed)
        reservoir_indices = list(range(min(sample_size, len(df))))
        
        # Process remaining elements using reservoir algorithm
        for i in range(sample_size, len(df)):
            # Generate random index between 0 and i (inclusive)
            j = random.randint(0, i)
            
            # If j is within reservoir size, replace element at j
            if j < sample_size:
                reservoir_indices[j] = i
        
        # Sort indices for determinism
        reservoir_indices.sort()
        sampled_df = df[reservoir_indices]
        
        # All elements have equal probability in reservoir sampling
        selection_prob = sample_size / len(df)
        selection_probabilities = {idx: selection_prob for idx in reservoir_indices}
        
        # Create stratum info
        stratum = StratumInfo(
            stratum_id="reservoir_sample",
            description="Reservoir sample with equal probability for all elements",
            population_size=len(df),
            sample_size=len(reservoir_indices),
            selection_probability=selection_prob,
            rows_indices=reservoir_indices,
            metadata={
                "sampling_method": "reservoir",
                "final_reservoir_size": len(reservoir_indices)
            }
        )
        
        return SamplingResult(
            sampled_df=sampled_df,
            original_size=len(df),
            sample_size=len(reservoir_indices),
            sampling_config=config,
            strata_info=[stratum],
            selection_probabilities=selection_probabilities,
            metadata={
                "data_signature": data_sig,
                "seed_used": seed,
                "selection_probability": selection_prob,
                "algorithm": "reservoir_sampling"
            }
        )
    
    def sample(self, df: pl.DataFrame, config: SamplingConfig) -> SamplingResult:
        """Main sampling method that delegates to appropriate strategy."""
        if config.strategy == SamplingStrategy.SIMPLE_RANDOM:
            return self.simple_random_sample(df, config)
        elif config.strategy == SamplingStrategy.STRATIFIED:
            return self.stratified_sample(df, config)
        elif config.strategy == SamplingStrategy.SYSTEMATIC:
            return self.systematic_sample(df, config)
        elif config.strategy == SamplingStrategy.CLUSTER:
            return self.cluster_sample(df, config)
        elif config.strategy == SamplingStrategy.RESERVOIR:
            return self.reservoir_sample(df, config)
        else:
            # Default to simple random for unsupported strategies
            return self.simple_random_sample(df, config)


class SamplingPolicy:
    """High-level sampling policy manager."""
    
    def __init__(self):
        self.sampler = DeterministicSampler()
        self.default_configs = self._initialize_default_configs()
    
    def _initialize_default_configs(self) -> Dict[str, SamplingConfig]:
        """Initialize default sampling configurations."""
        return {
            "small_dataset": SamplingConfig(
                strategy=SamplingStrategy.SIMPLE_RANDOM,
                sample_size=50,
                seed=42
            ),
            "medium_dataset": SamplingConfig(
                strategy=SamplingStrategy.STRATIFIED,
                sample_size=100,
                seed=42,
                stratification_method=StratificationMethod.AUTOMATIC,
                min_stratum_size=3,
                preserve_proportions=True
            ),
            "large_dataset": SamplingConfig(
                strategy=SamplingStrategy.STRATIFIED,
                sample_size=200,
                seed=42,
                stratification_method=StratificationMethod.BY_VALUE_DISTRIBUTION,
                min_stratum_size=5,
                max_stratum_size=50,
                preserve_proportions=True
            ),
            "pii_sensitive": SamplingConfig(
                strategy=SamplingStrategy.STRATIFIED,
                sample_size=75,
                seed=42,
                stratification_method=StratificationMethod.BY_NULL_PATTERN,
                min_stratum_size=2,
                preserve_proportions=True,
                balance_nulls=True
            ),
            "systematic_ordered": SamplingConfig(
                strategy=SamplingStrategy.SYSTEMATIC,
                sample_size=100,
                seed=42
            ),
            "cluster_analysis": SamplingConfig(
                strategy=SamplingStrategy.CLUSTER,
                sample_size=150,
                seed=42
            ),
            "streaming_data": SamplingConfig(
                strategy=SamplingStrategy.RESERVOIR,
                sample_size=80,
                seed=42
            ),
            "time_series": SamplingConfig(
                strategy=SamplingStrategy.SYSTEMATIC,
                sample_size=120,
                seed=42
            ),
            "geographic_clusters": SamplingConfig(
                strategy=SamplingStrategy.CLUSTER,
                sample_size=100,
                seed=42
            )
        }
    
    def determine_optimal_config(self, df: pl.DataFrame, context: Dict[str, Any] = None) -> SamplingConfig:
        """Determine optimal sampling configuration based on data characteristics."""
        context = context or {}
        
        dataset_size = len(df)
        num_columns = len(df.columns)
        
        # Analyze data characteristics
        null_density = self._calculate_null_density(df)
        type_diversity = self._calculate_type_diversity(df)
        
        # Determine appropriate configuration
        if dataset_size < 500:
            base_config = self.default_configs["small_dataset"]
        elif dataset_size < 10000:
            base_config = self.default_configs["medium_dataset"]
        else:
            base_config = self.default_configs["large_dataset"]
        
        # Adjust for PII sensitivity
        if context.get("contains_pii", False) or context.get("privacy_mode", False):
            base_config = self.default_configs["pii_sensitive"]
        
        # Adjust sample size based on dataset characteristics
        adjusted_sample_size = min(
            base_config.sample_size,
            max(50, int(dataset_size * 0.1))  # At most 10% of data, at least 50 rows
        )
        
        # Create customized config
        custom_config = SamplingConfig(
            strategy=base_config.strategy,
            sample_size=adjusted_sample_size,
            seed=base_config.seed,
            stratification_method=base_config.stratification_method,
            min_stratum_size=max(1, adjusted_sample_size // 20),
            preserve_proportions=base_config.preserve_proportions,
            balance_nulls=null_density > 0.3
        )
        
        return custom_config
    
    def _calculate_null_density(self, df: pl.DataFrame) -> float:
        """Calculate overall null density in the dataframe."""
        total_cells = len(df) * len(df.columns)
        null_cells = sum(df[col].null_count() for col in df.columns)
        return null_cells / total_cells if total_cells > 0 else 0.0
    
    def _calculate_type_diversity(self, df: pl.DataFrame) -> float:
        """Calculate diversity of data types."""
        unique_types = set(str(df[col].dtype) for col in df.columns)
        return len(unique_types) / len(df.columns) if len(df.columns) > 0 else 0.0
    
    def create_sample_pack(
        self, 
        df: pl.DataFrame, 
        run_id: str,
        config: Optional[SamplingConfig] = None,
        context: Dict[str, Any] = None
    ) -> SamplingResult:
        """Create a sample pack with appropriate sampling strategy."""
        if config is None:
            config = self.determine_optimal_config(df, context)
        
        return self.sampler.sample(df, config)


# Global sampling policy instance
sampling_policy = SamplingPolicy()
