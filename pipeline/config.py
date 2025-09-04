from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PreprocessConfig:
    sfreq: float = 256.0
    l_freq: float = 0.5
    h_freq: float = 40.0
    notch: Optional[float] = 60.0
    ica_n_components: Optional[int] = 10
    ica_max_iter: int = 200
    do_ica: bool = True

@dataclass
class ChannelSelectConfig:
    max_channels: int = 8
    scoring: str = 'roc_auc'
    step: int = 1
    cv_folds: int = 3

@dataclass
class FeatureConfig:
    # windowing already in data (X length); here choose chaos features
    sample_entropy_m: int = 2
    sample_entropy_r: float = 0.2  # as fraction of signal std
    permutation_entropy_order: int = 3
    permutation_entropy_delay: int = 1
    lyapunov_max_iter: int = 100

@dataclass
class ModelConfig:
    random_state: int = 42
    class_weight: Optional[str] = 'balanced'

@dataclass
class PipelineConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    channel_select: ChannelSelectConfig = field(default_factory=ChannelSelectConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    n_jobs: int = -1
    cache_dir: Optional[str] = None
