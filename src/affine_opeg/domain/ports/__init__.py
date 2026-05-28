"""Protocol-based ports.

Importing a Protocol from here is the only legal way for ``application`` code
to refer to external systems (DB, teacher APIs, sandbox, blob store, GPU
engine). Concrete adapters implement these protocols in
``affine_opeg.adapters``.
"""

from affine_opeg.domain.ports.blob_store import BlobStore
from affine_opeg.domain.ports.env import Env, Evaluator, PatchExtractor, SandboxExec
from affine_opeg.domain.ports.forward_engine import ForwardEngine
from affine_opeg.domain.ports.metadata_store import (
    MetadataStore,
    PairRepository,
    PairSetRepository,
    RolloutRepository,
    SamplingListRepository,
    StudentRepository,
    StudentScoreRepository,
    TaskRepository,
    TeacherRepository,
)
from affine_opeg.domain.ports.normalizer import TrajectoryNormalizer
from affine_opeg.domain.ports.pair_strategy import PairStrategy
from affine_opeg.domain.ports.sandbox import Sandbox, SandboxFactory
from affine_opeg.domain.ports.scoring import RolloutScoringStrategy, ScoringStrategy
from affine_opeg.domain.ports.task_source import TaskSource
from affine_opeg.domain.ports.teacher import TeacherProvider, TeacherRegistry

__all__ = [
    "BlobStore",
    "Env",
    "Evaluator",
    "ForwardEngine",
    "PatchExtractor",
    "RolloutScoringStrategy",
    "SandboxExec",
    "MetadataStore",
    "PairRepository",
    "PairSetRepository",
    "PairStrategy",
    "RolloutRepository",
    "Sandbox",
    "SandboxFactory",
    "SamplingListRepository",
    "ScoringStrategy",
    "StudentRepository",
    "StudentScoreRepository",
    "TaskRepository",
    "TaskSource",
    "TeacherProvider",
    "TeacherRegistry",
    "TeacherRepository",
    "TrajectoryNormalizer",
]
