"""LiftZero Learning Environment package.

Exports feature schemas, encoders, decision recorder, expert dataset,
fast twin simulator, and Gymnasium environments.
"""

from elevator_mas.learning.dataset import DatasetStats, ExpertDataset
from elevator_mas.learning.env import (
    ElevatorDecisionEnv,
    RewardConfig,
    VectorDecisionEnv,
)
from elevator_mas.learning.evaluate import Policy, evaluate, summarise
from elevator_mas.learning.features import (
    Encoded,
    EncodedBatch,
    encode_batch,
    encode_decision,
)
from elevator_mas.learning.recorder import (
    DecisionRecorder,
    record_expert_dataset,
)
from elevator_mas.learning.regimes import (
    EvalRegime,
    RegimesConfig,
    load_regimes_config,
)
from elevator_mas.learning.schema import (
    FEATURE_NAMES,
    FEATURE_VERSION,
    KC,
    KCAR,
    KG,
    MAX_CARS,
    SCHEMA_HASH,
)
from elevator_mas.learning.twin.state import TwinSimulator
from elevator_mas.learning.view import (
    DecisionContext,
    FleetView,
    from_board,
    from_event,
)

__all__ = [
    "DatasetStats",
    "DecisionContext",
    "DecisionRecorder",
    "ElevatorDecisionEnv",
    "Encoded",
    "EncodedBatch",
    "EvalRegime",
    "ExpertDataset",
    "FEATURE_NAMES",
    "FEATURE_VERSION",
    "FleetView",
    "KC",
    "KCAR",
    "KG",
    "MAX_CARS",
    "Policy",
    "RegimesConfig",
    "RewardConfig",
    "SCHEMA_HASH",
    "TwinSimulator",
    "VectorDecisionEnv",
    "encode_batch",
    "encode_decision",
    "evaluate",
    "from_board",
    "from_event",
    "load_regimes_config",
    "record_expert_dataset",
    "summarise",
]
