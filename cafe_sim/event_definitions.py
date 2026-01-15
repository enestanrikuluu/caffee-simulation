from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EventType(Enum):
    ARRIVAL = "arrival"
    SERVICE_END = "service_end"
    END_OF_RUN = "end_of_run"


@dataclass(order=True)
class Event:
    time_min: float
    event_type: EventType = field(compare=False)
    entity_id: int = field(compare=False)
    data: Any = field(default=None, compare=False)
    sequence_number: int = field(default=0)

    def __post_init__(self):
        if not hasattr(self, '_seq_counter'):
            Event._seq_counter = 0
        if self.sequence_number == 0:
            Event._seq_counter += 1
            self.sequence_number = Event._seq_counter
