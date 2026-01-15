import heapq
from typing import List, Optional
from cafe_sim.event_definitions import Event


class FutureEventList:
    def __init__(self):
        self._heap: List[Event] = []
        self._counter = 0

    def push(self, event: Event) -> None:
        self._counter += 1
        event.sequence_number = self._counter
        heapq.heappush(self._heap, event)

    def pop_next(self) -> Optional[Event]:
        if len(self._heap) == 0:
            return None
        return heapq.heappop(self._heap)

    def is_empty(self) -> bool:
        return len(self._heap) == 0

    def peek_next_time(self) -> Optional[float]:
        if len(self._heap) == 0:
            return None
        return self._heap[0].time_min

    def size(self) -> int:
        return len(self._heap)
