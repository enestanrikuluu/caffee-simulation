from collections import deque
from dataclasses import dataclass
import numpy as np


@dataclass
class Customer:
    customer_id: int
    arrival_time_min: float
    service_type: str
    service_duration_min: float
    service_start_time_min: float = -1.0
    service_end_time_min: float = -1.0
    barista_id: int = -1


class SimulationState:
    def __init__(self, barista_count: int):
        self._barista_count = barista_count
        self._current_time_min = 0.0
        self._last_event_time_min = 0.0
        
        self._waiting_queue = deque()
        self._barista_next_free_times_min = np.zeros(barista_count)
        
        self._customers = {}
        self._next_customer_id = 1
        
        self._time_weighted_queue_length_area = 0.0
        self._time_weighted_in_system_area = 0.0
        self._time_weighted_busy_servers_area = 0.0
        
        self._customers_in_queue = 0
        self._customers_in_service = 0
        
        self._warmup_end_time_min = 0.0
        self._warmup_phase_active = True

    def advance_time_to(self, new_time_min: float) -> None:
        if new_time_min < self._current_time_min:
            raise ValueError("Cannot move time backwards")
        
        self._update_time_weighted_areas(new_time_min)
        self._current_time_min = new_time_min

    def _update_time_weighted_areas(self, new_time_min: float) -> None:
        time_delta = new_time_min - self._last_event_time_min
        
        if time_delta > 0:
            self._time_weighted_queue_length_area += self._customers_in_queue * time_delta
            total_in_system = self._customers_in_queue + self._customers_in_service
            self._time_weighted_in_system_area += total_in_system * time_delta
            self._time_weighted_busy_servers_area += self._customers_in_service * time_delta
        
        self._last_event_time_min = new_time_min

    def enqueue_customer(self, customer_id: int) -> None:
        self._waiting_queue.append(customer_id)
        self._customers_in_queue += 1

    def dequeue_customer(self) -> int:
        if len(self._waiting_queue) == 0:
            raise ValueError("Cannot dequeue from empty queue")
        customer_id = self._waiting_queue.popleft()
        self._customers_in_queue -= 1
        return customer_id

    def get_idle_barista_id(self) -> int:
        for barista_id in range(self._barista_count):
            if self._barista_next_free_times_min[barista_id] <= self._current_time_min:
                return barista_id
        return -1

    def mark_barista_busy(self, barista_id: int, until_time_min: float) -> None:
        self._barista_next_free_times_min[barista_id] = until_time_min
        self._customers_in_service += 1

    def release_barista(self, barista_id: int) -> None:
        self._barista_next_free_times_min[barista_id] = self._current_time_min
        self._customers_in_service -= 1

    def add_customer(self, customer: Customer) -> None:
        self._customers[customer.customer_id] = customer

    def get_customer(self, customer_id: int) -> Customer:
        return self._customers[customer_id]

    def allocate_customer_id(self) -> int:
        customer_id = self._next_customer_id
        self._next_customer_id += 1
        return customer_id

    def get_current_time_min(self) -> float:
        return self._current_time_min

    def get_queue_length(self) -> int:
        return self._customers_in_queue

    def is_queue_empty(self) -> bool:
        return self._customers_in_queue == 0

    def get_barista_count(self) -> int:
        return self._barista_count

    def get_time_avg_queue_length(self, until_time_min: float) -> float:
        if until_time_min <= 0:
            return 0.0
        return self._time_weighted_queue_length_area / until_time_min

    def get_time_avg_system_count(self, until_time_min: float) -> float:
        if until_time_min <= 0:
            return 0.0
        return self._time_weighted_in_system_area / until_time_min

    def get_utilization(self, until_time_min: float) -> float:
        if until_time_min <= 0 or self._barista_count == 0:
            return 0.0
        return self._time_weighted_busy_servers_area / (until_time_min * self._barista_count)

    def set_warmup_end_time(self, time_min: float) -> None:
        self._warmup_end_time_min = time_min

    def end_warmup_phase(self) -> None:
        self._warmup_phase_active = False
        self._time_weighted_queue_length_area = 0.0
        self._time_weighted_in_system_area = 0.0
        self._time_weighted_busy_servers_area = 0.0
        self._last_event_time_min = self._current_time_min

    def is_warmup_active(self) -> bool:
        return self._warmup_phase_active

    def get_all_completed_customers(self):
        return [
            c for c in self._customers.values()
            if c.service_end_time_min > 0
        ]
