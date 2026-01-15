import numpy as np
from cafe_sim.simulation_state import SimulationState, Customer
from cafe_sim.future_event_list import FutureEventList
from cafe_sim.event_definitions import Event, EventType
from cafe_sim.arrival_process_model import ArrivalProcessModel
from cafe_sim.service_type_model import ServiceTypeModel
from cafe_sim.period_key import PeriodKey
from cafe_sim.configuration import SimulationConfig, KPISet


class CafeSimulationEngine:
    def __init__(
        self,
        period_key: PeriodKey,
        barista_count: int,
        run_length_min: float,
        arrival_model: ArrivalProcessModel,
        service_model: ServiceTypeModel,
        simulation_config: SimulationConfig,
    ):
        self._period_key = period_key
        self._barista_count = barista_count
        self._run_length_min = run_length_min
        self._arrival_model = arrival_model
        self._service_model = service_model
        self._simulation_config = simulation_config
        
        self._state = None
        self._event_list = None
        self._rng = None

    def run(self, rng_seed: int) -> tuple[KPISet, KPISet]:
        self._rng = np.random.default_rng(seed=rng_seed)
        
        self._state = SimulationState(self._barista_count)
        self._event_list = FutureEventList()
        
        self._initialize_state()
        
        if self._simulation_config.warmup_min > 0:
            self._execute_warmup_phase()
        
        full_period_kpis = self._execute_measurement_phase()
        
        post_warmup_kpis = self._compute_post_warmup_kpis()
        
        return full_period_kpis, post_warmup_kpis

    def _initialize_state(self) -> None:
        if self._simulation_config.initial_queue_size > 0:
            for _ in range(self._simulation_config.initial_queue_size):
                customer_id = self._state.allocate_customer_id()
                service_type = self._service_model.sample_service_type(
                    self._period_key, self._rng
                )
                service_duration = self._service_model.sample_service_duration_min(
                    self._period_key, service_type, self._rng
                )
                customer = Customer(
                    customer_id=customer_id,
                    arrival_time_min=0.0,
                    service_type=service_type.value,
                    service_duration_min=service_duration,
                )
                self._state.add_customer(customer)
                self._state.enqueue_customer(customer_id)
        
        if self._simulation_config.initial_busy_baristas > 0:
            for barista_id in range(
                min(self._simulation_config.initial_busy_baristas, self._barista_count)
            ):
                customer_id = self._state.allocate_customer_id()
                service_type = self._service_model.sample_service_type(
                    self._period_key, self._rng
                )
                service_duration = self._service_model.sample_service_duration_min(
                    self._period_key, service_type, self._rng
                )
                customer = Customer(
                    customer_id=customer_id,
                    arrival_time_min=0.0,
                    service_type=service_type.value,
                    service_duration_min=service_duration,
                    service_start_time_min=0.0,
                    service_end_time_min=service_duration,
                    barista_id=barista_id,
                )
                self._state.add_customer(customer)
                self._state.mark_barista_busy(barista_id, service_duration)
                
                end_event = Event(
                    time_min=service_duration,
                    event_type=EventType.SERVICE_END,
                    entity_id=customer_id,
                    data={"barista_id": barista_id},
                )
                self._event_list.push(end_event)
        
        first_arrival_delay = self._arrival_model.sample_next_arrival_delay_min(
            self._period_key, 0.0, self._rng
        )
        first_arrival_event = Event(
            time_min=first_arrival_delay,
            event_type=EventType.ARRIVAL,
            entity_id=self._state.allocate_customer_id(),
        )
        self._event_list.push(first_arrival_event)
        
        end_of_run_event = Event(
            time_min=self._run_length_min,
            event_type=EventType.END_OF_RUN,
            entity_id=-1,
        )
        self._event_list.push(end_of_run_event)

    def _execute_warmup_phase(self) -> None:
        warmup_end_time = self._simulation_config.warmup_min
        self._state.set_warmup_end_time(warmup_end_time)
        
        while not self._event_list.is_empty():
            event = self._event_list.pop_next()
            
            if event.time_min >= warmup_end_time:
                self._event_list.push(event)
                break
            
            self._state.advance_time_to(event.time_min)
            self._process_event(event)
        
        self._state.advance_time_to(warmup_end_time)
        self._state.end_warmup_phase()

    def _execute_measurement_phase(self) -> KPISet:
        while not self._event_list.is_empty():
            event = self._event_list.pop_next()
            
            self._state.advance_time_to(event.time_min)
            
            if event.event_type == EventType.END_OF_RUN:
                break
            
            self._process_event(event)
        
        return self._compute_kpis_from_state(self._run_length_min)

    def _process_event(self, event: Event) -> None:
        if event.event_type == EventType.ARRIVAL:
            self._handle_arrival(event)
        elif event.event_type == EventType.SERVICE_END:
            self._handle_service_end(event)

    def _handle_arrival(self, event: Event) -> None:
        current_time = self._state.get_current_time_min()
        customer_id = event.entity_id
        
        service_type = self._service_model.sample_service_type(
            self._period_key, self._rng
        )
        service_duration = self._service_model.sample_service_duration_min(
            self._period_key, service_type, self._rng
        )
        
        customer = Customer(
            customer_id=customer_id,
            arrival_time_min=current_time,
            service_type=service_type.value,
            service_duration_min=service_duration,
        )
        self._state.add_customer(customer)
        
        if current_time < self._run_length_min:
            next_arrival_delay = self._arrival_model.sample_next_arrival_delay_min(
                self._period_key, current_time, self._rng
            )
            next_arrival_time = current_time + next_arrival_delay
            
            if next_arrival_time < self._run_length_min:
                next_arrival_event = Event(
                    time_min=next_arrival_time,
                    event_type=EventType.ARRIVAL,
                    entity_id=self._state.allocate_customer_id(),
                )
                self._event_list.push(next_arrival_event)
        
        idle_barista_id = self._state.get_idle_barista_id()
        if idle_barista_id >= 0:
            self._start_service(customer_id, idle_barista_id)
        else:
            self._state.enqueue_customer(customer_id)

    def _handle_service_end(self, event: Event) -> None:
        customer_id = event.entity_id
        barista_id = event.data["barista_id"]
        
        customer = self._state.get_customer(customer_id)
        customer.service_end_time_min = self._state.get_current_time_min()
        
        self._state.release_barista(barista_id)
        
        if not self._state.is_queue_empty():
            next_customer_id = self._state.dequeue_customer()
            self._start_service(next_customer_id, barista_id)

    def _start_service(self, customer_id: int, barista_id: int) -> None:
        current_time = self._state.get_current_time_min()
        customer = self._state.get_customer(customer_id)
        
        customer.service_start_time_min = current_time
        customer.barista_id = barista_id
        
        service_end_time = current_time + customer.service_duration_min
        
        self._state.mark_barista_busy(barista_id, service_end_time)
        
        service_end_event = Event(
            time_min=service_end_time,
            event_type=EventType.SERVICE_END,
            entity_id=customer_id,
            data={"barista_id": barista_id},
        )
        self._event_list.push(service_end_event)

    def _compute_kpis_from_state(self, elapsed_time_min: float) -> KPISet:
        completed_customers = self._state.get_all_completed_customers()
        
        if len(completed_customers) == 0:
            return KPISet(
                mean_wait_min=0.0,
                median_wait_min=0.0,
                p90_wait_min=0.0,
                p95_wait_min=0.0,
                mean_queue_length=0.0,
                time_avg_queue_length=0.0,
                utilization_per_barista=0.0,
                p_wait_exceeds_2min=0.0,
                throughput=0,
            )
        
        wait_times = [
            c.service_start_time_min - c.arrival_time_min
            for c in completed_customers
        ]
        wait_times = np.array(wait_times)
        
        mean_wait = float(np.mean(wait_times))
        median_wait = float(np.median(wait_times))
        p90_wait = float(np.percentile(wait_times, 90))
        p95_wait = float(np.percentile(wait_times, 95))
        p_wait_exceeds_2min = float(np.mean(wait_times > 2.0))
        
        time_avg_queue_length = self._state.get_time_avg_queue_length(elapsed_time_min)
        utilization = self._state.get_utilization(elapsed_time_min)
        
        return KPISet(
            mean_wait_min=mean_wait,
            median_wait_min=median_wait,
            p90_wait_min=p90_wait,
            p95_wait_min=p95_wait,
            mean_queue_length=float(np.mean([c.arrival_time_min for c in completed_customers])),
            time_avg_queue_length=time_avg_queue_length,
            utilization_per_barista=utilization,
            p_wait_exceeds_2min=p_wait_exceeds_2min,
            throughput=len(completed_customers),
        )

    def _compute_post_warmup_kpis(self) -> KPISet:
        warmup_time = self._simulation_config.warmup_min
        
        if warmup_time <= 0:
            return self._compute_kpis_from_state(self._run_length_min)
        
        completed_customers = self._state.get_all_completed_customers()
        post_warmup_customers = [
            c for c in completed_customers
            if c.arrival_time_min >= warmup_time
        ]
        
        if len(post_warmup_customers) == 0:
            return KPISet(
                mean_wait_min=0.0,
                median_wait_min=0.0,
                p90_wait_min=0.0,
                p95_wait_min=0.0,
                mean_queue_length=0.0,
                time_avg_queue_length=0.0,
                utilization_per_barista=0.0,
                p_wait_exceeds_2min=0.0,
                throughput=0,
            )
        
        wait_times = [
            c.service_start_time_min - c.arrival_time_min
            for c in post_warmup_customers
        ]
        wait_times = np.array(wait_times)
        
        measurement_duration = self._run_length_min - warmup_time
        
        return KPISet(
            mean_wait_min=float(np.mean(wait_times)),
            median_wait_min=float(np.median(wait_times)),
            p90_wait_min=float(np.percentile(wait_times, 90)),
            p95_wait_min=float(np.percentile(wait_times, 95)),
            mean_queue_length=0.0,
            time_avg_queue_length=self._state.get_time_avg_queue_length(measurement_duration),
            utilization_per_barista=self._state.get_utilization(measurement_duration),
            p_wait_exceeds_2min=float(np.mean(wait_times > 2.0)),
            throughput=len(post_warmup_customers),
        )
