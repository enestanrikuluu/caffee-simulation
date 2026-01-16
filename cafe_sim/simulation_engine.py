import numpy as np
from cafe_sim.simulation_state import (
    create_simulation_state, create_customer, advance_time_to,
    add_customer, get_customer, allocate_customer_id, enqueue_customer,
    dequeue_customer, get_idle_barista_id, mark_barista_busy, release_barista,
    get_current_time_min, is_queue_empty, set_warmup_end_time, end_warmup_phase,
    get_time_avg_queue_length, get_utilization, get_all_completed_customers
)
from cafe_sim.future_event_list import (
    create_event_list, push_event, pop_next_event, is_event_list_empty
)
from cafe_sim.event_definitions import create_event, EVENT_TYPE_ARRIVAL, EVENT_TYPE_SERVICE_END, EVENT_TYPE_END_OF_RUN
from cafe_sim.arrival_process_model import sample_next_arrival_delay
from cafe_sim.service_type_model import sample_service_type, sample_service_duration
from cafe_sim.configuration import create_kpi_set


def create_simulation_engine(
    period_key,
    barista_count,
    run_length_min,
    arrival_model,
    service_model,
    simulation_config
):
    return {
        'period_key': period_key,
        'barista_count': barista_count,
        'run_length_min': run_length_min,
        'arrival_model': arrival_model,
        'service_model': service_model,
        'simulation_config': simulation_config,
        'state': None,
        'event_list': None,
        'rng': None,
    }


def run_simulation(engine, rng_seed):
    engine['rng'] = np.random.default_rng(seed=rng_seed)
    engine['state'] = create_simulation_state(engine['barista_count'])
    engine['event_list'] = create_event_list()

    _initialize_state(engine)

    if engine['simulation_config']['warmup_min'] > 0:
        _execute_warmup_phase(engine)

    full_period_kpis = _execute_measurement_phase(engine)
    post_warmup_kpis = _compute_post_warmup_kpis(engine)

    return full_period_kpis, post_warmup_kpis


def _initialize_state(engine):
    state = engine['state']
    event_list = engine['event_list']
    rng = engine['rng']
    period_key = engine['period_key']
    service_model = engine['service_model']
    arrival_model = engine['arrival_model']
    simulation_config = engine['simulation_config']
    run_length_min = engine['run_length_min']

    if simulation_config['initial_queue_size'] > 0:
        for _ in range(simulation_config['initial_queue_size']):
            customer_id = allocate_customer_id(state)
            svc_type = sample_service_type(service_model, period_key, rng)
            svc_duration = sample_service_duration(service_model, period_key, svc_type, rng)
            customer = create_customer(
                customer_id=customer_id,
                arrival_time_min=0.0,
                service_type=svc_type,
                service_duration_min=svc_duration,
            )
            add_customer(state, customer)
            enqueue_customer(state, customer_id)

    if simulation_config['initial_busy_baristas'] > 0:
        for barista_id in range(
            min(simulation_config['initial_busy_baristas'], engine['barista_count'])
        ):
            customer_id = allocate_customer_id(state)
            svc_type = sample_service_type(service_model, period_key, rng)
            svc_duration = sample_service_duration(service_model, period_key, svc_type, rng)
            customer = create_customer(
                customer_id=customer_id,
                arrival_time_min=0.0,
                service_type=svc_type,
                service_duration_min=svc_duration,
                service_start_time_min=0.0,
                service_end_time_min=svc_duration,
                barista_id=barista_id,
            )
            add_customer(state, customer)
            mark_barista_busy(state, barista_id, svc_duration)

            end_event = create_event(
                time_min=svc_duration,
                event_type=EVENT_TYPE_SERVICE_END,
                entity_id=customer_id,
                data={"barista_id": barista_id},
            )
            push_event(event_list, end_event)

    first_arrival_delay = sample_next_arrival_delay(
        arrival_model, period_key, 0.0, rng
    )
    first_arrival_event = create_event(
        time_min=first_arrival_delay,
        event_type=EVENT_TYPE_ARRIVAL,
        entity_id=allocate_customer_id(state),
    )
    push_event(event_list, first_arrival_event)

    end_of_run_event = create_event(
        time_min=run_length_min,
        event_type=EVENT_TYPE_END_OF_RUN,
        entity_id=-1,
    )
    push_event(event_list, end_of_run_event)


def _execute_warmup_phase(engine):
    state = engine['state']
    event_list = engine['event_list']
    warmup_end_time = engine['simulation_config']['warmup_min']

    set_warmup_end_time(state, warmup_end_time)

    while not is_event_list_empty(event_list):
        event = pop_next_event(event_list)

        if event['time_min'] >= warmup_end_time:
            push_event(event_list, event)
            break

        advance_time_to(state, event['time_min'])
        _process_event(engine, event)

    advance_time_to(state, warmup_end_time)
    end_warmup_phase(state)


def _execute_measurement_phase(engine):
    state = engine['state']
    event_list = engine['event_list']
    run_length_min = engine['run_length_min']

    while not is_event_list_empty(event_list):
        event = pop_next_event(event_list)

        advance_time_to(state, event['time_min'])

        if event['event_type'] == EVENT_TYPE_END_OF_RUN:
            break

        _process_event(engine, event)

    return _compute_kpis_from_state(engine, run_length_min)


def _process_event(engine, event):
    if event['event_type'] == EVENT_TYPE_ARRIVAL:
        _handle_arrival(engine, event)
    elif event['event_type'] == EVENT_TYPE_SERVICE_END:
        _handle_service_end(engine, event)


def _handle_arrival(engine, event):
    state = engine['state']
    event_list = engine['event_list']
    rng = engine['rng']
    period_key = engine['period_key']
    service_model = engine['service_model']
    arrival_model = engine['arrival_model']
    run_length_min = engine['run_length_min']

    current_time = get_current_time_min(state)
    customer_id = event['entity_id']

    svc_type = sample_service_type(service_model, period_key, rng)
    svc_duration = sample_service_duration(service_model, period_key, svc_type, rng)

    customer = create_customer(
        customer_id=customer_id,
        arrival_time_min=current_time,
        service_type=svc_type,
        service_duration_min=svc_duration,
    )
    add_customer(state, customer)

    if current_time < run_length_min:
        next_arrival_delay = sample_next_arrival_delay(
            arrival_model, period_key, current_time, rng
        )
        next_arrival_time = current_time + next_arrival_delay

        if next_arrival_time < run_length_min:
            next_arrival_event = create_event(
                time_min=next_arrival_time,
                event_type=EVENT_TYPE_ARRIVAL,
                entity_id=allocate_customer_id(state),
            )
            push_event(event_list, next_arrival_event)

    idle_barista_id = get_idle_barista_id(state)
    if idle_barista_id >= 0:
        _start_service(engine, customer_id, idle_barista_id)
    else:
        enqueue_customer(state, customer_id)


def _handle_service_end(engine, event):
    state = engine['state']
    customer_id = event['entity_id']
    barista_id = event['data']['barista_id']

    customer = get_customer(state, customer_id)
    customer['service_end_time_min'] = get_current_time_min(state)

    release_barista(state, barista_id)

    if not is_queue_empty(state):
        next_customer_id = dequeue_customer(state)
        _start_service(engine, next_customer_id, barista_id)


def _start_service(engine, customer_id, barista_id):
    state = engine['state']
    event_list = engine['event_list']

    current_time = get_current_time_min(state)
    customer = get_customer(state, customer_id)

    customer['service_start_time_min'] = current_time
    customer['barista_id'] = barista_id

    service_end_time = current_time + customer['service_duration_min']

    mark_barista_busy(state, barista_id, service_end_time)

    service_end_event = create_event(
        time_min=service_end_time,
        event_type=EVENT_TYPE_SERVICE_END,
        entity_id=customer_id,
        data={"barista_id": barista_id},
    )
    push_event(event_list, service_end_event)


def _compute_kpis_from_state(engine, elapsed_time_min):
    state = engine['state']
    completed_customers = get_all_completed_customers(state)

    if len(completed_customers) == 0:
        return create_kpi_set(
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
        c['service_start_time_min'] - c['arrival_time_min']
        for c in completed_customers
    ]
    wait_times = np.array(wait_times)

    mean_wait = float(np.mean(wait_times))
    median_wait = float(np.median(wait_times))
    p90_wait = float(np.percentile(wait_times, 90))
    p95_wait = float(np.percentile(wait_times, 95))
    p_wait_exceeds_2min = float(np.mean(wait_times > 2.0))

    time_avg_queue = get_time_avg_queue_length(state, elapsed_time_min)
    utilization = get_utilization(state, elapsed_time_min)

    return create_kpi_set(
        mean_wait_min=mean_wait,
        median_wait_min=median_wait,
        p90_wait_min=p90_wait,
        p95_wait_min=p95_wait,
        mean_queue_length=float(np.mean([c['arrival_time_min'] for c in completed_customers])),
        time_avg_queue_length=time_avg_queue,
        utilization_per_barista=utilization,
        p_wait_exceeds_2min=p_wait_exceeds_2min,
        throughput=len(completed_customers),
    )


def _compute_post_warmup_kpis(engine):
    state = engine['state']
    warmup_time = engine['simulation_config']['warmup_min']
    run_length_min = engine['run_length_min']

    if warmup_time <= 0:
        return _compute_kpis_from_state(engine, run_length_min)

    completed_customers = get_all_completed_customers(state)
    post_warmup_customers = [
        c for c in completed_customers
        if c['arrival_time_min'] >= warmup_time
    ]

    if len(post_warmup_customers) == 0:
        return create_kpi_set(
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
        c['service_start_time_min'] - c['arrival_time_min']
        for c in post_warmup_customers
    ]
    wait_times = np.array(wait_times)

    measurement_duration = run_length_min - warmup_time

    return create_kpi_set(
        mean_wait_min=float(np.mean(wait_times)),
        median_wait_min=float(np.median(wait_times)),
        p90_wait_min=float(np.percentile(wait_times, 90)),
        p95_wait_min=float(np.percentile(wait_times, 95)),
        mean_queue_length=0.0,
        time_avg_queue_length=get_time_avg_queue_length(state, measurement_duration),
        utilization_per_barista=get_utilization(state, measurement_duration),
        p_wait_exceeds_2min=float(np.mean(wait_times > 2.0)),
        throughput=len(post_warmup_customers),
    )
