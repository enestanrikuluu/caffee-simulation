from collections import deque
import numpy as np


def create_customer(
    customer_id,
    arrival_time_min,
    service_type,
    service_duration_min,
    service_start_time_min=-1.0,
    service_end_time_min=-1.0,
    barista_id=-1
):
    return {
        'customer_id': customer_id,
        'arrival_time_min': arrival_time_min,
        'service_type': service_type,
        'service_duration_min': service_duration_min,
        'service_start_time_min': service_start_time_min,
        'service_end_time_min': service_end_time_min,
        'barista_id': barista_id,
    }


def create_simulation_state(barista_count):
    return {
        'barista_count': barista_count,
        'current_time_min': 0.0,
        'last_event_time_min': 0.0,
        'waiting_queue': deque(),
        'barista_next_free_times_min': np.zeros(barista_count),
        'customers': {},
        'next_customer_id': 1,
        'time_weighted_queue_length_area': 0.0,
        'time_weighted_in_system_area': 0.0,
        'time_weighted_busy_servers_area': 0.0,
        'customers_in_queue': 0,
        'customers_in_service': 0,
        'warmup_end_time_min': 0.0,
        'warmup_phase_active': True,
    }


def advance_time_to(state, new_time_min):
    if new_time_min < state['current_time_min']:
        raise ValueError("Cannot move time backwards")

    _update_time_weighted_areas(state, new_time_min)
    state['current_time_min'] = new_time_min


def _update_time_weighted_areas(state, new_time_min):
    time_delta = new_time_min - state['last_event_time_min']

    if time_delta > 0:
        state['time_weighted_queue_length_area'] += state['customers_in_queue'] * time_delta
        total_in_system = state['customers_in_queue'] + state['customers_in_service']
        state['time_weighted_in_system_area'] += total_in_system * time_delta
        state['time_weighted_busy_servers_area'] += state['customers_in_service'] * time_delta

    state['last_event_time_min'] = new_time_min


def enqueue_customer(state, customer_id):
    state['waiting_queue'].append(customer_id)
    state['customers_in_queue'] += 1


def dequeue_customer(state):
    if len(state['waiting_queue']) == 0:
        raise ValueError("Cannot dequeue from empty queue")
    customer_id = state['waiting_queue'].popleft()
    state['customers_in_queue'] -= 1
    return customer_id


def get_idle_barista_id(state):
    for barista_id in range(state['barista_count']):
        if state['barista_next_free_times_min'][barista_id] <= state['current_time_min']:
            return barista_id
    return -1


def mark_barista_busy(state, barista_id, until_time_min):
    state['barista_next_free_times_min'][barista_id] = until_time_min
    state['customers_in_service'] += 1


def release_barista(state, barista_id):
    state['barista_next_free_times_min'][barista_id] = state['current_time_min']
    state['customers_in_service'] -= 1


def add_customer(state, customer):
    state['customers'][customer['customer_id']] = customer


def get_customer(state, customer_id):
    return state['customers'][customer_id]


def allocate_customer_id(state):
    customer_id = state['next_customer_id']
    state['next_customer_id'] += 1
    return customer_id


def get_current_time_min(state):
    return state['current_time_min']


def get_queue_length(state):
    return state['customers_in_queue']


def is_queue_empty(state):
    return state['customers_in_queue'] == 0


def get_barista_count(state):
    return state['barista_count']


def get_time_avg_queue_length(state, until_time_min):
    if until_time_min <= 0:
        return 0.0
    return state['time_weighted_queue_length_area'] / until_time_min


def get_time_avg_system_count(state, until_time_min):
    if until_time_min <= 0:
        return 0.0
    return state['time_weighted_in_system_area'] / until_time_min


def get_utilization(state, until_time_min):
    if until_time_min <= 0 or state['barista_count'] == 0:
        return 0.0
    return state['time_weighted_busy_servers_area'] / (until_time_min * state['barista_count'])


def set_warmup_end_time(state, time_min):
    state['warmup_end_time_min'] = time_min


def end_warmup_phase(state):
    state['warmup_phase_active'] = False
    state['time_weighted_queue_length_area'] = 0.0
    state['time_weighted_in_system_area'] = 0.0
    state['time_weighted_busy_servers_area'] = 0.0
    state['last_event_time_min'] = state['current_time_min']


def is_warmup_active(state):
    return state['warmup_phase_active']


def get_all_completed_customers(state):
    return [
        c for c in state['customers'].values()
        if c['service_end_time_min'] > 0
    ]
