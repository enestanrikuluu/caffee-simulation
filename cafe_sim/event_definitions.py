EVENT_TYPE_ARRIVAL = "arrival"
EVENT_TYPE_SERVICE_END = "service_end"
EVENT_TYPE_END_OF_RUN = "end_of_run"

_event_sequence_counter = 0


def create_event(time_min, event_type, entity_id, data=None, sequence_number=None):
    global _event_sequence_counter
    if sequence_number is None:
        _event_sequence_counter += 1
        sequence_number = _event_sequence_counter

    return {
        'time_min': time_min,
        'event_type': event_type,
        'entity_id': entity_id,
        'data': data,
        'sequence_number': sequence_number,
    }


def reset_event_sequence():
    global _event_sequence_counter
    _event_sequence_counter = 0


def event_sort_key(event):
    return (event['time_min'], event['sequence_number'])
