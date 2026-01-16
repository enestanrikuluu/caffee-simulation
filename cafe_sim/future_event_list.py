import heapq
from cafe_sim.event_definitions import event_sort_key


def create_event_list():
    return {
        'heap': [],
        'counter': 0,
    }


def push_event(fel, event):
    fel['counter'] += 1
    event['sequence_number'] = fel['counter']
    heapq.heappush(fel['heap'], (event_sort_key(event), event))


def pop_next_event(fel):
    if len(fel['heap']) == 0:
        return None
    _, event = heapq.heappop(fel['heap'])
    return event


def is_event_list_empty(fel):
    return len(fel['heap']) == 0


def peek_next_time(fel):
    if len(fel['heap']) == 0:
        return None
    return fel['heap'][0][1]['time_min']


def event_list_size(fel):
    return len(fel['heap'])
