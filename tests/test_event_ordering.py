import pytest
from cafe_sim.future_event_list import FutureEventList
from cafe_sim.event_definitions import Event, EventType


class TestFutureEventList:
    def test_events_ordered_by_time(self):
        fel = FutureEventList()
        
        event1 = Event(time_min=5.0, event_type=EventType.ARRIVAL, entity_id=1)
        event2 = Event(time_min=2.0, event_type=EventType.ARRIVAL, entity_id=2)
        event3 = Event(time_min=8.0, event_type=EventType.SERVICE_END, entity_id=3)
        
        fel.push(event1)
        fel.push(event2)
        fel.push(event3)
        
        assert fel.pop_next().entity_id == 2
        assert fel.pop_next().entity_id == 1
        assert fel.pop_next().entity_id == 3

    def test_simultaneous_events_ordered_by_insertion(self):
        fel = FutureEventList()
        
        event1 = Event(time_min=5.0, event_type=EventType.ARRIVAL, entity_id=1)
        event2 = Event(time_min=5.0, event_type=EventType.ARRIVAL, entity_id=2)
        event3 = Event(time_min=5.0, event_type=EventType.SERVICE_END, entity_id=3)
        
        fel.push(event1)
        fel.push(event2)
        fel.push(event3)
        
        first = fel.pop_next()
        second = fel.pop_next()
        third = fel.pop_next()
        
        assert first.entity_id == 1
        assert second.entity_id == 2
        assert third.entity_id == 3

    def test_is_empty(self):
        fel = FutureEventList()
        assert fel.is_empty()
        
        event = Event(time_min=5.0, event_type=EventType.ARRIVAL, entity_id=1)
        fel.push(event)
        assert not fel.is_empty()
        
        fel.pop_next()
        assert fel.is_empty()

    def test_peek_next_time(self):
        fel = FutureEventList()
        assert fel.peek_next_time() is None
        
        fel.push(Event(time_min=10.0, event_type=EventType.ARRIVAL, entity_id=1))
        fel.push(Event(time_min=5.0, event_type=EventType.ARRIVAL, entity_id=2))
        
        assert fel.peek_next_time() == 5.0
        assert fel.size() == 2

    def test_pop_from_empty_returns_none(self):
        fel = FutureEventList()
        assert fel.pop_next() is None
