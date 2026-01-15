import pytest
import numpy as np
from cafe_sim.simulation_state import SimulationState


class TestAreaIntegrals:
    def test_queue_length_area_simple(self):
        state = SimulationState(barista_count=2)
        
        state.advance_time_to(0.0)
        state.enqueue_customer(1)
        state.enqueue_customer(2)
        
        state.advance_time_to(5.0)
        
        time_avg_queue = state.get_time_avg_queue_length(5.0)
        
        assert time_avg_queue == 2.0
    
    def test_queue_length_area_varying(self):
        state = SimulationState(barista_count=2)
        
        state.advance_time_to(0.0)
        
        state.enqueue_customer(1)
        state.advance_time_to(2.0)
        
        state.enqueue_customer(2)
        state.advance_time_to(5.0)
        
        state.dequeue_customer()
        state.advance_time_to(8.0)
        
        expected_area = 1 * 2.0 + 2 * 3.0 + 1 * 3.0
        expected_avg = expected_area / 8.0
        
        time_avg_queue = state.get_time_avg_queue_length(8.0)
        
        assert np.isclose(time_avg_queue, expected_avg)
    
    def test_utilization_single_server(self):
        state = SimulationState(barista_count=1)
        
        state.advance_time_to(0.0)
        
        state.mark_barista_busy(0, 3.0)
        state.advance_time_to(3.0)
        
        state.release_barista(0)
        state.advance_time_to(10.0)
        
        utilization = state.get_utilization(10.0)
        
        assert np.isclose(utilization, 0.3)
    
    def test_utilization_multiple_servers(self):
        state = SimulationState(barista_count=3)
        
        state.advance_time_to(0.0)
        
        state.mark_barista_busy(0, 5.0)
        state.mark_barista_busy(1, 5.0)
        
        state.advance_time_to(5.0)
        
        state.release_barista(0)
        state.release_barista(1)
        
        state.mark_barista_busy(2, 10.0)
        state.advance_time_to(10.0)
        
        expected_busy_area = 2 * 5.0 + 1 * 5.0
        expected_utilization = expected_busy_area / (10.0 * 3)
        
        utilization = state.get_utilization(10.0)
        
        assert np.isclose(utilization, expected_utilization)
    
    def test_warmup_resets_areas(self):
        state = SimulationState(barista_count=2)
        
        state.advance_time_to(0.0)
        state.enqueue_customer(1)
        state.enqueue_customer(2)
        
        state.advance_time_to(5.0)
        
        time_avg_before = state.get_time_avg_queue_length(5.0)
        assert time_avg_before == 2.0
        
        state.set_warmup_end_time(5.0)
        state.end_warmup_phase()
        
        state.advance_time_to(10.0)
        
        time_avg_after = state.get_time_avg_queue_length(5.0)
        assert time_avg_after == 2.0
    
    def test_system_count_area(self):
        state = SimulationState(barista_count=2)
        
        state.advance_time_to(0.0)
        
        state.enqueue_customer(1)
        state.enqueue_customer(2)
        state.mark_barista_busy(0, 10.0)
        state.mark_barista_busy(1, 10.0)
        
        state.advance_time_to(5.0)
        
        state.dequeue_customer()
        state.dequeue_customer()
        
        state.advance_time_to(10.0)
        
        expected_in_system_area = (2 + 2) * 5.0 + (0 + 2) * 5.0
        expected_avg = expected_in_system_area / 10.0
        
        time_avg_system = state.get_time_avg_system_count(10.0)
        
        assert np.isclose(time_avg_system, expected_avg)
    
    def test_no_time_advancement_zero_area(self):
        state = SimulationState(barista_count=2)
        
        state.enqueue_customer(1)
        state.enqueue_customer(2)
        
        time_avg_queue = state.get_time_avg_queue_length(0.0)
        
        assert time_avg_queue == 0.0
    
    def test_area_accumulation_with_multiple_updates(self):
        state = SimulationState(barista_count=1)
        
        times = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        queue_levels = [0, 2, 1, 3, 0, 1]
        
        for i in range(len(times)):
            state.advance_time_to(times[i])
            
            while state.get_queue_length() < queue_levels[i]:
                customer_id = i * 10 + state.get_queue_length()
                state.enqueue_customer(customer_id)
            
            while state.get_queue_length() > queue_levels[i]:
                state.dequeue_customer()
        
        state.advance_time_to(6.0)
        
        expected_area = 0*1 + 2*1 + 1*1 + 3*1 + 0*1 + 1*1
        expected_avg = expected_area / 6.0
        
        time_avg_queue = state.get_time_avg_queue_length(6.0)
        
        assert np.isclose(time_avg_queue, expected_avg)
    
    def test_backward_time_raises_error(self):
        state = SimulationState(barista_count=1)
        
        state.advance_time_to(5.0)
        
        with pytest.raises(ValueError, match="Cannot move time backwards"):
            state.advance_time_to(3.0)
    
    def test_utilization_zero_servers(self):
        state = SimulationState(barista_count=0)
        
        state.advance_time_to(5.0)
        
        utilization = state.get_utilization(5.0)
        
        assert utilization == 0.0
