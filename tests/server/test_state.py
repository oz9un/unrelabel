from unrelabel.server.state import AppState


def test_initial_state_is_empty():
    state = AppState()
    assert state.dataset is None
    assert state.model is None
    assert state.result is None
    assert state.baseline_accuracy is None


def test_reset_clears_all():
    state = AppState()
    state.baseline_accuracy = 0.95
    state.reset()
    assert state.baseline_accuracy is None
