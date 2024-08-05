from unittest.mock import MagicMock

from pyclad.callbacks.callback import Callback
from pyclad.callbacks.composite_callback import CallbackComposite


def test_composite_calls_all_callbacks():
    callbacks = [Callback(), Callback(), Callback()]
    for c in callbacks:
        setattr(c, "before_training", MagicMock())
    composite = CallbackComposite(callbacks)
    composite.before_training()

    for c in callbacks:
        getattr(c, "before_training").assert_called_once()
