"""Keep training ranks synchronized while rank zero runs Fisher measurement."""

import time
from concurrent.futures import Future
from threading import Thread

import torch.distributed as dist


def rank_zero_with_heartbeat(action, interval=1.0):
    """Broadcast readiness while a rank-zero action runs, then share its result."""
    future = Future()
    if dist.get_rank() == 0:
        def run():
            try:
                future.set_result(action())
            except Exception as error:
                future.set_exception(error)

        Thread(target=run, daemon=True).start()
    while True:
        message = [None]
        if dist.get_rank() == 0 and future.done():
            try:
                message[0] = {"result": future.result()}
            except Exception as error:
                message[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(message, src=0)
        if message[0] is not None:
            if "error" in message[0]:
                raise RuntimeError(message[0]["error"])
            return message[0]["result"]
        time.sleep(interval)
