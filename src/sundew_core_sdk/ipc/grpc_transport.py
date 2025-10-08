"""gRPC transport layer binding for the IPC adapter."""

from __future__ import annotations

from concurrent import futures
from typing import Iterable, Iterator, Tuple

import grpc

from .adapter import IPCAdapter
from sundew_ipc_v1_pb2 import Acknowledge, GateDecision, TelemetryPush
from sundew_ipc_v1_pb2_grpc import (
    SundewGateServicer,
    SundewGateStub,
    add_SundewGateServicer_to_server,
)


class GrpcSundewGate(SundewGateServicer):
    def __init__(self, adapter: IPCAdapter) -> None:
        self.adapter = adapter

    def Connect(self, request_iterator: Iterable, context) -> Iterator[GateDecision]:
        for event in request_iterator:
            decision = self.adapter.handle_score_event(event)
            yield decision

    def PushTelemetry(self, request: TelemetryPush, context) -> Acknowledge:
        self.adapter.record_telemetry(request)
        return Acknowledge(sequence=request.samples, status=1)


def serve(
    adapter: IPCAdapter,
    host: str = '127.0.0.1',
    port: int = 50051,
    max_workers: int = 2,
) -> Tuple[grpc.Server, int]:
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=max_workers))
    add_SundewGateServicer_to_server(GrpcSundewGate(adapter), server)
    bound_port = server.add_insecure_port(f'{host}:{port}')
    server.start()
    return server, bound_port


__all__ = ['serve', 'GrpcSundewGate', 'SundewGateStub']
