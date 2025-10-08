#include "sundew_ipc_v1.h"

sundew_status_t sundew_gate_init(const sundew_init_request_t *req, sundew_init_response_t *resp) {
    if (!req || !resp) {
        return SUNDEW_STATUS_INVALID_PAYLOAD;
    }
    resp->status = SUNDEW_STATUS_OK;
    resp->heartbeat_interval_ms = 2000;
    resp->message = "shim";
    return SUNDEW_STATUS_OK;
}

sundew_status_t sundew_gate_score(const sundew_score_event_t *event, sundew_gate_decision_t *decision) {
    if (!event || !decision) {
        return SUNDEW_STATUS_INVALID_PAYLOAD;
    }
    decision->sequence = event->sequence;
    decision->should_activate = event->sequence % 2;
    decision->confidence = 0.5;
    decision->threshold = 0.2;
    decision->risk_probability = 0.3;
    decision->status = SUNDEW_STATUS_OK;
    return SUNDEW_STATUS_OK;
}

sundew_status_t sundew_gate_telemetry(const sundew_telemetry_push_t *telemetry, sundew_ack_t *ack) {
    if (!telemetry || !ack) {
        return SUNDEW_STATUS_INVALID_PAYLOAD;
    }
    ack->sequence += 1;
    ack->status = SUNDEW_STATUS_OK;
    return SUNDEW_STATUS_OK;
}
