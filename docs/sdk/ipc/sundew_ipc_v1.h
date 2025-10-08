#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    SUNDEW_STATUS_OK = 1,
    SUNDEW_STATUS_INVALID_PAYLOAD = 2,
    SUNDEW_STATUS_UNSUPPORTED_FEATURE = 3,
    SUNDEW_STATUS_OVERLOAD = 4
} sundew_status_t;

typedef struct {
    const char *board;
    const char *firmware_version;
    const char *sdk_version;
    const char *config_hash;
} sundew_init_request_t;

typedef struct {
    sundew_status_t status;
    uint32_t heartbeat_interval_ms;
    const char *message;
} sundew_init_response_t;

typedef struct {
    const char *key;
    double value;
} sundew_feature_kv_t;

typedef struct {
    uint64_t sequence;
    int64_t timestamp_ns;
    const sundew_feature_kv_t *features;
    size_t feature_count;
} sundew_score_event_t;

typedef struct {
    uint64_t sequence;
    int should_activate;
    double confidence;
    double threshold;
    double risk_probability;
    sundew_status_t status;
} sundew_gate_decision_t;

typedef struct {
    double activation_rate;
    double average_power_w;
    double energy_buffer;
    double temperature_c;
    uint64_t samples;
} sundew_telemetry_push_t;

typedef struct {
    uint64_t sequence;
    sundew_status_t status;
} sundew_ack_t;

int sundew_gate_init(const sundew_init_request_t *req, sundew_init_response_t *resp);
int sundew_gate_score(const sundew_score_event_t *event, sundew_gate_decision_t *decision);
int sundew_gate_telemetry(const sundew_telemetry_push_t *telemetry, sundew_ack_t *ack);

#ifdef __cplusplus
}
#endif
