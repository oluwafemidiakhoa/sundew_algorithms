
# SDK Developer Dependencies

Some tooling used during Phase 1 lives outside the standard `pip install -r requirements.txt` set.

## IPC tooling (`grpcio-tools`)
```bash
pip install grpcio grpcio-tools
python tools/generate_ipc_bindings.py
```

This compiles `docs/sdk/ipc/sundew_ipc_v1.proto` to Python bindings inside `src/`.

## Plotting utilities (`matplotlib`, `Pillow`)
```bash
pip install matplotlib pillow
python benchmarks/power/plot_export.py results/balanced.json
```

The plot script emits helpful errors when these packages are absent.

## Optional: C toolchain
To build the IPC shim stub locally, install gcc/clang (Linux/macOS) or MSVC Build Tools (Windows), then run:
```bash
python tools/build_sundew_shim.py
```
If no compiler is found the script will exit with guidance.


## Protobuf runtime version
If you run the IPC tests with the system Python, ensure `protobuf` matches the
version used for code generation (currently 6.31.1). Otherwise, the loader emits
`google.protobuf.runtime_version.VersionError` and tests skip. On Windows, you
can install the newer runtime via:
```bash
python -m pip install --upgrade protobuf
```
