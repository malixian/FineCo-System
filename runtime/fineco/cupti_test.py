import os
import ctypes
#import pandas as pd

# Load CUPTI library
cupti_path = os.environ.get("CUPTI_LIBRARY_PATH")
if not cupti_path:
    raise ValueError("CUPTI_LIBRARY_PATH environment variable is not set")
cupti = ctypes.cdll.LoadLibrary(cupti_path)

# Initialize CUPTI
cupti_err = cupti.cuptiActivityInitialize(0)
if cupti_err != 0:
    raise ValueError("Failed to initialize CUPTI: {cupti_err}")

# Start CUPTI activity tracing
cupti_err = cupti.cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE)
if cupti_err != 0:
    raise ValueError("Failed to enable CUPTI activity tracing: {cupti_err}")

# Run CUDA application
#my_cuda_app()

# Stop CUPTI activity tracing
cupti_err = cupti.cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DEVICE)
if cupti_err != 0:
    raise ValueError("Failed to disable CUPTI activity tracing: {cupti_err}")

# Retrieve CUPTI activity records
buffer_size = 4 * 1024 * 1024
buffer = ctypes.create_string_buffer(buffer_size)
cupti_err = cupti.cuptiActivityGetAllRecords(ctypes.byref(buffer), ctypes.byref(ctypes.c_size_t(buffer_size)))
if cupti_err != 0:
    raise ValueError("Failed to get CUPTI activity records: {cupti_err}")

# Convert activity records to pandas DataFrame
activity_records = pd.DataFrame.from_records(ctypes.cast(buffer, ctypes.POINTER(cupti_activity_record)).contents for i in range(num_records))

# Print active warps per second
active_warps = activity_records[activity_records.kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL].groupby("start").max().active_warps
active_warps_per_second = active_warps.diff().fillna(0)
print("Active warps per second: {active_warps_per_second.mean()}")

