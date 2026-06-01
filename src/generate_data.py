import json
import random
import numpy as np
import os

# Set random seeds
random.seed(42)
np.random.seed(42)

def generate_traces():
    categories = [
        {
            "label": "AuthTokenExpiry",
            "class": "com.example.AuthTokenExpiry",
            "msg": "Token expired during request",
            "severity": "warning",
            "frames": [
                {"file": "AuthInterceptor.kt", "method": "intercept", "line": 42},
                {"file": "TokenRefreshManager.kt", "method": "refresh", "line": 15}
            ]
        },
        {
            "label": "NullPointerException",
            "class": "java.lang.NullPointerException",
            "msg": "Attempt to invoke virtual method on a null object reference",
            "severity": "error",
            "frames": [
                {"file": "ProfileLoader.kt", "method": "loadProfile", "line": 89},
                {"file": "ProfileLoader.kt", "method": "parseResponse", "line": 105}
            ]
        },
        {
            "label": "OutOfMemoryError",
            "class": "java.lang.OutOfMemoryError",
            "msg": "Failed to allocate a 1048576 byte allocation with 4194304 free bytes",
            "severity": "error",
            "frames": [
                {"file": "ImageCache.kt", "method": "allocate", "line": 120},
                {"file": "ImageCache.kt", "method": "getBitmap", "line": 45}
            ]
        },
        {
            "label": "NetworkTimeoutException",
            "class": "java.net.SocketTimeoutException",
            "msg": "timeout",
            "severity": "warning",
            "frames": [
                {"file": "ApiClient.kt", "method": "execute", "line": 200},
                {"file": "ApiClient.kt", "method": "fetch", "line": 50}
            ]
        },
        {
            "label": "DatabaseCursorException",
            "class": "android.database.CursorIndexOutOfBoundsException",
            "msg": "Index -1 requested, with a size of 10",
            "severity": "error",
            "frames": [
                {"file": "ChildDataRepository.kt", "method": "getChild", "line": 34},
                {"file": "ChildDataRepository.kt", "method": "queryDb", "line": 70}
            ]
        },
        {
            "label": "ANROnMainThread",
            "class": "com.example.ANRError",
            "msg": "Application Not Responding",
            "severity": "error",
            "frames": [
                {"file": "SyncService.kt", "method": "onStartCommand", "line": 100},
                {"file": "MainThread.java", "method": "run", "line": 42}
            ]
        },
        {
            "label": "ClassCastException",
            "class": "java.lang.ClassCastException",
            "msg": "String cannot be cast to Integer",
            "severity": "error",
            "frames": [
                {"file": "NotificationFactory.kt", "method": "build", "line": 33},
                {"file": "NotificationFactory.kt", "method": "getExtras", "line": 55}
            ]
        },
        {
            "label": "IndexOutOfBoundsException",
            "class": "java.lang.IndexOutOfBoundsException",
            "msg": "Index 5 out of bounds for length 3",
            "severity": "error",
            "frames": [
                {"file": "AttendanceAdapter.kt", "method": "onBindViewHolder", "line": 40},
                {"file": "AttendanceAdapter.kt", "method": "getItem", "line": 65}
            ]
        },
        {
            "label": "FileNotFoundException",
            "class": "java.io.FileNotFoundException",
            "msg": "No such file or directory",
            "severity": "error",
            "frames": [
                {"file": "MediaUploader.kt", "method": "openFile", "line": 120},
                {"file": "MediaUploader.kt", "method": "upload", "line": 45}
            ]
        },
        {
            "label": "StackOverflowError",
            "class": "java.lang.StackOverflowError",
            "msg": "stack size 8MB",
            "severity": "error",
            "frames": [
                {"file": "RecursiveTreeView.kt", "method": "draw", "line": 20},
                {"file": "RecursiveTreeView.kt", "method": "draw", "line": 20}
            ]
        }
    ]

    counts = [45, 55, 48, 52, 50, 42, 58, 47, 53, 50]
    
    traces = []
    clusters = {c["label"]: [] for c in categories}
    
    trace_id_counter = 1
    
    for count, cat in zip(counts, categories):
        for _ in range(count):
            # Base trace
            trace_id = f"trace_{trace_id_counter:03d}"
            trace_id_counter += 1
            
            exception_class = cat["class"]
            message = cat["msg"]
            severity = cat["severity"]
            frames = [dict(f) for f in cat["frames"]]
            
            # Apply noise with 15% probability
            if random.random() < 0.15:
                # Add noise
                noise_type = random.choice(['line_num', 'message', 'swap'])
                if noise_type == 'line_num':
                    idx = random.randint(0, len(frames)-1)
                    frames[idx]["line"] += random.randint(-10, 10)
                    frames[idx]["line"] = max(1, frames[idx]["line"])
                elif noise_type == 'message':
                    message += f" (code: {random.randint(1, 999)})"
                elif noise_type == 'swap':
                    if len(frames) > 1:
                        frames[0], frames[1] = frames[1], frames[0]
            
            # Construct raw text
            raw_text_lines = [f"{exception_class}: {message}"]
            for f in frames:
                raw_text_lines.append(f"  at {f['file']}:{f['method']}({f['line']})")
            raw_text = "\n".join(raw_text_lines)
            
            trace = {
                "id": trace_id,
                "exception_class": exception_class,
                "message": message,
                "android_version": random.choice(["11", "12", "13", "14"]),
                "device_model": random.choice(["Pixel 6", "Pixel 7", "Galaxy S22", "Galaxy S23"]),
                "app_version": random.choice(["3.4.0", "3.4.1", "3.5.0"]),
                "severity": severity,
                "frames": frames,
                "raw_text": raw_text
            }
            
            traces.append(trace)
            clusters[cat["label"]].append(trace_id)
            
    # Shuffle traces
    random.shuffle(traces)
    
    # Save traces
    os.makedirs("data", exist_ok=True)
    with open("data/raw_traces.json", "w") as f:
        json.dump(traces, f, indent=2)
        
    # Save 5 hand-labelled clusters
    os.makedirs("eval", exist_ok=True)
    selected_labels = ["AuthTokenExpiry", "NullPointerException", "OutOfMemoryError", "NetworkTimeoutException", "DatabaseCursorException"]
    ground_truth = {
        "clusters": [
            {"label": lbl, "trace_ids": clusters[lbl]} for lbl in selected_labels
        ]
    }
    with open("eval/ground_truth.json", "w") as f:
        json.dump(ground_truth, f, indent=2)
        
    print("Generation complete!")
    print(f"Total traces: {len(traces)}")
    print("Traces per category:")
    for lbl, ids in clusters.items():
        print(f"  {lbl}: {len(ids)}")

if __name__ == "__main__":
    generate_traces()
