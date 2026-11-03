
import os
import sys
import time
import threading
import traceback
from datetime import datetime

import pandas as pd

# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

RAW_DIR = os.path.join(
    BASE_DIR,
    "tools",
    "CICFlowMeter_corrected",
    "data",
    "daily"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

PROCESSED_DIR = os.path.join(
    REPORT_DIR,
    "live_monitor_processed"
)

# =========================================================
# MONITOR SETTINGS
# =========================================================

POLL_INTERVAL = 5
STABLE_CHECK_INTERVAL = 3
STABLE_CHECKS_REQUIRED = 3

# =========================================================
# LIVE MONITOR
# =========================================================


class LiveNIDSMonitor:

    def __init__(self, detector=None):

        self.detector = detector

        self.running = False
        self.thread = None

        self.stop_event = threading.Event()
        self.lock = threading.RLock()

        self.processed_files = {}
        self.latest_results = pd.DataFrame()

        self.status = "Stopped"
        self.last_error = None
        self.last_processed_file = None
        self.last_processed_time = None

        os.makedirs(
            REPORT_DIR,
            exist_ok=True
        )

        os.makedirs(
            PROCESSED_DIR,
            exist_ok=True
        )

        self._load_processed_state()

    # =====================================================
    # STATE FILE
    # =====================================================

    @property
    def state_file(self):

        return os.path.join(
            PROCESSED_DIR,
            "processed_state.csv"
        )

    def _load_processed_state(self):

        if not os.path.exists(self.state_file):
            return

        try:

            state = pd.read_csv(
                self.state_file
            )

            for _, row in state.iterrows():

                self.processed_files[
                    str(row["file_path"])
                ] = str(row["fingerprint"])

        except Exception as error:

            self.last_error = (
                f"State loading failed: {error}"
            )

    def _save_processed_state(self):

        rows = [
            {
                "file_path": path,
                "fingerprint": fingerprint
            }
            for path, fingerprint
            in self.processed_files.items()
        ]

        state_df = pd.DataFrame(
            rows,
            columns=[
                "file_path",
                "fingerprint"
            ]
        )

        temp_file = self.state_file + ".tmp"

        state_df.to_csv(
            temp_file,
            index=False
        )

        os.replace(
            temp_file,
            self.state_file
        )

    # =====================================================
    # FILE FINGERPRINT
    # =====================================================

    def _fingerprint(self, file_path):

        stat = os.stat(file_path)

        return (
            f"{stat.st_size}:"
            f"{stat.st_mtime_ns}"
        )

    # =====================================================
    # IGNORE FILES THAT EXIST BEFORE START
    # =====================================================

    def _baseline_existing_files(self):

        if not os.path.isdir(RAW_DIR):
            return

        with self.lock:

            for filename in os.listdir(RAW_DIR):

                if not filename.lower().endswith(".csv"):
                    continue

                file_path = os.path.join(
                    RAW_DIR,
                    filename
                )

                if not os.path.isfile(file_path):
                    continue

                try:

                    self.processed_files[file_path] = (
                        self._fingerprint(file_path)
                    )

                except OSError:
                    continue

            self._save_processed_state()

    # =====================================================
    # CHECK FILE STABILITY
    # =====================================================

    def _wait_until_stable(self, file_path):

        previous_fingerprint = None
        stable_count = 0

        while not self.stop_event.is_set():

            try:

                current_fingerprint = (
                    self._fingerprint(file_path)
                )

                if current_fingerprint == previous_fingerprint:

                    stable_count += 1

                else:

                    stable_count = 0

                previous_fingerprint = (
                    current_fingerprint
                )

                if stable_count >= STABLE_CHECKS_REQUIRED:

                    return True

            except (FileNotFoundError, PermissionError):

                stable_count = 0
                previous_fingerprint = None

            self.stop_event.wait(
                STABLE_CHECK_INTERVAL
            )

        return False

    # =====================================================
    # FIND NEW OR UPDATED FILES
    # =====================================================

    def _find_pending_files(self):

        if not os.path.isdir(RAW_DIR):
            return []

        pending = []

        for filename in os.listdir(RAW_DIR):

            if not filename.lower().endswith(".csv"):
                continue

            file_path = os.path.join(
                RAW_DIR,
                filename
            )

            if not os.path.isfile(file_path):
                continue

            try:

                fingerprint = self._fingerprint(
                    file_path
                )

            except OSError:
                continue

            old_fingerprint = (
                self.processed_files.get(file_path)
            )

            if fingerprint != old_fingerprint:

                pending.append(file_path)

        return sorted(pending)

    # =====================================================
    # CONVERT RAW CSV TO V3
    # =====================================================

    def _convert_file(self, raw_file):

        from preprocessing.cicflowmeter_to_v3 import (
            convert
        )

        filename = os.path.basename(raw_file)

        output_name = (
            os.path.splitext(filename)[0]
            + "_v3.csv"
        )

        output_file = os.path.join(
            PROCESSED_DIR,
            output_name
        )

        convert(
            raw_file,
            output_file
        )

        return output_file

    # =====================================================
    # RUN AI-NIDS PREDICTIONS
    # =====================================================

    def _predict_file(self, v3_file):

        if self.detector is None:

            from detection.unified_detector import (
                UnifiedNIDSDetector
            )

            self.detector = UnifiedNIDSDetector()

        input_df = pd.read_csv(v3_file)

        if input_df.empty:

            raise ValueError(
                "Converted CSV contains no flows."
            )

        results = self.detector.predict(
            input_df
        )

        results = results.copy()

        output_df = input_df.reset_index(
            drop=True
        ).copy()

        output_df = pd.concat(
            [
                output_df,
                results.reset_index(drop=True)
            ],
            axis=1
        )

        output_df = output_df.loc[
            :,
            ~output_df.columns.duplicated()
        ]

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )

        base_name = os.path.splitext(
            os.path.basename(v3_file)
        )[0]

        prediction_file = os.path.join(
            PROCESSED_DIR,
            f"{base_name}_predictions_{timestamp}.csv"
        )

        output_df.to_csv(
            prediction_file,
            index=False
        )

        with self.lock:

            self.latest_results = output_df.copy()

            self.last_processed_file = (
                os.path.basename(v3_file)
            )

            self.last_processed_time = (
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
            )

        return output_df, prediction_file

    # =====================================================
    # PROCESS ONE FILE
    # =====================================================

    def _process_file(self, raw_file):

        self.status = (
            f"Waiting for stable file: "
            f"{os.path.basename(raw_file)}"
        )

        if not self._wait_until_stable(raw_file):
            return

        try:

            fingerprint_before = self._fingerprint(
                raw_file
            )

            self.status = (
                f"Converting: {os.path.basename(raw_file)}"
            )

            v3_file = self._convert_file(
                raw_file
            )

            fingerprint_after = self._fingerprint(
                raw_file
            )

            # Do not accept a snapshot if capture changed.
            if fingerprint_before != fingerprint_after:

                self.status = (
                    "Capture changed during conversion. "
                    "Waiting for another stable snapshot."
                )

                return

            if self.stop_event.is_set():
                return

            self.status = "Running AI detection"

            results, prediction_file = (
                self._predict_file(v3_file)
            )

            with self.lock:

                self.processed_files[
                    raw_file
                ] = fingerprint_after

                self._save_processed_state()

            self.status = (
                f"Processed {len(results)} flows"
            )

            self.last_error = None

            print("\nAI-NIDS PROCESSING COMPLETE")

            print("Raw file:", raw_file)

            print("Prediction file:", prediction_file)

            print("Total flows:", len(results))

            if "verdict" in results.columns:

                print("\nPrediction distribution:")

                print(
                    results["verdict"]
                    .value_counts()
                    .to_string()
                )

        except Exception as error:

            self.last_error = (
                f"{type(error).__name__}: {error}"
            )

            self.status = "Processing failed"

            print("\nLIVE MONITOR ERROR:", self.last_error)

            traceback.print_exc()

    # =====================================================
    # BACKGROUND MONITOR LOOP
    # =====================================================

    def _monitor_loop(self):

        self.status = "Monitoring"

        print("\nAI-NIDS LIVE MONITOR STARTED")

        print("Watching folder:", RAW_DIR)

        try:

            while not self.stop_event.is_set():

                pending_files = (
                    self._find_pending_files()
                )

                for raw_file in pending_files:

                    if self.stop_event.is_set():
                        break

                    self._process_file(
                        raw_file
                    )

                self.stop_event.wait(
                    POLL_INTERVAL
                )

        except Exception as error:

            self.last_error = str(error)

            self.status = "Monitor error"

            traceback.print_exc()

        finally:

            with self.lock:
                self.running = False

            self.status = "Stopped"

            print("\nAI-NIDS LIVE MONITOR STOPPED")

    # =====================================================
    # START MONITORING
    # =====================================================

    def start(self):

        with self.lock:

            if self.running:
                return False

            if not os.path.isdir(RAW_DIR):

                self.last_error = (
                    f"CICFlowMeter folder not found: {RAW_DIR}"
                )

                self.status = "Folder not found"

                return False

            # Ignore all captures that already exist.
            self._baseline_existing_files()

            self.stop_event.clear()

            self.running = True

            self.last_error = None

            self.status = "Starting"

            self.thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="AI-NIDS-Live-Monitor"
            )

            self.thread.start()

            return True

    # =====================================================
    # STOP MONITORING
    # =====================================================

    def stop(self):

        with self.lock:

            if not self.running:
                return False

            self.stop_event.set()

            thread = self.thread

        if thread is not None and thread.is_alive():

            thread.join(timeout=10)

        with self.lock:

            if thread is not None and thread.is_alive():

                self.status = (
                    "Stopping: waiting for current processing"
                )

                return True

            self.running = False
            self.status = "Stopped"

        return True

    # =====================================================
    # GET STATUS
    # =====================================================

    def get_status(self):

        with self.lock:

            return {
                "running": self.running,
                "status": self.status,
                "last_error": self.last_error,
                "last_processed_file": (
                    self.last_processed_file
                ),
                "last_processed_time": (
                    self.last_processed_time
                ),
                "latest_results": (
                    self.latest_results.copy()
                )
            }


# =========================================================
# STANDALONE TEST
# =========================================================

if __name__ == "__main__":

    monitor = LiveNIDSMonitor()

    print("Raw folder:", RAW_DIR)

    print("Processed folder:", PROCESSED_DIR)

    if monitor.start():

        print(
            "\nMonitoring started."
            "\nExisting files will be ignored."
            "\nPress CTRL+C to stop."
        )

        try:

            while monitor.running:
                time.sleep(1)

        except KeyboardInterrupt:

            print("\nStopping monitor...")

            monitor.stop()

    else:

        print(
            "Unable to start monitor:",
            monitor.last_error
        )
