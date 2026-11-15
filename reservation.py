import cv2
import numpy as np
import os
import time
import json
import difflib
import threading
import winsound
from pathlib import Path


# All project data is stored relative to this file, not the original developer's Mac.
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RESERVATION_SAVE_DIR = BASE_DIR / "reservation"
RESERVATION_EXTRACTION_FILE = DATA_DIR / "reservation_extraction.txt"
RESERVED_FILE = DATA_DIR / "reserved.txt"
USER_RESERVATIONS_FILE = DATA_DIR / "User_Reservations.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESERVATION_SAVE_DIR.mkdir(parents=True, exist_ok=True)


# Clear temporary text files at startup, matching the original project behavior.
FILES_TO_CLEAR = [RESERVATION_EXTRACTION_FILE, RESERVED_FILE]

for file_path in FILES_TO_CLEAR:
    try:
        file_path.write_text("", encoding="utf-8")
    except OSError as exc:
        print(f"Warning: Could not clear {file_path}: {exc}")


# Clear old reservation snapshots at startup, matching the original project behavior.
# Remove this block if you want to keep snapshots from previous runs.
for old_file in RESERVATION_SAVE_DIR.iterdir():
    try:
        if old_file.is_file():
            old_file.unlink()
    except OSError as exc:
        print(f"Warning: Could not delete {old_file}: {exc}")


# Comparison similarity threshold (0-1)
SIMILARITY_THRESHOLD = 0.85

# Global trackers
snapshot_counter = {}
last_snap_time = {}
prev_reserved_occupied = {}


def beep(duration=0.5, freq=1000):
    """Play a beep using Windows' built-in sound support."""
    try:
        winsound.Beep(int(freq), int(duration * 1000))
    except RuntimeError as exc:
        print(f"Audio alert unavailable: {exc}")


def alert_slot(slot_name):
    for _ in range(3):
        beep(0.5)
        time.sleep(0.2)


# Helper: load registered plate
def _get_registered_plate(slot_name):
    """Return registered plate (no spaces, uppercase) for a given slot, or None."""
    try:
        if not USER_RESERVATIONS_FILE.exists():
            return None

        with USER_RESERVATIONS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)

        for entry in data:
            if (
                entry.get("slotId") == slot_name
                and entry.get("status", "").lower() == "active"
            ):
                plate = entry.get("vehiclePlate", "")
                return plate.replace(" ", "").upper()

    except Exception as exc:
        print(f"Warning: Error reading reservations file: {exc}")

    return None


# Helper: parse extraction line
def _parse_extraction_line(line):
    try:
        if "-" not in line:
            return None, None

        filename_part, plate_part = line.strip().split("-", 1)
        filename = filename_part.strip()
        plate = plate_part.strip().replace(" ", "").upper()
        return filename, plate

    except Exception:
        return None, None


# Comparator
def _compare_and_print(slot_name, detected_plate):
    try:
        registered_plate = _get_registered_plate(slot_name)

        if not registered_plate:
            print(f"No active reservation found for {slot_name}")
            return

        registered_plate = registered_plate.replace(" ", "").upper()
        detected_plate = detected_plate.replace(" ", "").upper()

        similarity = difflib.SequenceMatcher(
            None, registered_plate, detected_plate
        ).ratio()

        diff_count = sum(
            1 for a, b in zip(registered_plate, detected_plate) if a != b
        )
        diff_count += abs(len(registered_plate) - len(detected_plate))

        if similarity >= SIMILARITY_THRESHOLD or diff_count <= 3:
            print(
                f"{slot_name}: Match | Registered: {registered_plate} "
                f"| Detected: {detected_plate}"
            )
        else:
            print(
                f"{slot_name}: Mismatch | Registered: {registered_plate} "
                f"| Detected: {detected_plate}"
            )
            alert_slot(slot_name)

    except Exception as exc:
        print(f"Error comparing plates for {slot_name}: {exc}")


# Background watcher
def _comparison_watcher(start_delay=5, poll_interval=1):
    processed_lines = set()

    time.sleep(start_delay)
    print(
        f"Plate comparator started after {start_delay} seconds. "
        f"Monitoring {RESERVATION_EXTRACTION_FILE}"
    )

    while True:
        try:
            if RESERVATION_EXTRACTION_FILE.exists():
                with RESERVATION_EXTRACTION_FILE.open(
                    "r", encoding="utf-8"
                ) as file:
                    lines = [line.rstrip("\n") for line in file]

                for line in lines:
                    if not line or line in processed_lines:
                        continue

                    if "reservation_slot_" in line and "-" in line:
                        filename, plate = _parse_extraction_line(line)

                        if filename and plate:
                            slot_name = None

                            if "reservation_slot_" in filename:
                                try:
                                    after = filename.split("reservation_slot_", 1)[1]
                                    slot_name = after.split("_", 1)[0].upper()
                                except Exception:
                                    slot_name = None

                            if slot_name:
                                _compare_and_print(slot_name, plate)
                            else:
                                print(
                                    f"Could not infer slot from filename: {filename}"
                                )

                    processed_lines.add(line)

            time.sleep(poll_interval)

        except Exception as exc:
            print(f"Comparator watcher error: {exc}")
            time.sleep(poll_interval)


def start_comparison_watcher(start_delay=5, poll_interval=1):
    thread = threading.Thread(
        target=_comparison_watcher,
        args=(start_delay, poll_interval),
        daemon=True,
    )
    thread.start()


# Reservation snapshot function
def handle_reservation_snapshots(
    posList_parking_video,
    posList_parking_image,
    img_portrait2,
    current_status,
    reserved_slots,
    target_slot=None,
):
    global snapshot_counter, last_snap_time, prev_reserved_occupied

    total_slots = (
        len(posList_parking_image[1])
        if len(posList_parking_image) > 1
        else len(posList_parking_image[0])
    )

    if not snapshot_counter:
        snapshot_counter = {f"B{i + 1}": 0 for i in range(total_slots)}

    if not last_snap_time:
        last_snap_time = {f"B{i + 1}": 0 for i in range(total_slots)}

    if not prev_reserved_occupied:
        prev_reserved_occupied = {
            f"B{i + 1}": False for i in range(total_slots)
        }

    active_coords = (
        posList_parking_image[1]
        if len(posList_parking_image) > 1
        else posList_parking_image[0]
    )

    for i, polygon in enumerate(active_coords):
        slot_name = f"B{i + 1}"

        if slot_name not in reserved_slots:
            prev_reserved_occupied[slot_name] = False
            continue

        if target_slot is not None and slot_name != target_slot:
            continue

        if i >= len(current_status):
            continue

        is_occupied = current_status[i]["occupied"]
        now = time.time()
        was_prev_reserved_occupied = prev_reserved_occupied[slot_name]

        if (
            is_occupied
            and not was_prev_reserved_occupied
            and now - last_snap_time[slot_name] >= 5
        ):
            points = np.array(polygon, np.int32)
            mask = np.zeros_like(img_portrait2, dtype=np.uint8)
            cv2.fillPoly(mask, [points], (255, 255, 255))
            masked_image = cv2.bitwise_and(img_portrait2, mask)

            x, y, width, height = cv2.boundingRect(points)
            snapshot = masked_image[y : y + height, x : x + width]

            snapshot_counter[slot_name] += 1
            filename = RESERVATION_SAVE_DIR / (
                f"reservation_slot_{slot_name}_"
                f"{snapshot_counter[slot_name]}.jpg"
            )

            saved = cv2.imwrite(str(filename), snapshot)
            if saved:
                last_snap_time[slot_name] = now
            else:
                print(f"Warning: Could not save reservation snapshot: {filename}")

        prev_reserved_occupied[slot_name] = is_occupied


# Start comparison watcher automatically
start_comparison_watcher(start_delay=5, poll_interval=1)
