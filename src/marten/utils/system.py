import os
import time
import fcntl
import threading

from sqlalchemy import Engine, text
from typing import List


def get_physical_core_ids():
    core_info = {}
    with open("/proc/cpuinfo") as f:
        for line in f:
            if line.strip():
                if line.startswith("processor"):
                    cpu_id = int(line.strip().split(": ")[1])
                elif line.startswith("core id"):
                    core_id = int(line.strip().split(": ")[1])
                    core_info[cpu_id] = core_id
    # Map physical core IDs to one of their logical CPUs
    unique_cores = {}
    for cpu_id, core_id in core_info.items():
        if core_id not in unique_cores.values():
            unique_cores[cpu_id] = core_id
    return list(unique_cores.keys())


def release_cpu_cores(alchemyEngine: Engine, cores: List[int]):
    with alchemyEngine.begin() as conn:
        # Lock the row for exclusive access
        result = conn.execute(
            text(
            """
            SELECT value FROM sys_params
            WHERE name = :name
            FOR UPDATE
            """
            ),
            {"name": "unique_cpu_core_ids"},
        )
        row = result.fetchone()
        current_cores = []
        if row and row[0]:
            current_cores = [
                int(core_id) for core_id in row[0].split(",") if core_id.strip()
            ]
        # Combine and remove duplicates
        new_cores = sorted(set(current_cores + cores))
        updated_cores_str = ",".join(str(core_id) for core_id in new_cores)
        conn.execute(
            text(
            """
            UPDATE sys_params SET value = :value
            WHERE name = :name
            """
            ),
            {"value": updated_cores_str, "name": "unique_cpu_core_ids"},
        )


def bind_cpu_cores(
    alchemyEngine: Engine, num_cores: int = 1, max_wait: float = 15.0
) -> List[int]:
    def _allocate():
        nonlocal alchemyEngine, num_cores
        with alchemyEngine.begin() as conn:
            # Start a transaction and lock the row for exclusive access
            result = conn.execute(
                text(
                """
                SELECT value FROM sys_params
                WHERE name = :name
                FOR UPDATE
                """
                ),
                {"name": "unique_cpu_core_ids"},
            )
            row = result.fetchone()
            if not row or not row[0]:
                return None

            # Parse the available core IDs from the comma-separated string
            available_core_ids = [
                int(core_id) for core_id in row[0].split(",") if core_id.strip()
            ]

            if not available_core_ids:
                return None

            requested_cores = num_cores
            num_cores_to_allocate = min(requested_cores, len(available_core_ids))
            # if len(available_core_ids) < num_cores:
            #     raise Exception(
            #         f"Not enough cores available: requested {num_cores}, available {len(available_core_ids)}"
            #     )

            # Allocate the required number of cores
            allocated_cores = available_core_ids[:num_cores_to_allocate]
            remaining_cores = available_core_ids[num_cores_to_allocate:]

            # Update the value in the database after removal
            updated_cores_str = ",".join(str(core_id) for core_id in remaining_cores)

            # Update the database with the remaining cores
            conn.execute(
                text(
                """
                UPDATE sys_params SET value = :value
                WHERE name = :name
                """
                ),
                {"value": updated_cores_str, "name": "unique_cpu_core_ids"},
            )

            return allocated_cores

    start_time = time.time()
    while True:
        allocated_cores = _allocate()

        if allocated_cores:
            # p = psutil.Process()
            # p.cpu_affinity(allocated_cores)

            # Set CPU affinity for the current thread
            thread_id = threading.get_ident()
            # Note: affects the calling process or thread, depending on the OS and implementation.
            os.sched_setaffinity(0, allocated_cores)

            return allocated_cores

        if max_wait and time.time() - start_time > max_wait:
            return []

        time.sleep(0.5)


def init_cpu_core_id(alchemyEngine: Engine):
    core_ids = get_physical_core_ids()
    core_ids_str = ",".join(str(core_id) for core_id in core_ids)
    with alchemyEngine.begin() as conn:
        conn.execute(
            text(
            """
            INSERT INTO sys_params (name, value)
            VALUES (:name, :value)
            ON CONFLICT (name) DO UPDATE SET value = EXCLUDED.value
            """
            ),
            {"name": "unique_cpu_core_ids", "value": core_ids_str},
        )

def bool_envar(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


class FileLock:
    def __init__(self, name):
        self.lockfile = name
        self.fd = None

    def acquire(self, timeout=10):
        self.fd = open(self.lockfile, "w")
        start = time.time()
        while True:
            try:
                fcntl.flock(self.fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                # self.fd.close()
                if time.time() - start > timeout:
                    return False
                time.sleep(0.1)
            except OSError as e:
                # Handle file system errors
                self.fd.close()
                raise e

    def release(self):
        if self.fd:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            self.fd.close()
            self.fd = None
