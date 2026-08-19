"""
Script to print all the available environments in the extension.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import sys
import traceback
from prettytable import PrettyTable

import instinctlab.tasks

# Registering the Gym ids used to be a side effect of the import above. It is an explicit call now,
# and a caller that only imports the package sees an empty table rather than an error.
instinctlab.tasks.register_legacy_isaac_tasks()


def main():
    """Print all environments registered in `isaac.lab_demo` extension."""
    # print all the available environments
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab Template Extension"
    # set alignment of table columns
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    # count of environments
    index = 0
    # acquire all Isaac environments names
    for task_spec in gym.registry.values():
        if "Instinct-" in task_spec.id:
            # add details to table
            table.add_row([index + 1, task_spec.id, task_spec.entry_point, task_spec.kwargs["env_cfg_entry_point"]])
            # increment count
            index += 1

    print(table)


if __name__ == "__main__":
    import os

    status = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        status = 1

    # Isaac Sim's shutdown decides this process's exit status on its own, so a failure reported
    # after ``close()`` reaches the caller as success. Flush, then leave before it gets the chance.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
