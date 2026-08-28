"""List task ids and their configuration factories."""

from instinctlab.tasks.registry import TASKS


def main() -> None:
    task_width = max(len("Task"), *(len(task_id) for task_id in TASKS))
    print(f'{"Task":<{task_width}}  Config factory')
    print(f'{"-" * task_width}  {"-" * 40}')
    for task_id, registered in sorted(TASKS.items()):
        if isinstance(registered, str):
            print(f"{task_id:<{task_width}}  {registered}")
            continue
        for engine, factory in sorted(registered.items()):
            print(f"{task_id:<{task_width}}  [{engine}] {factory}")


if __name__ == "__main__":
    main()
