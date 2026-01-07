from data.swesmith.generate_with_plain_docker import create_sandboxed_tasks
from data.unique_scale.utils import dedup_tasks
from data.commons import upload_tasks_to_hf

def main() -> None:
    swesmith_tasks_path = create_sandboxed_tasks(limit=1e10)
    deduped_tasks = dedup_tasks(swesmith_tasks_path)
    breakpoint()
    upload_tasks_to_hf(str(deduped_tasks), "DCAgent/exp_usc_swesmith")


if __name__ == "__main__":
    main()