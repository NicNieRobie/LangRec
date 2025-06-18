import os.path

from datasphere.job_orchestrator import DataSphereJobOrchestrator
from utils.logger import configure_logger

if __name__ == "__main__":
    configure_logger()

    import csv

    pending = []
    with open(os.path.join('datasphere_data', 'tasks_args.csv'), newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row.get("requires_cloud", "").lower() == "true":
                pending.append(row)

    orchestrator = DataSphereJobOrchestrator()
    orchestrator.stage_jobs(pending)

    try:
        orchestrator.run()
    except KeyboardInterrupt:
        print("Ctrl+C received. Cleaning up...")
        orchestrator.terminate()
        print("Clean shutdown complete.")
