# Remote Run Drop Folder

Use this folder for scenario run files copied from the experiment machine.
The copied run artifacts are ignored by git; this README is kept so the folder exists.

## Pull The Latest Scenario Run

Run this from the local machine:

```bash
REMOTE=USER@REMOTE_HOST
REMOTE_ABIODUN=/path/to/remote/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
LOCAL_ABIODUN=/home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun
RUN=$(ssh "$REMOTE" "ls -td '$REMOTE_ABIODUN'/metrics_logs/scenesense_scenarios/* | head -1")
mkdir -p "$LOCAL_ABIODUN/remote_files"
rsync -avP "$REMOTE:$RUN/" "$LOCAL_ABIODUN/remote_files/$(basename "$RUN")/"
```

## Pull A Specific Scenario Run

```bash
rsync -avP USER@REMOTE_HOST:/path/to/remote/abiodun/metrics_logs/scenesense_scenarios/RUN_FOLDER/ \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/remote_files/RUN_FOLDER/
```

## Smaller Debug Copy

If the full run folder is large, copy only the files needed for diagnosis:

```bash
rsync -avP \
  USER@REMOTE_HOST:/path/to/remote/abiodun/metrics_logs/scenesense_scenarios/RUN_FOLDER/summary.txt \
  USER@REMOTE_HOST:/path/to/remote/abiodun/metrics_logs/scenesense_scenarios/RUN_FOLDER/scenario_event_summary.json \
  USER@REMOTE_HOST:/path/to/remote/abiodun/metrics_logs/scenesense_scenarios/RUN_FOLDER/scenario_event_trace.csv \
  USER@REMOTE_HOST:/path/to/remote/abiodun/metrics_logs/scenesense_scenarios/RUN_FOLDER/scenario_manifest.json \
  USER@REMOTE_HOST:/path/to/remote/abiodun/metrics_logs/scenesense_scenarios/RUN_FOLDER/actors.json \
  /home/shr_aisvcs/workarea/carla_0_10_env/Carla-0.10.0-Linux-Shipping/PythonAPI/neu_collab/abiodun/remote_files/RUN_FOLDER/
```
