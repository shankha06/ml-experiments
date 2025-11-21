#!/bin/bash

# Configuration
PYTHON_SCRIPT="worker_task.py" # Name of your python file
TIMEOUT_DURATION="61m"         # 61 minutes
LOG_FILE="process_log.txt"     # Where to save the output
MAX_RUNS=10                    # Maximum number of loops

echo "--- Starting Watchdog for $PYTHON_SCRIPT ---"
echo "Time limit set to: $TIMEOUT_DURATION"
echo "Max runs: $MAX_RUNS"
echo "Press [CTRL+C] to stop the loop."

# Initialize counter
CURRENT_RUN=1

# Loop until max runs is reached
while [ $CURRENT_RUN -le $MAX_RUNS ]; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Starting run $CURRENT_RUN of $MAX_RUNS at: $(date)" | tee -a "$LOG_FILE"

    # logic: 
    # timeout [duration] [command]
    # -k 1m: If it doesn't close after the timeout signal, force kill it 1 min later
    timeout -k 1m "$TIMEOUT_DURATION" python3 "$PYTHON_SCRIPT"
    
    # Capture the exit code of the python script (or the timeout command)
    EXIT_CODE=$?

    # Analyze the exit code
    if [ $EXIT_CODE -eq 124 ]; then
        echo "ALERT: Script timed out after $TIMEOUT_DURATION." | tee -a "$LOG_FILE"
        echo "Action: Killing process..." | tee -a "$LOG_FILE"
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "ALERT: Script crashed or exited with error (Code: $EXIT_CODE)." | tee -a "$LOG_FILE"
    else
        echo "INFO: Script finished successfully (Code 0)." | tee -a "$LOG_FILE"
    fi

    # Increment the run counter
    ((CURRENT_RUN++))

    # Only sleep if we are going to run again
    if [ $CURRENT_RUN -le $MAX_RUNS ]; then
        echo "Action: Restarting in 5 seconds..." | tee -a "$LOG_FILE"
        sleep 5
    else
        echo "Action: Max runs reached. Exiting watchdog." | tee -a "$LOG_FILE"
    fi
done