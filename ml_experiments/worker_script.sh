#!/bin/bash

# Configuration
PYTHON_SCRIPT="worker_task.py" # Name of your python file
TIMEOUT_DURATION="61m"         # 61 minutes
LOG_FILE="process_log.txt"     # Where to save the output

echo "--- Starting Watchdog for $PYTHON_SCRIPT ---"
echo "Time limit set to: $TIMEOUT_DURATION"
echo "Press [CTRL+C] to stop the loop."

# Infinite loop
while true; do
    echo "----------------------------------------" | tee -a "$LOG_FILE"
    echo "Starting execution at: $(date)" | tee -a "$LOG_FILE"

    # logic: 
    # timeout [duration] [command]
    # -k 1m: If it doesn't close after the timeout signal, force kill it 1 min later
    timeout -k 1m "$TIMEOUT_DURATION" python3 "$PYTHON_SCRIPT"
    
    # Capture the exit code of the python script (or the timeout command)
    EXIT_CODE=$?

    # Analyze the exit code
    if [ $EXIT_CODE -eq 124 ]; then
        echo "ALERT: Script timed out after $TIMEOUT_DURATION." | tee -a "$LOG_FILE"
        echo "Action: Killing process and restarting..." | tee -a "$LOG_FILE"
    elif [ $EXIT_CODE -ne 0 ]; then
        echo "ALERT: Script crashed or exited with error (Code: $EXIT_CODE)." | tee -a "$LOG_FILE"
        echo "Action: Restarting..." | tee -a "$LOG_FILE"
    else
        echo "INFO: Script finished successfully (Code 0)." | tee -a "$LOG_FILE"
        echo "Action: Restarting loop..." | tee -a "$LOG_FILE"
    fi

    # Small safety pause to prevent CPU thrashing if the script crashes instantly on boot
    echo "Cooling down for 5 seconds before restart..."
    sleep 5
done