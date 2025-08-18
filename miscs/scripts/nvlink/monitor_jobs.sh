#!/bin/bash
# Job monitoring script for lmms-eval SLURM jobs - FIXED VERSION

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Source configuration
source "$SCRIPT_DIR/config.sh"

echo "=========================================="
echo "SLURM Job Monitor for lmms-eval"
echo "=========================================="
echo ""

# Function to show job summary
show_job_summary() {
    echo "📊 Job Summary for user: $USER"
    echo "----------------------------------------"
    
    # Count jobs by state (using 197_lmms-eval prefix)
    running=0
    pending=0
    
    # Check running jobs
    if squeue -u $USER -h -t R | grep -q "197_lmms-eval"; then
        running=$(squeue -u $USER -h -t R | grep "197_lmms-eval" | wc -l)
    fi
    
    # Check pending jobs
    if squeue -u $USER -h -t PD | grep -q "197_lmms-eval"; then
        pending=$(squeue -u $USER -h -t PD | grep "197_lmms-eval" | wc -l)
    fi
    
    total=$((running + pending))
    
    echo "  Running: $running"
    echo "  Pending: $pending"
    echo "  Total: $total"
    echo ""
}

# Function to show detailed job list
show_job_details() {
    echo "📋 Detailed Job List:"
    echo "----------------------------------------"
    
    # Check if there are any jobs
    if squeue -u $USER | grep -q "197_lmms-eval"; then
        squeue -u $USER -o "%.18i %.40j %.8T %.10M %.6D %R" | grep -E "(JOBID|197_lmms-eval)"
    else
        echo "No lmms-eval jobs found"
    fi
    echo ""
}

# Function to show recent completions
show_recent_completions() {
    echo "✅ Recently Completed Jobs (last 24h):"
    echo "----------------------------------------"
    
    # Get date for 24 hours ago
    start_date=$(date -d '24 hours ago' +%Y-%m-%d)
    
    # Check for recent completions
    if sacct -u $USER -S $start_date --format=JobID,JobName%40,State,ExitCode,Elapsed -n | grep -q "197_lmms-eval"; then
        sacct -u $USER -S $start_date --format=JobID,JobName%40,State,ExitCode,Elapsed -n | grep "197_lmms-eval" | head -20
    else
        echo "No recent completions"
    fi
    echo ""
}

# Function to check output files
check_outputs() {
    echo "📁 Recent Output Files:"
    echo "----------------------------------------"
    if [ -d "$OUTPUT_BASE_DIR" ]; then
        # Find recent JSON files
        recent_files=$(find "$OUTPUT_BASE_DIR" -type f -name "*.json" -mtime -1 2>/dev/null | head -10)
        if [ -n "$recent_files" ]; then
            echo "$recent_files" | while read -r file; do
                ls -lh "$file"
            done
        else
            echo "No recent outputs"
        fi
    else
        echo "Output directory not found"
    fi
    echo ""
}

# Function to check log files for errors
check_errors() {
    echo "⚠️  Recent Errors in Logs:"
    echo "----------------------------------------"
    if [ -d "$LOG_BASE_DIR" ]; then
        # Check for non-empty error files
        error_files=$(find "$LOG_BASE_DIR" -name "*.err" -size +0 -mtime -1 2>/dev/null | head -5)
        if [ -n "$error_files" ]; then
            echo "$error_files" | while read -r err_file; do
                echo "Error in: $err_file"
                echo "Last 5 lines:"
                tail -5 "$err_file"
                echo "---"
            done
        else
            echo "No recent errors found (or all error files are empty)"
        fi
    else
        echo "Log directory not found"
    fi
    echo ""
}

# Function to cancel all jobs
cancel_all_jobs() {
    echo "🛑 Cancelling all lmms-eval jobs..."
    
    # Get job IDs
    job_ids=$(squeue -u $USER -h -o "%i %j" | grep "197_lmms-eval" | awk '{print $1}')
    
    if [ -z "$job_ids" ]; then
        echo "No jobs to cancel"
    else
        for job_id in $job_ids; do
            echo "Cancelling job $job_id..."
            scancel $job_id
        done
        echo "All jobs cancelled"
    fi
}

# Main function
main() {
    # Check for specific command
    if [ $# -ge 1 ]; then
        case $1 in
            cancel)
                cancel_all_jobs
                exit 0
                ;;
            details)
                show_job_details
                exit 0
                ;;
            errors)
                check_errors
                exit 0
                ;;
            outputs)
                check_outputs
                exit 0
                ;;
            summary)
                show_job_summary
                exit 0
                ;;
            *)
                echo "Unknown command: $1"
                echo "Available commands: cancel, details, errors, outputs, summary"
                exit 1
                ;;
        esac
    fi
    
    # Default: show all information
    show_job_summary
    show_job_details
    show_recent_completions
    check_outputs
    check_errors
    
    echo "=========================================="
    echo "💡 Tips:"
    echo "  - Cancel all jobs: $0 cancel"
    echo "  - Show only details: $0 details"
    echo "  - Check errors: $0 errors"
    echo "  - Check outputs: $0 outputs"
    echo "  - Summary only: $0 summary"
    echo "  - Watch live: watch -n 10 $0"
    echo "=========================================="
}

# Run main function
main "$@"