#!/bin/bash
# Monitor pre-computation progress

echo "=== Pre-computation Progress Monitor ==="
echo "Started at: $(date)"
echo ""

while true; do
    clear
    echo "=== Pre-computation Progress Monitor ==="
    echo "Time: $(date)"
    echo ""

    # Check if process is running
    if docker exec 75bb13cca2c5 pgrep -f "precompute_augmentations_parallel.py" > /dev/null; then
        echo "✓ Pre-computation process is RUNNING"

        # Show process stats
        echo ""
        echo "--- Process Stats ---"
        docker exec 75bb13cca2c5 ps aux | grep precompute | grep python | head -1 | awk '{print "CPU: " $3 "%, Memory: " $4 "%, Time: " $10}'

        # Count worker processes
        WORKERS=$(docker exec 75bb13cca2c5 ps aux | grep -c "python -u scripts/precompute_augmentations_parallel.py")
        echo "Active workers: $WORKERS"

        # Show recent progress logs
        echo ""
        echo "--- Recent Progress ---"
        docker exec 75bb13cca2c5 tail -50 /tmp/precompute_parallel.log | grep -E "(INFO|Progress|Success|Failed|Cache)" | tail -10

        # Check if cache file exists
        echo ""
        echo "--- Cache Status ---"
        if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.pkl; then
            SIZE=$(docker exec 75bb13cca2c5 ls -lh experiments/augmentation_cache.pkl | awk '{print $5}')
            echo "✓ Cache file exists: $SIZE"
        else
            echo "⏳ Cache file not yet created"
        fi

        # Check checkpoint
        if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.checkpoint.pkl; then
            SIZE=$(docker exec 75bb13cca2c5 ls -lh experiments/augmentation_cache.checkpoint.pkl | awk '{print $5}')
            echo "✓ Checkpoint exists: $SIZE"
        fi
    else
        echo "✗ Pre-computation process is NOT RUNNING"

        # Check if completed successfully
        if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.pkl; then
            echo ""
            echo "✓✓✓ Pre-computation COMPLETED! ✓✓✓"
            echo ""
            SIZE=$(docker exec 75bb13cca2c5 ls -lh experiments/augmentation_cache.pkl | awk '{print $5}')
            echo "Cache file: $SIZE"

            # Show final stats
            if docker exec 75bb13cca2c5 test -f experiments/augmentation_cache.json; then
                echo ""
                echo "--- Final Statistics ---"
                docker exec 75bb13cca2c5 cat experiments/augmentation_cache.json | grep -E "(num_texts|num_methods|total_cached|success|failed)" | head -10
            fi

            echo ""
            echo "Ready to start HPO!"
            break
        else
            echo ""
            echo "--- Last 20 log lines ---"
            docker exec 75bb13cca2c5 tail -20 /tmp/precompute_parallel.log
        fi

        break
    fi

    echo ""
    echo "Refreshing in 30 seconds... (Ctrl+C to stop monitoring)"
    sleep 30
done
