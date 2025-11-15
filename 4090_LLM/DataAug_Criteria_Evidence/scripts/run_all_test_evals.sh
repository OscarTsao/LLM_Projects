#!/bin/bash
# Run test evaluations for all 4 architectures in parallel

echo "Starting test evaluations for all 4 architectures..."
echo "This will take approximately 4-6 hours"
echo ""

mkdir -p logs/test_eval_withaug

# Launch all 4 in background
echo "Launching Criteria..."
nohup python scripts/simple_test_eval.py --arch criteria --epochs 100 \
    > logs/test_eval_withaug/criteria.log 2>&1 &
echo $! > logs/test_eval_withaug/criteria.pid
CRITERIA_PID=$!

sleep 5

echo "Launching Evidence..."
nohup python scripts/simple_test_eval.py --arch evidence --epochs 100 \
    > logs/test_eval_withaug/evidence.log 2>&1 &
echo $! > logs/test_eval_withaug/evidence.pid
EVIDENCE_PID=$!

sleep 5

echo "Launching Share..."
nohup python scripts/simple_test_eval.py --arch share --epochs 100 \
    > logs/test_eval_withaug/share.log 2>&1 &
echo $! > logs/test_eval_withaug/share.pid
SHARE_PID=$!

sleep 5

echo "Launching Joint..."
nohup python scripts/simple_test_eval.py --arch joint --epochs 100 \
    > logs/test_eval_withaug/joint.log 2>&1 &
echo $! > logs/test_eval_withaug/joint.pid
JOINT_PID=$!

echo ""
echo "All evaluations launched:"
echo "  Criteria: PID $CRITERIA_PID"
echo "  Evidence: PID $EVIDENCE_PID"
echo "  Share: PID $SHARE_PID"
echo "  Joint: PID $JOINT_PID"
echo ""
echo "Monitor progress with:"
echo "  tail -f logs/test_eval_withaug/criteria.log"
echo "  tail -f logs/test_eval_withaug/evidence.log"
echo "  tail -f logs/test_eval_withaug/share.log"
echo "  tail -f logs/test_eval_withaug/joint.log"
echo ""
echo "Check GPU usage:"
echo "  nvidia-smi"
