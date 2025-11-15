#!/bin/bash
# Script to train models with different augmentation methods using best config
# Each training run will be saved to a unique timestamped directory

set -e  # Exit on error

echo "========================================="
echo "Training with Different Augmentation Methods"
echo "Using best_config as base configuration"
echo "========================================="

# Base command with best config
BASE_CMD="python -m src.training.train --config-name=best_config"

# Array of augmentation configurations
declare -a CONFIGS=(
    "dataset=original:No Augmentation (Original Data Only)"
    "dataset=original_nlpaug:NLPAug Only"
    "dataset=original_textattack:TextAttack Only"
    "dataset=original_hybrid:Hybrid (NLPAug + TextAttack)"
)

# Optional: Different encoders to test
declare -a ENCODERS=(
    "microsoft/deberta-base:DeBERTa"
    "FacebookAI/roberta-base:RoBERTa"
    "google-bert/bert-base-uncased:BERT"
)

# Function to run training
run_training() {
    local config=$1
    local encoder=$2
    local config_name=$3
    local encoder_name=$4

    echo ""
    echo "========================================="
    echo "Training: $config_name with $encoder_name"
    echo "Config: $config"
    echo "Encoder: $encoder"
    echo "========================================="

    $BASE_CMD $config model.pretrained_model_name=$encoder

    echo "✓ Completed: $config_name with $encoder_name"
    echo ""
}

# Parse command line arguments
TRAIN_MODE="augmentation"  # Default: train all augmentation methods
ENCODER_MODE="best"        # Default: use best encoder only

while [[ $# -gt 0 ]]; do
    case $1 in
        --mode)
            TRAIN_MODE="$2"
            shift 2
            ;;
        --encoder)
            ENCODER_MODE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --mode <mode>      Training mode (default: augmentation)"
            echo "                     - augmentation: Train with all augmentation methods"
            echo "                     - encoders: Train with all encoders"
            echo "                     - full: Train all combinations"
            echo "  --encoder <mode>   Encoder selection (default: best)"
            echo "                     - best: Use DeBERTa only"
            echo "                     - all: Use all encoders"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                              # Train all augmentation methods with DeBERTa"
            echo "  $0 --mode encoders              # Train all encoders with hybrid augmentation"
            echo "  $0 --mode full                  # Train all combinations"
            echo "  $0 --encoder all                # Train all augmentation methods with all encoders"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Execute training based on mode
case $TRAIN_MODE in
    augmentation)
        echo "Mode: Training all augmentation methods with best encoder (DeBERTa)"
        for config_entry in "${CONFIGS[@]}"; do
            IFS=':' read -r config description <<< "$config_entry"
            run_training "$config" "microsoft/deberta-base" "$description" "DeBERTa"
        done
        ;;

    encoders)
        echo "Mode: Training all encoders with hybrid augmentation"
        for encoder_entry in "${ENCODERS[@]}"; do
            IFS=':' read -r encoder encoder_name <<< "$encoder_entry"
            run_training "dataset=original_hybrid" "$encoder" "Hybrid Augmentation" "$encoder_name"
        done
        ;;

    full)
        echo "Mode: Training all combinations of augmentation methods and encoders"
        if [ "$ENCODER_MODE" = "all" ] || [ "$TRAIN_MODE" = "full" ]; then
            for config_entry in "${CONFIGS[@]}"; do
                IFS=':' read -r config description <<< "$config_entry"
                for encoder_entry in "${ENCODERS[@]}"; do
                    IFS=':' read -r encoder encoder_name <<< "$encoder_entry"
                    run_training "$config" "$encoder" "$description" "$encoder_name"
                done
            done
        else
            echo "Error: Full mode requires --encoder all"
            exit 1
        fi
        ;;

    *)
        echo "Error: Unknown mode '$TRAIN_MODE'"
        echo "Use --help for usage information"
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "All Training Runs Completed!"
echo "========================================="
echo ""
echo "Results are saved in unique timestamped directories under outputs/"
echo "View results with: ls -lt outputs/\$(date +%Y-%m-%d)/"
echo ""
echo "Compare results in MLflow:"
echo "  make mlflow-ui"
echo "  Open http://localhost:5000"
