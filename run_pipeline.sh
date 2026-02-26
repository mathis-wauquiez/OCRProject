#!/bin/bash
# ==========================================================================
#  Master script — run the full OCR pipeline end-to-end
#
#  Assumes images are already in data/datasets/<book>/.
#
#  Usage:
#    bash run_pipeline.sh                     # run all stages
#    bash run_pipeline.sh --book book1        # specify book name
#    bash run_pipeline.sh --skip-build        # skip C++ vectorizer build
#    bash run_pipeline.sh --only extraction   # run a single stage
#    bash run_pipeline.sh --from clustering   # resume from a stage
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──
BOOK="book1"
SKIP_BUILD=false
ONLY=""
FROM=""
WORKERS=1
EXTRACTION_CONFIG="extraction_pipeline"
PREPROCESSING_CONFIG="preprocessing"

# ── Parse arguments ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --book)          BOOK="$2"; shift 2 ;;
        --skip-build)    SKIP_BUILD=true; shift ;;
        --only)          ONLY="$2"; shift 2 ;;
        --from)          FROM="$2"; shift 2 ;;
        --workers)       WORKERS="$2"; shift 2 ;;
        --extraction-config)    EXTRACTION_CONFIG="$2"; shift 2 ;;
        --preprocessing-config) PREPROCESSING_CONFIG="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: bash run_pipeline.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --book NAME              Book name (default: book1)"
            echo "  --skip-build             Skip C++ vectorizer build"
            echo "  --only STAGE             Run only one stage"
            echo "  --from STAGE             Resume from a stage (inclusive)"
            echo "  --workers N              Parallel workers for extraction (default: 1)"
            echo "  --extraction-config NAME Hydra config for extraction (default: extraction_pipeline)"
            echo "  --preprocessing-config N Hydra config for preprocessing (default: preprocessing)"
            echo ""
            echo "Stages: build, extraction, preprocessing, alignment, clustering, figure, figures"
            exit 0
            ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Stage control ──
should_run() {
    local stage="$1"
    if [[ -n "$ONLY" ]]; then
        [[ "$stage" == "$ONLY" ]]
        return
    fi
    if [[ -n "$FROM" ]]; then
        local stages=(build extraction preprocessing alignment clustering figure figures)
        local from_idx=-1 stage_idx=-1
        for i in "${!stages[@]}"; do
            [[ "${stages[$i]}" == "$FROM" ]] && from_idx=$i
            [[ "${stages[$i]}" == "$stage" ]] && stage_idx=$i
        done
        [[ $stage_idx -ge $from_idx ]]
        return
    fi
    return 0
}

echo "=========================================="
echo "  OCR Pipeline — Full Run"
echo "  Book: $BOOK"
echo "=========================================="
echo ""

# Check that images exist
if [[ ! -d "data/datasets/${BOOK}" ]]; then
    echo "ERROR: Image directory data/datasets/${BOOK} not found."
    echo "Place your scanned page images there before running the pipeline."
    exit 1
fi

# ── Stage 0: Build C++ vectorizer ──
if should_run "build" && [[ "$SKIP_BUILD" == false ]]; then
    echo ">> Stage 0: Building C++ vectorizer..."
    if [[ ! -f src/vectorization/build/main ]]; then
        (cd src/vectorization && bash build_script.sh)
    else
        echo "   Vectorizer already built, skipping. Use --skip-build=false to force."
    fi
    echo ""
fi

# ── Stage 1: Extraction ──
if should_run "extraction"; then
    echo ">> Stage 1: Character extraction..."
    python scripts/run_extraction.py \
        --image-folder "data/datasets/${BOOK}" \
        --save-folder "results/extraction/${BOOK}" \
        --config "$EXTRACTION_CONFIG" \
        --workers "$WORKERS"
    echo ""
fi

# ── Stage 2: Preprocessing ──
if should_run "preprocessing"; then
    echo ">> Stage 2: Preprocessing (vectorize + HOG + CHAT OCR)..."
    python scripts/run_preprocessing.py --config-name "$PREPROCESSING_CONFIG"
    echo ""
fi

# ── Stage 3: Transcription alignment ──
if should_run "alignment"; then
    echo ">> Stage 3: Transcription alignment..."
    python scripts/align_transcription.py \
        --dataframe "results/preprocessing/${BOOK}" \
        --transcriptions "data/datasets/transcript/" \
        --output "results/preprocessing/${BOOK}" \
        --images "data/datasets/${BOOK}"
    echo ""
fi

# ── Stage 4: Clustering ──
if should_run "clustering"; then
    echo ">> Stage 4: Clustering + post-clustering refinement..."
    python scripts/run_clust_sweep.py
    echo ""
fi

# ── Stage 5: Main figure generation ──
if should_run "figure"; then
    echo ">> Stage 5: Generating main pipeline figure..."
    # Pick the first image in the dataset as the example page
    EXAMPLE_IMAGE=$(ls "data/datasets/${BOOK}/"*.jpg 2>/dev/null | head -1)
    if [[ -z "$EXAMPLE_IMAGE" ]]; then
        EXAMPLE_IMAGE=$(ls "data/datasets/${BOOK}/"*.png 2>/dev/null | head -1)
    fi
    if [[ -n "$EXAMPLE_IMAGE" ]]; then
        BASENAME=$(basename "$EXAMPLE_IMAGE")
        COMPONENTS_FILE="results/extraction/${BOOK}/components/${BASENAME}.npz"

        FIGURE_ARGS=(
            --image "$EXAMPLE_IMAGE"
            --dataframe "results/preprocessing/${BOOK}"
            --output "paper/figures/generated/main_pipeline.pdf"
        )
        if [[ -f "$COMPONENTS_FILE" ]]; then
            FIGURE_ARGS+=(--components "$COMPONENTS_FILE")
        fi

        python scripts/figure_generation/generate_paper_main_figure.py "${FIGURE_ARGS[@]}"
    else
        echo "   WARNING: No images found in data/datasets/${BOOK}/, skipping figure."
    fi
    echo ""
fi

# ── Stage 6: All paper figures ──
if should_run "figures"; then
    echo ">> Stage 6: Generating all paper figures..."
    python scripts/figure_generation/generate_all_figures.py \
        --clustering-dir "results/clustering/${BOOK}" \
        --preprocessing-dir "results/preprocessing/${BOOK}" \
        --images-dir "data/datasets/${BOOK}" \
        --output-dir "paper/figures/generated"
    echo ""
fi

echo "=========================================="
echo "  Pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Extraction:     results/extraction/${BOOK}/"
echo "  Preprocessing:  results/preprocessing/${BOOK}/"
echo "  Alignment viz:  results/preprocessing/alignment_viz/"
echo "  Clustering:     results/clustering/${BOOK}/"
echo "  Main figure:    paper/figures/generated/main_pipeline.pdf"
echo "  All figures:    paper/figures/generated/"
echo ""
