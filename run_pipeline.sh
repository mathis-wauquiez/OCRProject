#!/bin/bash
# ==========================================================================
#  Master script — run the full OCR pipeline end-to-end
#
#  Usage:
#    bash run_pipeline.sh                     # run all stages
#    bash run_pipeline.sh --book book1        # specify book name
#    bash run_pipeline.sh --skip-download     # skip image download
#    bash run_pipeline.sh --skip-build        # skip C++ vectorizer build
#    bash run_pipeline.sh --only extraction   # run a single stage
#    bash run_pipeline.sh --from clustering   # resume from a stage
# ==========================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──
BOOK="book1"
SKIP_DOWNLOAD=false
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
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
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
            echo "  --skip-download          Skip image download step"
            echo "  --skip-build             Skip C++ vectorizer build"
            echo "  --only STAGE             Run only one stage"
            echo "  --from STAGE             Resume from a stage (inclusive)"
            echo "  --workers N              Parallel workers for extraction (default: 1)"
            echo "  --extraction-config NAME Hydra config for extraction (default: extraction_pipeline)"
            echo "  --preprocessing-config N Hydra config for preprocessing (default: preprocessing)"
            echo ""
            echo "Stages: download, build, extraction, preprocessing, clustering"
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
        local stages=(download build extraction preprocessing clustering)
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

# ── Stage 0: Download data ──
if should_run "download" && [[ "$SKIP_DOWNLOAD" == false ]]; then
    echo ">> Stage 0: Downloading images..."
    python scripts/download_data.py
    echo ""
fi

# ── Stage 0b: Build C++ vectorizer ──
if should_run "build" && [[ "$SKIP_BUILD" == false ]]; then
    echo ">> Stage 0b: Building C++ vectorizer..."
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
        --save-folder "outputs/extraction/${BOOK}" \
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

# ── Stage 3: Clustering ──
if should_run "clustering"; then
    echo ">> Stage 3: Clustering + post-clustering refinement..."
    python scripts/sweep_clustering.py
    echo ""
fi

echo "=========================================="
echo "  Pipeline complete!"
echo "=========================================="
echo ""
echo "Outputs:"
echo "  Extraction:     outputs/extraction/${BOOK}/"
echo "  Preprocessing:  outputs/preprocessing/${BOOK}/"
echo "  Clustering:     outputs/clustering/${BOOK}/"
echo ""
