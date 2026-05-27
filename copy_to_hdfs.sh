#!/bin/bash
# Copy data_MeanFlow to HDFS-FUSE target in parallel.
# Strategy: one cp per file, run up to PARALLEL files concurrently.
# Big files (data.mdb shards) saturate independent FUSE upload streams.
set -euo pipefail

SRC=/opt/tiger/toys/MeanFlow/data_MeanFlow
DST=/mnt/hdfs/sg/junkun/data_and_model/open_source/ILSVRC/imagenet-1k
PARALLEL=${PARALLEL:-8}
LOG=/tmp/copy_to_hdfs.log
: > "$LOG"

echo "src: $SRC"
echo "dst: $DST"
echo "parallel: $PARALLEL"

# Pre-create the directory tree so cp doesn't race on mkdir.
( cd "$SRC" && find . -type d -print0 ) \
  | xargs -0 -I {} mkdir -p "$DST/{}"

# Total bytes for progress sanity.
SRC_BYTES=$(du -sb "$SRC" | awk '{print $1}')
echo "total bytes: $SRC_BYTES ($(numfmt --to=iec --suffix=B "$SRC_BYTES"))"

START=$(date +%s)

# Copy each file in parallel. cp -f overwrites; --no-preserve=mode avoids
# chmod round-trips on FUSE.
( cd "$SRC" && find . -type f -print0 ) \
  | xargs -0 -n1 -P "$PARALLEL" -I {} bash -c '
      rel="$1"
      src="'"$SRC"'/$rel"
      dst="'"$DST"'/$rel"
      t0=$(date +%s)
      sz=$(stat -c %s "$src")
      cp -f --no-preserve=mode "$src" "$dst"
      t1=$(date +%s)
      dur=$((t1 - t0))
      [ "$dur" -gt 0 ] && rate=$(( sz / dur / 1048576 )) || rate=inf
      printf "[%ds] done %s (%s, %s MB/s)\n" "$dur" "$rel" \
        "$(numfmt --to=iec --suffix=B "$sz")" "$rate" | tee -a '"$LOG"'
    ' _ {}

END=$(date +%s)
ELAPSED=$((END - START))
AVG_MBPS=$(( SRC_BYTES / ELAPSED / 1048576 ))
echo
echo "elapsed: ${ELAPSED}s  avg throughput: ${AVG_MBPS} MB/s"

echo "verifying sizes..."
DST_BYTES=$(du -sb "$DST" | awk '{print $1}')
if [ "$SRC_BYTES" = "$DST_BYTES" ]; then
  echo "OK: src and dst byte counts match ($SRC_BYTES)"
else
  echo "MISMATCH: src=$SRC_BYTES dst=$DST_BYTES" >&2
  exit 1
fi
