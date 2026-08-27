#!/usr/bin/env bash
set -uo pipefail

report_dir="${1:-reports}"
cases=(
  tensor_core
  fp32_pipe
  instructions
  dram_bandwidth
  l2_bandwidth
  register_limited
  shared_limited
  long_scoreboard
  short_scoreboard
  barrier
)

if [[ ! -x ./ncu_diagnostic_tree ]]; then
  echo "ncu_diagnostic_tree does not exist; run 'make ARCH=sm_XX' first" >&2
  exit 2
fi

mkdir -p "${report_dir}"
failed=()

for case_name in "${cases[@]}"; do
  echo
  echo "===== profiling ${case_name} ====="
  if ./profile_one.sh "${case_name}" "${report_dir}"; then
    echo "===== completed ${case_name} ====="
  else
    echo "===== FAILED ${case_name} =====" >&2
    failed+=("${case_name}")
  fi
done

echo
echo "Generated reports:"
for case_name in "${cases[@]}"; do
  if [[ -f "${report_dir}/${case_name}.ncu-rep" ]]; then
    echo "  ${report_dir}/${case_name}.ncu-rep"
  fi
done

if (( ${#failed[@]} > 0 )); then
  echo "Failed cases: ${failed[*]}" >&2
  exit 1
fi

echo "All ${#cases[@]} cases completed successfully."
