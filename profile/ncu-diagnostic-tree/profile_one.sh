#!/usr/bin/env bash
set -euo pipefail

case_name="${1:-}"
report_dir="${2:-reports}"

if [[ -z "${case_name}" ]]; then
  echo "usage: $0 CASE [REPORT_DIR]" >&2
  ./ncu_diagnostic_tree --list >&2
  exit 2
fi

if [[ ! -x ./ncu_diagnostic_tree ]]; then
  echo "ncu_diagnostic_tree does not exist; run 'make' first" >&2
  exit 2
fi

if ! ./ncu_diagnostic_tree --list | awk '{print $1}' | grep -Fxq "${case_name}"; then
  echo "unknown case: ${case_name}" >&2
  ./ncu_diagnostic_tree --list >&2
  exit 2
fi

mkdir -p "${report_dir}"
ncu --set full \
  --target-processes all \
  --force-overwrite \
  --export "${report_dir}/${case_name}" \
  ./ncu_diagnostic_tree "${case_name}" --profile-once \
  2>&1 | tee "${report_dir}/${case_name}.txt"

echo "report: ${report_dir}/${case_name}.ncu-rep"
