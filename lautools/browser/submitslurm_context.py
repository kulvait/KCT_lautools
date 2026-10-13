from __future__ import annotations

import shlex
from dataclasses import dataclass, field

# Options the wrapper understands. Options that take a value (also -a, -f,
# --pipeline-step) are skipped as a pair so their values are never parsed.
CONTEXT_OPTIONS = {"-w": "working dir", "-p": "pattern", "-f": "partition"}

WRAPPER = r'''
declare -gA LAUTOOLS_SS_DEFAULT=()
declare -gA LAUTOOLS_SS_FORCE=()
submitslurm() {
    local -a out=() pre=()
    local -A seen=()
    local tok key opt
    while (($#)); do
        tok="$1"; key=""
        case "$tok" in
            -w|--working-dir) key=-w ;;
            -p|--pattern)     key=-p ;;
            -f|--partition)   key=-f ;;
        esac
        case "$tok" in
            -w|--working-dir|-p|--pattern|-f|--partition|-a|--slurmargs|--pipeline-step)
                if [[ -n $key && -n ${LAUTOOLS_SS_FORCE[$key]+x} && -n ${LAUTOOLS_SS_DEFAULT[$key]+x} ]]; then
                    shift 2 2>/dev/null || shift; continue   # forced: drop caller's value
                fi
                [[ -n $key ]] && seen[$key]=1
                if (($# >= 2)); then out+=("$tok" "$2"); shift 2; else out+=("$tok"); shift; fi ;;
            *) out+=("$tok"); shift ;;
        esac
    done
    for opt in "${!LAUTOOLS_SS_DEFAULT[@]}"; do
        [[ -z ${seen[$opt]+x} ]] && pre+=("$opt" "${LAUTOOLS_SS_DEFAULT[$opt]}")
    done
    printf '+ submitslurm' >&2; printf ' %q' "${pre[@]}" "${out[@]}" >&2; printf '\n' >&2
    command submitslurm "${pre[@]}" "${out[@]}"
}
'''


@dataclass
class SubmitslurmContext:
    enabled: bool = True
    values: dict[str, str] = field(default_factory=dict)   # {"-w": "wd", "-p": "17k"}
    forced: set[str] = field(default_factory=set)          # keys that override the call

    def to_bash(self) -> str:
        lines = ["LAUTOOLS_SS_DEFAULT=()", "LAUTOOLS_SS_FORCE=()"]
        if self.enabled:
            for key, value in self.values.items():
                if value.strip():
                    lines.append(f"LAUTOOLS_SS_DEFAULT[{key}]={shlex.quote(value.strip())}")
                    if key in self.forced:
                        lines.append(f"LAUTOOLS_SS_FORCE[{key}]=1")
        return "\n".join(lines) + "\n"
