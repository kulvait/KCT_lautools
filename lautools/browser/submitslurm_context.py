"""Submission context of the Pipeline tab shell: submitslurm options added to every call.

The options are those of laupy.submitslurm_options (the submitslurm parser), so every
option that applies to all submissions can be set, by default (the call's own value wins)
or forced (replaces the call's value). Free text adds further arguments. The job
environment can be edited on top of what submitslurm computes (env_overrides, path_override).

The shell's submitslurm function runs submitslurm_merge.py, which merges each call with the
context written to a JSON file (write_json) and then runs the real submitslurm.
"""
from __future__ import annotations

import copy
import json
import os
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

from laupy.submitslurm_options import OptionSpec, context_option_specs

MODE_OFF, MODE_DEFAULT, MODE_FORCE = "off", "default", "force"
MODES = (MODE_OFF, MODE_DEFAULT, MODE_FORCE)

MERGE_SCRIPT = Path(__file__).with_name("submitslurm_merge.py")


def wrapper_bash(context_file: str | Path) -> str:
    """Shell code defining submitslurm() in the Pipeline tab shell."""
    return (
        f"_LAUTOOLS_PY={shlex.quote(sys.executable)}\n"
        f"_LAUTOOLS_SS_MERGE={shlex.quote(str(MERGE_SCRIPT))}\n"
        f"_LAUTOOLS_SS_CONTEXT={shlex.quote(str(context_file))}\n"
        "submitslurm() {\n"
        '    LAUTOOLS_SUBMITSLURM_CONTEXT="$_LAUTOOLS_SS_CONTEXT" "$_LAUTOOLS_PY" "$_LAUTOOLS_SS_MERGE" "$@"\n'
        "}\n"
    )


@dataclass
class OptionSetting:
    """mode: off/default/force. value: str for value and choice options, list[str] for
    multi_choice, shell words (str) for append options, ignored for flags."""
    mode: str = MODE_OFF
    value: object = None


@dataclass
class SubmitslurmContext:
    enabled: bool = True
    options: dict[str, OptionSetting] = field(default_factory=dict)   # by option dest
    free_text: str = ""
    job_path_from_project: bool = True       # default --job-path: the project PATH settings
    project_path_dirs: list[str] = field(default_factory=list)       # set by the main window
    env_overrides: dict[str, str] = field(default_factory=dict)      # edited job variables (--env)
    path_override: list[str] | None = None   # edited job PATH (--env PATH=...), None: computed

    def copy(self) -> "SubmitslurmContext":
        return copy.deepcopy(self)

    def setting(self, dest: str) -> OptionSetting:
        return self.options.get(dest) or OptionSetting()

    @staticmethod
    def tokens(spec: OptionSpec, setting: OptionSetting) -> list[str]:
        """Option tokens for a setting; empty if it adds nothing."""
        value = setting.value
        if spec.kind == "flag":
            return [spec.name]
        if spec.kind == "multi_choice":
            values = [v for v in (value or []) if v]
            return [spec.name, *values] if values else []
        if spec.kind == "append":
            words = shlex.split(value or "")
            return [token for word in words for token in (spec.name, word)]
        text = str(value).strip() if value is not None else ""
        return [spec.name, text] if text else []

    def effective_options(self) -> dict[str, OptionSetting]:
        """The option settings including the derived ones (job path from the project)."""
        options = dict(self.options)
        if (self.path_override is None and self.job_path_from_project and self.project_path_dirs
                and self.setting("job_path").mode == MODE_OFF):
            options["job_path"] = OptionSetting(MODE_DEFAULT, os.pathsep.join(self.project_path_dirs))
        if self.path_override is not None:
            # an edited PATH is exported as --env PATH=..., which excludes --job-path
            options.pop("job_path", None)
        return options

    def arguments(self) -> tuple[dict[str, list[str]], set[str], list[str]]:
        """(defaults, forced, extra) for laupy.submitslurm_options.merge_arguments().

        Raises ValueError if the free text cannot be split into words.
        """
        defaults: dict[str, list[str]] = {}
        forced: set[str] = set()
        options = self.effective_options()
        for spec in context_option_specs():
            setting = options.get(spec.dest)
            if setting is None or setting.mode == MODE_OFF:
                continue
            tokens = self.tokens(spec, setting)
            if tokens:
                defaults[spec.dest] = tokens
                if setting.mode == MODE_FORCE:
                    forced.add(spec.dest)
        env_tokens = []
        for name, value in self.env_overrides.items():
            env_tokens += ["--env", f"{name}={value}"]
        if self.path_override is not None:
            env_tokens += ["--env", "PATH=" + os.pathsep.join(self.path_override)]
        if env_tokens:
            defaults["env"] = defaults.get("env", []) + env_tokens
        extra = shlex.split(self.free_text) if self.free_text.strip() else []
        return defaults, forced, extra

    def write_json(self, path: str | Path) -> None:
        """The context for submitslurm_merge.py (written atomically)."""
        try:
            defaults, forced, extra = self.arguments()
            error = None
        except ValueError as e:
            defaults, forced, extra, error = {}, set(), [], f"free text: {e}"
        data = {"enabled": self.enabled and error is None, "defaults": defaults,
                "forced": sorted(forced), "extra": extra, "error": error}
        path = Path(path)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=1))
        os.replace(tmp, path)

    def summary(self) -> str:
        """One line for the context bar."""
        try:
            defaults, forced, extra = self.arguments()
        except ValueError:
            return "free text invalid"
        parts = []
        for dest, tokens in defaults.items():
            if dest in ("working_dir", "pattern", "partition"):
                continue   # shown in the bar itself
            text = " ".join(tokens) if len(" ".join(tokens)) <= 40 else tokens[0] + " …"
            parts.append(text + (" (forced)" if dest in forced else ""))
        if extra:
            parts.append("+ " + " ".join(extra))
        return " · ".join(parts) if parts else "no further options"
