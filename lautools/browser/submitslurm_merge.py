"""submitslurm in the Pipeline tab shell: merge the call with the submission context, run submitslurm.

Called by the shell function submitslurm() (see submitslurm_context.wrapper_bash) as
    LAUTOOLS_SUBMITSLURM_CONTEXT=<json> python submitslurm_merge.py ARGS...
The JSON holds defaults/forced/extra for laupy.submitslurm_options.merge_arguments().
Prints the resulting call and replaces itself with the real submitslurm (found on PATH).
If the context cannot be applied, the call runs unchanged after a warning.
"""
import json
import os
import shlex
import sys


def merged_arguments(argv, context_file):
    if not context_file:
        return argv
    try:
        with open(context_file) as f:
            context = json.load(f)
        if context.get("error"):
            print(f"lautools: submission context not applied: {context['error']}", file=sys.stderr)
            return argv
        if not context.get("enabled"):
            return argv
        from laupy.submitslurm_options import merge_arguments
        return merge_arguments(argv, context.get("defaults", {}), set(context.get("forced", [])), context.get("extra", []))
    except Exception as e:
        print(f"lautools: submission context not applied: {e}", file=sys.stderr)
        return argv


def main():
    # Not passed on to submitslurm (and so not to the jobs)
    context_file = os.environ.pop("LAUTOOLS_SUBMITSLURM_CONTEXT", "")
    argv = merged_arguments(sys.argv[1:], context_file)
    print("+ submitslurm " + " ".join(shlex.quote(a) for a in argv), file=sys.stderr, flush=True)
    try:
        os.execvp("submitslurm", ["submitslurm"] + argv)
    except OSError as e:
        print(f"lautools: cannot run submitslurm: {e}", file=sys.stderr)
        sys.exit(127)


if __name__ == "__main__":
    main()
