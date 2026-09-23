#!/usr/bin/env python3
"""Run the upstream LiveCodeBench grader at an explicit dataset revision."""

import os
import runpy

from datasets import load_dataset as hf_load_dataset
from lcb_runner.benchmarks import code_generation as codegen


def pinned_load_dataset(*args, **kwargs):
    kwargs["revision"] = os.environ["LCB_DATASET_REVISION"]
    return hf_load_dataset(*args, **kwargs)


codegen.load_dataset = pinned_load_dataset
runpy.run_module("lcb_runner.runner.custom_evaluator", run_name="__main__")
