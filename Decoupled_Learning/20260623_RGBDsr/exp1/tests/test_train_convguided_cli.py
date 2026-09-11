"""Source-level CLI contract tests for the convguided training script."""

import ast
import os
from pathlib import Path


SCRIPT_PATH = Path(
    os.environ.get("TRAIN_CONVGUIDED_SCRIPT", Path(__file__).resolve().parents[1] / "train_convguided_base.py")
)
EXPECTED_TRAIN_ROOT = r"D:\nuoqi\halo-free\data_NYUv2\NYUv2-resplit\NYUv2"
EXPECTED_TEST_ROOT = r"D:\nuoqi\halo-free\data_NYUv2\benchmark_test_449"


def argument_calls(tree):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
    ]


def keyword_value(call, name):
    return next(keyword.value for keyword in call.keywords if keyword.arg == name)


def is_name(node, name):
    return isinstance(node, ast.Name) and node.id == name


def is_args_attribute(node, name):
    return (
        isinstance(node, ast.Attribute)
        and node.attr == name
        and is_name(node.value, "args")
    )


def test_parser_accepts_train_and_test_root_options():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    calls_by_option = {
        call.args[0].value: call
        for call in argument_calls(tree)
        if call.args
        and isinstance(call.args[0], ast.Constant)
        and isinstance(call.args[0].value, str)
    }

    for option, default_name in (
        ("--train-root", "DEFAULT_TRAIN_ROOT"),
        ("--test-root", "DEFAULT_TEST_ROOT"),
    ):
        call = calls_by_option[option]
        assert is_name(keyword_value(call, "type"), "Path")
        assert is_name(keyword_value(call, "default"), default_name)


def test_root_defaults_preserve_the_windows_dataset_paths():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    assignments = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    for name, expected_path in (
        ("DEFAULT_TRAIN_ROOT", EXPECTED_TRAIN_ROOT),
        ("DEFAULT_TEST_ROOT", EXPECTED_TEST_ROOT),
    ):
        value = assignments[name]
        assert isinstance(value, ast.Call)
        assert is_name(value.func, "Path")
        assert isinstance(value.args[0], ast.Constant)
        assert value.args[0].value == expected_path


def test_full_image_datasets_use_the_cli_roots():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    datasets = {
        target.id: node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and is_name(node.value.func, "NYUv2FullImageDataset")
        for target in node.targets
        if isinstance(target, ast.Name)
    }

    assert is_args_attribute(datasets["train_dataset"].args[0], "train_root")
    assert is_args_attribute(datasets["validation_dataset"].args[0], "train_root")
    assert is_args_attribute(datasets["test_dataset"].args[0], "test_root")


def test_model_version_tracks_the_requested_scale():
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    model_version = next(
        node.value
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(is_name(target, "model_version") for target in node.targets)
    )

    assert isinstance(model_version, ast.JoinedStr)
    assert len(model_version.values) == 2
    assert isinstance(model_version.values[0], ast.Constant)
    assert model_version.values[0].value == "conv_survey_dgf_x"
    assert isinstance(model_version.values[1], ast.FormattedValue)
    assert is_args_attribute(model_version.values[1].value, "scale")
