import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN_SOURCE = (ROOT / "app" / "main.py").read_text()
MAIN_TREE = ast.parse(MAIN_SOURCE)
README = (ROOT / "README.md").read_text()


def _find_call(name):
    for node in ast.walk(MAIN_TREE):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == name:
                return node
    raise AssertionError(f"Could not find call to {name}")


def test_default_tile_constant_is_512():
    for node in MAIN_TREE.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEFAULT_TILE":
                    assert isinstance(node.value, ast.Constant)
                    assert node.value.value == 512
                    return
    raise AssertionError("DEFAULT_TILE constant not found")


def test_upsampler_is_constructed_with_default_tile():
    call = _find_call("RealESRGANer")
    keywords = {keyword.arg: keyword.value for keyword in call.keywords}

    tile = keywords.get("tile")
    assert isinstance(tile, ast.Name)
    assert tile.id == "DEFAULT_TILE"

    tile_pad = keywords.get("tile_pad")
    assert isinstance(tile_pad, ast.Constant)
    assert tile_pad.value == 10

    pre_pad = keywords.get("pre_pad")
    assert isinstance(pre_pad, ast.Constant)
    assert pre_pad.value == 0


def test_endpoint_exposes_validated_tile_query_parameter():
    endpoint = next(
        node
        for node in MAIN_TREE.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "upscale_image"
    )
    args_by_name = dict(zip([arg.arg for arg in endpoint.args.args], endpoint.args.defaults))

    tile_default = args_by_name["tile"]
    assert isinstance(tile_default, ast.Call)
    assert isinstance(tile_default.func, ast.Name)
    assert tile_default.func.id == "Query"

    default_arg = tile_default.args[0]
    assert isinstance(default_arg, ast.Name)
    assert default_arg.id == "DEFAULT_TILE"

    keywords = {keyword.arg: keyword.value for keyword in tile_default.keywords}
    ge = keywords.get("ge")
    assert isinstance(ge, ast.Constant)
    assert ge.value == 0


def test_endpoint_serializes_per_request_tile_mutation():
    assert "upsampler_lock = asyncio.Lock()" in MAIN_SOURCE
    assert "async with upsampler_lock:" in MAIN_SOURCE
    assert "previous_tile_size = upsampler.tile_size" in MAIN_SOURCE
    assert "upsampler.tile_size = tile" in MAIN_SOURCE
    assert "upsampler.tile_size = previous_tile_size" in MAIN_SOURCE
    assert "upsampler.tile =" not in MAIN_SOURCE


def test_endpoint_mutates_realesrganer_tile_size_contract_in_finally():
    endpoint = next(
        node
        for node in MAIN_TREE.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "upscale_image"
    )

    tile_size_assignments = []
    restore_in_finally = False
    legacy_tile_assignments = []

    for node in ast.walk(endpoint):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "upsampler"
                ):
                    if target.attr == "tile_size":
                        tile_size_assignments.append(node)
                    if target.attr == "tile":
                        legacy_tile_assignments.append(node)

        if isinstance(node, ast.Try):
            for finalizer_node in node.finalbody:
                if not isinstance(finalizer_node, ast.Assign):
                    continue
                for target in finalizer_node.targets:
                    if (
                        isinstance(target, ast.Attribute)
                        and isinstance(target.value, ast.Name)
                        and target.value.id == "upsampler"
                        and target.attr == "tile_size"
                        and isinstance(finalizer_node.value, ast.Name)
                        and finalizer_node.value.id == "previous_tile_size"
                    ):
                        restore_in_finally = True

    assert not legacy_tile_assignments
    assert any(
        isinstance(node.value, ast.Name) and node.value.id == "tile"
        for node in tile_size_assignments
    )
    assert restore_in_finally


def test_readme_documents_query_parameters_and_tiling_guidance():
    assert "Optional Query Parameters" in README
    assert "outscale" in README
    assert "tile" in README
    assert "http://localhost:8000/upscale/?outscale=3.75&tile=512" in README
    assert "default is **512**" in README.lower()
    assert "256" in README
    assert "tile=0" in README
    assert "out-of-memory" in README
