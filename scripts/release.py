"""Build once, check, and publish the same WASS2S distributions."""
from __future__ import annotations

import argparse
import ast
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
import venv

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
REPOSITORY = "hmandela/WASS2S"


def run(command: list[str], *, cwd: Path = ROOT, capture: bool = False,
        env: dict[str, str] | None = None) -> str:
    print("+ " + shlex.join(map(str, command)), flush=True)
    result = subprocess.run(command, cwd=cwd, check=True, text=True, env=env,
                            stdout=subprocess.PIPE if capture else None)
    return result.stdout.strip() if capture else ""


def version() -> str:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    module_version = None
    module = ast.parse((ROOT / "wass2s/__init__.py").read_text())
    for node in module.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            module_version = ast.literal_eval(node.value)
    value = project["version"]
    if project["name"] != "wass2s" or value != module_version:
        raise RuntimeError("Synchronisez les versions dans pyproject.toml et wass2s/__init__.py.")
    # Pixi is optional: pip-only source trees need no Pixi configuration.
    pixi_path = ROOT / "pixi.toml"
    if pixi_path.exists():
        pixi = tomllib.loads(pixi_path.read_text())["workspace"]
        if value != pixi["version"]:
            raise RuntimeError("Synchronisez les versions dans pyproject.toml, pixi.toml et wass2s/__init__.py.")
    if not re.fullmatch(r"[0-9]+(?:\.[0-9]+)+(?:[a-z0-9.]+)?", value):
        raise RuntimeError(f"Version de publication invalide : {value!r}")
    return value


def archives(value: str) -> list[Path]:
    return [DIST / f"wass2s-{value}-py3-none-any.whl", DIST / f"wass2s-{value}.tar.gz"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest() -> str:
    """Hash release inputs, excluding environments, caches and generated output."""
    ignored = {".git", ".pixi", "dist", "build", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
    files = []
    def excluded(name: str) -> bool:
        return name in ignored or name.startswith((".venv", ".release-")) or name.endswith(".egg-info")
    for directory, subdirectories, names in os.walk(ROOT):
        subdirectories[:] = [name for name in subdirectories if not excluded(name)]
        for name in names:
            path = Path(directory) / name
            if not excluded(name) and path.suffix not in {".pyc", ".pyo", ".nbc", ".nbi"}:
                files.append(path)
    result = hashlib.sha256()
    for path in sorted(files):
        result.update(path.relative_to(ROOT).as_posix().encode())
        result.update(b"\0")
        result.update(path.read_bytes())
        result.update(b"\0")
    return result.hexdigest()


def twine_check(paths: list[Path]) -> None:
    run([sys.executable, "-m", "twine", "check", "--strict", *map(str, paths)])


def smoke_test(wheel: Path) -> None:
    """Install the built wheel separately and resolve its declared dependencies."""
    with tempfile.TemporaryDirectory(prefix="wass2s-wheel-") as folder:
        directory = Path(folder)
        environment = directory / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(environment)
        python = environment / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        dependencies = os.environ.copy()
        if sys.platform.startswith("linux"):
            # Reuse shared libraries from the invoking environment when available.
            library = str(Path(sys.prefix) / "lib")
            previous = dependencies.get("LD_LIBRARY_PATH", "")
            dependencies["LD_LIBRARY_PATH"] = library + (os.pathsep + previous if previous else "")
        run([str(python), "-I", "-m", "pip", "install", "--no-deps", "--force-reinstall", str(wheel)], cwd=directory, env=dependencies)
        # system_site_packages exposes the base Python's packages, not those
        # installed only in an invoking pip venv. Resolve any missing requirements
        # after forcing our wheel to take precedence over an inherited install.
        run([str(python), "-I", "-m", "pip", "install", str(wheel)], cwd=directory, env=dependencies)
        run([str(python), "-I", "-m", "pip", "check"], cwd=directory, env=dependencies)
        code = (
            "from pathlib import Path; import sys, wass2s; "
            "from importlib.metadata import version; "
            "assert Path(wass2s.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), wass2s.__file__; "
            "assert version('wass2s') == wass2s.__version__; "
            "print('Wheel installe :', wass2s.__version__, wass2s.__file__)"
        )
        run([str(python), "-I", "-c", code], cwd=directory, env=dependencies)
        run([str(python), "-I", str(ROOT / "scripts/check_environment.py")], cwd=directory, env=dependencies)


def check(*, skip_tests: bool = False) -> None:
    value = version()
    before = source_digest()
    if skip_tests:
        print("Tests non relancés (--skip-tests).", flush=True)
    else:
        run([sys.executable, "-m", "pytest", "tests"])
    DIST.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wass2s-build-") as folder:
        staging = Path(folder)
        run([sys.executable, "-m", "build", "--outdir", str(staging)])
        staged = [staging / path.name for path in archives(value)]
        if not all(path.is_file() for path in staged):
            raise RuntimeError("La construction n'a pas produit le wheel et l'archive source attendus.")
        twine_check(staged)
        smoke_test(staged[0])
        if source_digest() != before:
            raise RuntimeError("Les sources ont changé pendant le contrôle ; relancez release-check.")
        for source, target in zip(staged, archives(value)):
            shutil.copyfile(source, target)
    checksums = [f"# source-sha256 {before}", f"# tests {'skipped' if skip_tests else 'passed'}"]
    checksums.extend(f"{digest(path)}  {path.name}" for path in archives(value))
    (DIST / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(f"WASS2S {value} : archives construites et vérifiées dans dist/.")
    print("Conservez ces mêmes fichiers pour TestPyPI, PyPI et GitHub.")


def checked_archives() -> tuple[str, list[Path]]:
    value = version()
    paths = archives(value)
    manifest = DIST / "SHA256SUMS"
    if not manifest.is_file() or not all(path.is_file() for path in paths):
        raise RuntimeError("Lancez d'abord python scripts/release.py check.")
    lines = manifest.read_text().splitlines()
    if f"# source-sha256 {source_digest()}" not in lines:
        raise RuntimeError("Les sources ont changé : relancez release-check avant publication.")
    expected = [f"{digest(path)}  {path.name}" for path in paths]
    actual = [line for line in lines if line and not line.startswith("#")]
    if actual != expected:
        raise RuntimeError("Les archives diffèrent des fichiers vérifiés : relancez release-check.")
    twine_check(paths)
    return value, paths


def upload(repository: str, *, dry_run: bool = False) -> None:
    _, paths = checked_archives()
    command = [sys.executable, "-m", "twine", "upload", "--repository", repository,
               "--username", "__token__", *map(str, paths)]
    if dry_run:
        print("Prévisualisation : " + shlex.join(command))
    else:
        run(command)


def github(*, dry_run: bool = False) -> None:
    value, paths = checked_archives()
    tag = f"v{value}"
    if run(["git", "status", "--porcelain", "--untracked-files=normal"], capture=True):
        raise RuntimeError("Enregistrez vos changements dans Git avant la release.")
    origin = run(["git", "remote", "get-url", "origin"], capture=True)
    if origin.removesuffix(".git").removesuffix("/") not in {
        f"https://github.com/{REPOSITORY}", f"git@github.com:{REPOSITORY}",
        f"ssh://git@github.com/{REPOSITORY}",
    }:
        raise RuntimeError(f"Le remote origin doit désigner {REPOSITORY}.")
    head = run(["git", "rev-parse", "HEAD"], capture=True)
    tagged = run(["git", "rev-parse", f"{tag}^{{commit}}"], capture=True)
    if head != tagged:
        raise RuntimeError(f"Le tag {tag} doit désigner le commit courant.")
    remote_tags = run(["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}",
                       f"refs/tags/{tag}^{{}}"], capture=True)
    remote = dict(line.split()[::-1] for line in remote_tags.splitlines())
    remote_commit = remote.get(f"refs/tags/{tag}^{{}}", remote.get(f"refs/tags/{tag}"))
    if remote_commit != head:
        raise RuntimeError(f"Poussez le tag {tag} du commit courant sur origin avant la release.")
    command = ["gh", "release", "create", tag, *map(str, paths), str(DIST / "SHA256SUMS"),
               "--repo", REPOSITORY, "--verify-tag", "--title", f"WASS2S {value}", "--generate-notes"]
    if dry_run:
        print("Prévisualisation : " + shlex.join(command))
    else:
        run(command)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    checking = commands.add_parser("check", help="Tests, construction, contrôle strict et installation du wheel")
    checking.add_argument("--skip-tests", action="store_true", help="Ne pas relancer les tests déjà exécutés")
    uploading = commands.add_parser("upload", help="Publier les archives vérifiées sans reconstruction")
    uploading.add_argument("--repository", choices=["testpypi", "pypi"], required=True)
    uploading.add_argument("--dry-run", action="store_true")
    releasing = commands.add_parser("github", help="Créer la release du tag Git déjà poussé")
    releasing.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "check":
            check(skip_tests=args.skip_tests)
        elif args.command == "upload":
            upload(args.repository, dry_run=args.dry_run)
        else:
            github(dry_run=args.dry_run)
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(f"Publication interrompue : {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
