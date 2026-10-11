"""Load and validate the yml configuration."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "config"


@dataclass(frozen=True)
class FactorSpec:
    name: str
    function: str
    expected_sign: int
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IPS:
    raw: Dict[str, Any]
    factors: List[FactorSpec]

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw[name]

    @property
    def start_date(self) -> date:
        return _as_date(self.raw["run"]["start_date"])

    @property
    def initial_capital(self) -> float:
        return float(self.raw["run"]["initial_capital"])


def _as_date(v) -> date:
    return v if isinstance(v, date) else date.fromisoformat(str(v))


def load_config(ips_path: Optional[Path] = None,
                factors_path: Optional[Path] = None) -> IPS:
    ips_path = Path(ips_path or CONFIG_DIR / "ips.yml")
    factors_path = Path(factors_path or CONFIG_DIR / "factors.yml")
    raw = yaml.safe_load(ips_path.read_text())
    specs = []
    for f in yaml.safe_load(factors_path.read_text())["factors"]:
        if f["expected_sign"] not in (1, -1):
            raise ValueError(f"{f['name']}: expected_sign must be +1 or -1")
        specs.append(FactorSpec(f["name"], f["function"], int(f["expected_sign"]),
                                dict(f.get("params", {}))))
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError("duplicate factor names in factors.yml")
    return IPS(raw=raw, factors=specs)
