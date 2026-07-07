# Shadows the rdkit wheel's bundled stubs, which contain a syntax error
# (rdkit-stubs/Chem/rdchem.pyi) that aborts mypy. rdkit is a weak-stub library;
# expose everything as Any.
from typing import Any

def __getattr__(name: str) -> Any: ...
