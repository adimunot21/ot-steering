"""Base class for pydantic-validated configs.

Every YAML config in this project is loaded into a subclass of
:class:`BaseConfig`. The base class is intentionally empty — it exists so
downstream configs share a common ancestor (useful for type-narrowing,
registries, and uniform model-config settings like ``extra="forbid"``, which
makes typos in YAML fail loudly rather than silently).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Root config class. Subclass and add fields.

    All subclasses inherit the following pydantic settings:

    - ``extra="forbid"`` — unknown keys in the YAML raise ``ValidationError``,
      catching typos like ``learning_rete`` immediately.
    - ``frozen=True`` — instances are immutable after construction, so a config
      passed around the codebase cannot be silently mutated by a downstream
      function.
    - ``validate_assignment=True`` — even if a subclass overrides ``frozen``,
      any assignment is type-checked.
    - ``populate_by_name=True`` — fields with aliases can be set by either name.
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )
