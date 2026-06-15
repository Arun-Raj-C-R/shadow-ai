"""
Parametric Feature History & Dependency Graph
================================================
Real CAD remembers operations.

Example:  Sketch → Extrude → Fillet → Shell

If sketch changes: everything rebuilds. That's parametric CAD.

Implements:
    - Feature tree (ordered list of operations)
    - Dependency graph (DAG)
    - Recomputation engine
    - Feature freeze/suppress
    - Rollback to any point in history
"""

import time
from .vector_math import Vec3


class Feature:
    """Base class for a parametric feature (operation in the timeline)."""
    _counter = 0

    def __init__(self, name, operation, params=None, dependencies=None):
        Feature._counter += 1
        self.id = Feature._counter
        self.name = name
        self.operation = operation  # Callable: fn(params, deps) → result
        self.params = params or {}
        self.dependencies = dependencies or []
        self.result = None
        self.suppressed = False
        self.frozen = False
        self.timestamp = time.time()
        self.error = None
        self.dirty = True

    def execute(self, dep_results=None):
        """Execute this feature's operation."""
        if self.suppressed:
            return None
        if self.frozen and self.result is not None:
            return self.result
        try:
            self.result = self.operation(self.params, dep_results or {})
            self.error = None
            self.dirty = False
        except Exception as e:
            self.error = str(e)
            self.result = None
        return self.result

    def invalidate(self):
        """Mark feature as needing recomputation."""
        if not self.frozen:
            self.dirty = True

    def __repr__(self):
        status = "✅" if self.result and not self.error else "❌" if self.error else "⏳"
        sup = " [SUPPRESSED]" if self.suppressed else ""
        frz = " [FROZEN]" if self.frozen else ""
        return f"Feature({self.id}: {self.name} {status}{sup}{frz})"


class FeatureTree:
    """
    Parametric feature tree — ordered timeline of CAD operations.

    Maintains a dependency graph (DAG) and supports:
        - Adding/removing features
        - Modifying parameters → cascading rebuild
        - Suppressing/unsuppressing features
        - Rollback to any point
    """

    def __init__(self, name="Part"):
        self.name = name
        self.features = []
        self._id_map = {}

    def add_feature(self, name, operation, params=None, dependencies=None):
        """Add a new feature to the end of the timeline."""
        dep_ids = []
        if dependencies:
            for dep in dependencies:
                if isinstance(dep, Feature):
                    dep_ids.append(dep.id)
                elif isinstance(dep, int):
                    dep_ids.append(dep)

        feature = Feature(name, operation, params, dep_ids)
        self.features.append(feature)
        self._id_map[feature.id] = feature
        return feature

    def get_feature(self, feature_id):
        return self._id_map.get(feature_id)

    def modify_params(self, feature_id, new_params):
        """Modify parameters of a feature and mark downstream as dirty."""
        feature = self._id_map.get(feature_id)
        if not feature:
            raise ValueError(f"Feature {feature_id} not found")
        feature.params.update(new_params)
        feature.invalidate()
        self._invalidate_downstream(feature_id)

    def _invalidate_downstream(self, feature_id):
        """Mark all features that depend on this one as dirty."""
        for f in self.features:
            if feature_id in f.dependencies:
                f.invalidate()
                self._invalidate_downstream(f.id)

    def rebuild(self, verbose=False):
        """
        Recompute all dirty features in dependency order.

        Algorithm:
            1. Topological sort (already in order since features are appended)
            2. For each dirty feature:
                a. Gather dependency results
                b. Execute feature
                c. Mark clean
        """
        results = {}
        rebuilt = 0

        for feature in self.features:
            if feature.suppressed:
                continue

            if feature.dirty or feature.result is None:
                # Gather dependency results
                dep_results = {}
                for dep_id in feature.dependencies:
                    dep_feature = self._id_map.get(dep_id)
                    if dep_feature and dep_feature.result is not None:
                        dep_results[dep_id] = dep_feature.result

                feature.execute(dep_results)
                rebuilt += 1

                if verbose:
                    status = "✅" if not feature.error else f"❌ {feature.error}"
                    print(f"  Rebuilt: {feature.name} → {status}")

            if feature.result is not None:
                results[feature.id] = feature.result

        if verbose:
            print(f"  Rebuild complete: {rebuilt}/{len(self.features)} features recomputed")

        return results

    def suppress(self, feature_id):
        """Suppress a feature (exclude from rebuild)."""
        f = self._id_map.get(feature_id)
        if f:
            f.suppressed = True
            self._invalidate_downstream(feature_id)

    def unsuppress(self, feature_id):
        f = self._id_map.get(feature_id)
        if f:
            f.suppressed = False
            f.invalidate()
            self._invalidate_downstream(feature_id)

    def freeze(self, feature_id):
        """Freeze a feature (result won't change on rebuild)."""
        f = self._id_map.get(feature_id)
        if f:
            f.frozen = True

    def unfreeze(self, feature_id):
        f = self._id_map.get(feature_id)
        if f:
            f.frozen = False
            f.invalidate()

    def rollback_to(self, feature_id):
        """Suppress all features after the given feature."""
        found = False
        for f in self.features:
            if f.id == feature_id:
                found = True
                continue
            if found:
                f.suppressed = True

    def get_final_result(self):
        """Get the result of the last non-suppressed feature."""
        for f in reversed(self.features):
            if not f.suppressed and f.result is not None:
                return f.result
        return None

    def timeline(self):
        """Print the feature timeline."""
        lines = [f"Feature Tree: {self.name}"]
        lines.append("=" * 50)
        for i, f in enumerate(self.features):
            deps = f" ← [{', '.join(str(d) for d in f.dependencies)}]" if f.dependencies else ""
            lines.append(f"  {i+1}. {f}{deps}")
        lines.append("=" * 50)
        return "\n".join(lines)

    def info(self):
        total = len(self.features)
        active = sum(1 for f in self.features if not f.suppressed)
        errors = sum(1 for f in self.features if f.error)
        return f"FeatureTree '{self.name}': {active}/{total} active, {errors} errors"

    def __repr__(self):
        return self.info()
