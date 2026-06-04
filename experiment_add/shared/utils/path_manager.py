from __future__ import annotations

from pathlib import Path
from typing import Any

from experiment_add.shared.utils.io import ensure_dir, read_yaml


VALID_PIPELINES = frozenset({"mineru", "ppstructure"})
VALID_CONDITIONS = frozenset({"clean", "area_matched_erasure", "structural_probe", "large_area_erasure"})


class PathManager:
    """Resolve canonical experiment_add paths from `configs/base.yaml`."""

    def __init__(
        self,
        config_path: str | Path,
        create_dirs: bool = False,
        qa_pairs_suffix: str | None = None,
    ):
        self.config_path = Path(config_path).resolve()
        self.project_root = self._infer_project_root(self.config_path)
        self.config: dict[str, Any] = read_yaml(self.config_path)
        self._outputs = self.config.get("outputs", {})
        self._logs = self.config.get("logs", {})
        self._data = self.config.get("data", {})
        suffix = (qa_pairs_suffix or "").strip()
        if suffix and not all(ch.isalnum() or ch in "_-" for ch in suffix):
            raise ValueError(
                f"qa_pairs_suffix must be alphanumeric/_/- only, got '{suffix}'"
            )
        self.qa_pairs_suffix = suffix or None
        if create_dirs:
            self.create_canonical_dirs()

    @staticmethod
    def _infer_project_root(config_path: Path) -> Path:
        """Infer repository root from `experiment_add/configs/base.yaml`."""
        if config_path.parent.name == "configs" and config_path.parent.parent.name == "experiment_add":
            return config_path.parent.parent.parent
        return Path.cwd()

    def _resolve(self, value: str | Path) -> Path:
        """Resolve a config path relative to repository root."""
        path = value if isinstance(value, Path) else Path(value)
        return path if path.is_absolute() else self.project_root / path

    def _validate_pipeline(self, pipeline: str) -> str:
        """Validate a parser pipeline name."""
        if pipeline not in VALID_PIPELINES:
            raise ValueError(f"Unsupported pipeline '{pipeline}'. Expected one of {sorted(VALID_PIPELINES)}.")
        return pipeline

    def _validate_condition(self, condition: str) -> str:
        """Validate a perturbation condition name."""
        if condition not in VALID_CONDITIONS:
            raise ValueError(f"Unsupported condition '{condition}'. Expected one of {sorted(VALID_CONDITIONS)}.")
        return condition

    @property
    def page_manifest_500(self) -> Path:
        """Path to the 500-page manifest."""
        return self._resolve(self._data["page_manifest_500"])

    @property
    def page_manifest_debug20(self) -> Path:
        """Path to the debug 20-page manifest."""
        return self._resolve(self._data["page_manifest_debug20"])

    @property
    def shared_output_root(self) -> Path:
        """Root directory for shared outputs."""
        return self._resolve(self._outputs["shared"]["shared_root"])

    @property
    def exp1_output_root(self) -> Path:
        """Root directory for experiment 1 outputs."""
        return self._resolve(self._outputs["exp1_qa"]["exp1_root"])

    @property
    def exp2_output_root(self) -> Path:
        """Root directory for experiment 2 outputs."""
        return self._resolve(self._outputs["exp2_retrieval"]["exp2_root"])

    @property
    def shared_log_root(self) -> Path:
        """Root directory for shared logs."""
        return self._resolve(self._logs["shared"])

    @property
    def exp1_log_root(self) -> Path:
        """Root directory for experiment 1 logs."""
        return self._resolve(self._logs["exp1_qa"])

    @property
    def exp2_log_root(self) -> Path:
        """Root directory for experiment 2 logs."""
        return self._resolve(self._logs["exp2_retrieval"])

    def clean_parse_dir(self, pipeline: str) -> Path:
        """Directory containing clean parse artifacts for a pipeline."""
        pipeline = self._validate_pipeline(pipeline)
        return self._resolve(self._outputs["shared"]["clean_parse"][pipeline])

    def clean_parse_pages_dir(self, pipeline: str) -> Path:
        """Directory containing per-page clean parse JSON files."""
        return self.clean_parse_dir(pipeline) / "pages"

    def clean_parse_merged_path(self, pipeline: str) -> Path:
        """Path to merged clean parse JSONL for a pipeline."""
        pipeline = self._validate_pipeline(pipeline)
        return self.clean_parse_dir(pipeline) / f"merged_clean_{pipeline}.jsonl"

    @property
    def qa_pairs_dir(self) -> Path:
        """Directory containing shared QA pair artifacts.

        When ``qa_pairs_suffix`` is set, returns a sibling directory named
        ``<canonical_name>_<suffix>`` so v2/debug runs do not overwrite the
        canonical ``qa_pairs/`` artifacts consumed by exp1/exp2.
        """
        canonical = self._resolve(self._outputs["shared"]["qa_pairs_root"])
        if self.qa_pairs_suffix:
            return canonical.parent / f"{canonical.name}_{self.qa_pairs_suffix}"
        return canonical

    @property
    def qa_candidates_raw_path(self) -> Path:
        """Path to unfiltered shared QA candidates.

        Honors ``qa_pairs_suffix`` so v2/debug candidates do not overwrite
        the canonical file.
        """
        if self.qa_pairs_suffix:
            return self.qa_pairs_dir / "qa_candidates_raw.jsonl"
        return self._resolve(self._outputs["shared"]["qa_candidates_raw"])

    @property
    def qa_pairs_filtered_path(self) -> Path:
        """Path to filtered shared QA pairs before final export."""
        return self.qa_pairs_dir / "qa_pairs_filtered.jsonl"

    @property
    def qa_pairs_shared_path(self) -> Path:
        """Path to canonical shared QA pairs consumed by exp1 and exp2.

        Honors ``qa_pairs_suffix`` so v2/debug shared QA does not overwrite
        the canonical file consumed by downstream stages.
        """
        if self.qa_pairs_suffix:
            return self.qa_pairs_dir / "qa_pairs_shared.jsonl"
        return self._resolve(self._outputs["shared"]["qa_pairs_shared"])

    @property
    def qa_filter_report_path(self) -> Path:
        """Path to the shared QA filtering report."""
        return self.qa_pairs_dir / "qa_filter_report.json"

    def perturbed_pages_dir(self, condition: str) -> Path:
        """Directory for perturbed page artifacts for a condition."""
        condition = self._validate_condition(condition)
        if condition == "clean":
            return self.shared_output_root / "clean_pages"
        return self._resolve(self._outputs["shared"]["perturbed_pages"][condition])

    def perturbed_images_dir(self, condition: str) -> Path:
        """Directory for perturbed image files for a condition."""
        return self.perturbed_pages_dir(condition) / "images"

    def perturb_metadata_path(self, condition: str) -> Path:
        """Path to per-condition perturbation metadata JSONL."""
        return self.perturbed_pages_dir(condition) / "perturb_metadata.jsonl"

    @property
    def merged_perturb_metadata_path(self) -> Path:
        """Path to merged perturbation metadata across conditions."""
        return self._resolve(self._outputs["shared"]["perturbed_pages_root"]) / "perturb_metadata_merged.jsonl"

    @property
    def perturb_summary_path(self) -> Path:
        """Path to perturbation summary JSON."""
        return self._resolve(self._outputs["shared"]["perturbed_pages_root"]) / "perturb_summary.json"

    def perturbed_parse_dir(self, pipeline: str, condition: str) -> Path:
        """Directory containing perturbed parses for a pipeline and condition."""
        pipeline = self._validate_pipeline(pipeline)
        condition = self._validate_condition(condition)
        return self._resolve(self._outputs["shared"]["perturbed_parse"][pipeline]) / condition

    def perturbed_parse_pages_dir(self, pipeline: str, condition: str) -> Path:
        """Directory containing per-page perturbed parse JSON files."""
        return self.perturbed_parse_dir(pipeline, condition) / "pages"

    def perturbed_parse_merged_path(self, pipeline: str, condition: str) -> Path:
        """Path to merged perturbed parse JSONL."""
        return self.perturbed_parse_dir(pipeline, condition) / "perturbed_parse_merged.jsonl"

    @property
    def parser_metrics_dir(self) -> Path:
        """Directory containing shared parser metrics."""
        return self._resolve(self._outputs["shared"]["parser_metrics_root"])

    @property
    def merged_parser_metrics_path(self) -> Path:
        """Path to merged parser metrics JSONL."""
        return self.parser_metrics_dir / "parser_metrics_merged.jsonl"

    def exp1_qa_runs_dir(self, pipeline: str, condition: str) -> Path:
        """Directory containing exp1 QA run artifacts for a pipeline and condition."""
        pipeline = self._validate_pipeline(pipeline)
        condition = self._validate_condition(condition)
        return self.exp1_output_root / "qa_runs" / pipeline / condition

    def exp1_answers_path(self, pipeline: str, condition: str) -> Path:
        """Path to exp1 answer JSONL for a pipeline and condition."""
        return self.exp1_qa_runs_dir(pipeline, condition) / "answers.jsonl"

    @property
    def exp1_metrics_dir(self) -> Path:
        """Directory containing exp1 metrics tables."""
        return self._resolve(self._outputs["exp1_qa"]["metrics_root"])

    @property
    def exp1_summary_table_path(self) -> Path:
        """Path to exp1 summary metrics table."""
        return self.exp1_metrics_dir / "qa_summary_table.tex"

    @property
    def exp1_non_overlap_table_path(self) -> Path:
        """Path to exp1 non-overlap subset table."""
        return self.exp1_metrics_dir / "exp1_non_overlap_table.csv"

    @property
    def exp1_correlation_table_path(self) -> Path:
        """Path to exp1 correlation table."""
        return self.exp1_metrics_dir / "exp1_correlation_table.csv"

    def create_canonical_dirs(self) -> None:
        """Create all canonical directories needed by shared and exp1 tooling.

        When a ``qa_pairs_suffix`` is configured, ``self.qa_pairs_dir``
        already points at the suffixed sibling directory, so this also
        ensures that suffixed directory exists without ever creating or
        touching the canonical ``qa_pairs/`` directory beyond its existing
        state.
        """
        dirs = [
            self.shared_output_root,
            self.exp1_output_root,
            self.exp2_output_root,
            self.shared_log_root,
            self.exp1_log_root,
            self.exp2_log_root,
            self.qa_pairs_dir,
            self.parser_metrics_dir,
            self.exp1_metrics_dir,
        ]
        for pipeline in VALID_PIPELINES:
            dirs.extend([
                self.clean_parse_dir(pipeline),
                self.clean_parse_pages_dir(pipeline),
            ])
            for condition in VALID_CONDITIONS:
                dirs.extend([
                    self.perturbed_parse_dir(pipeline, condition),
                    self.perturbed_parse_pages_dir(pipeline, condition),
                    self.exp1_qa_runs_dir(pipeline, condition),
                ])
        for condition in VALID_CONDITIONS:
            dirs.extend([
                self.perturbed_pages_dir(condition),
                self.perturbed_images_dir(condition),
            ])
        for directory in dirs:
            ensure_dir(directory)
