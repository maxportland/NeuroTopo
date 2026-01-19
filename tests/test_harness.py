#!/usr/bin/env python3
"""
Reusable test harness for MeshRepair - JSON config driven.

Usage:
    python tests/test_harness.py                    # Run 'quick' suite
    python tests/test_harness.py --suite full      # Run 'full' suite
    python tests/test_harness.py --config my.json  # Use custom config
    python tests/test_harness.py --list            # List available suites
    
Artifacts are saved to results/<timestamp>/ folder.
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from meshretopo.core.mesh import Mesh
from meshretopo.analysis.semantic import SemanticAnalyzer, SemanticSegmentation
from meshretopo.pipeline import RetopoPipeline
from meshretopo.evaluation.metrics import MeshEvaluator
from meshretopo.utils.keychain import ensure_api_key


@dataclass
class TestResult:
    """Result of a single test."""
    name: str
    passed: bool
    duration: float
    message: str = ""
    data: dict = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "passed": self.passed,
            "duration": self.duration,
            "message": self.message,
            "data": self.data
        }


class TestHarness:
    """JSON-config driven test harness for MeshRepair."""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path or self._default_config_path()
        self.config = self._load_config()
        self.results: list[TestResult] = []
        self.logger = self._setup_logging()
        
        # Set up results directory
        self.results_dir = self._setup_results_dir()
        
        # Set up API key
        self._setup_api_key()
    
    def _default_config_path(self) -> str:
        """Get default config path."""
        return str(Path(__file__).parent / "test_config.json")
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        with open(self.config_path, "r") as f:
            return json.load(f)
    
    def _setup_results_dir(self) -> Path:
        """Set up results directory with timestamp."""
        base_dir = Path(__file__).parent.parent / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = base_dir / timestamp
        results_dir.mkdir(parents=True, exist_ok=True)
        return results_dir
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging based on config."""
        level = getattr(logging, self.config.get("output", {}).get("log_level", "INFO"))
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        return logging.getLogger("TestHarness")
    
    def _setup_api_key(self):
        """Set up API key from keychain or environment."""
        provider = self.config.get("defaults", {}).get("api_provider", "openai")
        try:
            api_key = ensure_api_key(provider)
            # Set in environment for downstream use
            if provider == "openai":
                os.environ["OPENAI_API_KEY"] = api_key
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = api_key
            self.logger.info(f"API key loaded for {provider}")
        except ValueError as e:
            self.logger.error(str(e))
            sys.exit(1)
    
    def _get_mesh_config(self, mesh_name: str) -> dict:
        """Get configuration for a specific mesh."""
        for mesh in self.config.get("test_meshes", []):
            if mesh["name"] == mesh_name:
                return mesh
        raise ValueError(f"Unknown mesh: {mesh_name}")
    
    def _merge_settings(self, suite_settings: dict) -> dict:
        """Merge suite settings with defaults."""
        defaults = self.config.get("defaults", {}).copy()
        defaults.update(suite_settings)
        return defaults
    
    def _save_mesh(self, mesh: Mesh, name: str) -> Path:
        """Save a mesh to the results directory."""
        mesh_path = self.results_dir / f"{name}.obj"
        mesh.to_file(str(mesh_path))
        self.logger.info(f"Saved mesh: {mesh_path.name}")
        return mesh_path
    
    def _save_image(self, image_data, name: str) -> Path:
        """Save an image to the results directory."""
        import numpy as np
        from PIL import Image
        
        image_path = self.results_dir / f"{name}.png"
        if isinstance(image_data, np.ndarray):
            Image.fromarray(image_data).save(image_path)
        else:
            # Assume it's already a PIL Image or path
            if hasattr(image_data, 'save'):
                image_data.save(image_path)
            else:
                shutil.copy(image_data, image_path)
        self.logger.info(f"Saved image: {image_path.name}")
        return image_path
    
    def _save_json(self, data: dict, name: str) -> Path:
        """Save JSON data to the results directory."""
        json_path = self.results_dir / f"{name}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.logger.info(f"Saved JSON: {json_path.name}")
        return json_path
    
    def _save_report(self, suite_name: str, start_time: float):
        """Save final test report."""
        end_time = time.time()
        
        report = {
            "suite": suite_name,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": end_time - start_time,
            "results_dir": str(self.results_dir),
            "summary": {
                "total": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
            "tests": [r.to_dict() for r in self.results],
            "config": {
                "config_path": self.config_path,
                "version": self.config.get("version", "unknown"),
            }
        }
        
        self._save_json(report, "test_report")
        return report
    
    def list_suites(self):
        """List available test suites."""
        print("\nAvailable test suites:")
        print("-" * 60)
        for name, suite in self.config.get("test_suites", {}).items():
            print(f"  {name:15} - {suite.get('description', 'No description')}")
            print(f"                   Tests: {', '.join(suite.get('tests', []))}")
        print()
    
    def run_suite(self, suite_name: str) -> bool:
        """Run a test suite by name."""
        suite_start_time = time.time()
        
        suites = self.config.get("test_suites", {})
        if suite_name not in suites:
            self.logger.error(f"Unknown suite: {suite_name}")
            self.list_suites()
            return False
        
        suite = suites[suite_name]
        settings = self._merge_settings(suite.get("settings", {}))
        
        self.logger.info("=" * 60)
        self.logger.info(f"Running test suite: {suite_name}")
        self.logger.info(f"Description: {suite.get('description', 'N/A')}")
        self.logger.info(f"Results dir: {self.results_dir}")
        self.logger.info("=" * 60)
        
        all_passed = True
        
        for mesh_name in suite.get("meshes", []):
            mesh_config = self._get_mesh_config(mesh_name)
            
            self.logger.info(f"\nTesting mesh: {mesh_name}")
            self.logger.info(f"  Path: {mesh_config['path']}")
            
            # Load mesh
            mesh_path = Path(self.config_path).parent.parent / mesh_config["path"]
            if not mesh_path.exists():
                self.results.append(TestResult(
                    name=f"{mesh_name}/load",
                    passed=False,
                    duration=0,
                    message=f"Mesh file not found: {mesh_path}"
                ))
                all_passed = False
                continue
            
            mesh = Mesh.from_file(str(mesh_path))
            self.logger.info(f"  Loaded: {mesh.num_vertices} verts, {mesh.num_faces} faces")
            
            # Store current mesh name for artifact saving
            self._current_mesh_name = mesh_name
            
            # Run tests for this mesh
            for test_name in suite.get("tests", []):
                result = self._run_test(test_name, mesh, mesh_config, settings)
                self.results.append(result)
                
                status = "✓ PASS" if result.passed else "✗ FAIL"
                self.logger.info(f"  {status}: {test_name} ({result.duration:.2f}s)")
                if result.message:
                    self.logger.info(f"         {result.message}")
                
                if not result.passed:
                    all_passed = False
        
        # Save final report
        self._save_report(suite_name, suite_start_time)
        self._print_summary()
        return all_passed
    
    def _run_test(
        self,
        test_name: str,
        mesh: Mesh,
        mesh_config: dict,
        settings: dict
    ) -> TestResult:
        """Run a single test."""
        start = time.time()
        
        try:
            if test_name == "semantic_analysis":
                return self._test_semantic_analysis(mesh, mesh_config, settings, start)
            elif test_name == "segment_validation":
                return self._test_segment_validation(mesh, mesh_config, settings, start)
            elif test_name == "retopology":
                return self._test_retopology(mesh, mesh_config, settings, start)
            elif test_name == "quality_evaluation":
                return self._test_quality_evaluation(mesh, mesh_config, settings, start)
            elif test_name == "ai_quality_assessment":
                return self._test_ai_quality_assessment(mesh, mesh_config, settings, start)
            else:
                return TestResult(
                    name=test_name,
                    passed=False,
                    duration=time.time() - start,
                    message=f"Unknown test: {test_name}"
                )
        except Exception as e:
            return TestResult(
                name=test_name,
                passed=False,
                duration=time.time() - start,
                message=f"Exception: {e}"
            )
    
    def _test_semantic_analysis(
        self,
        mesh: Mesh,
        mesh_config: dict,
        settings: dict,
        start: float
    ) -> TestResult:
        """Test semantic analysis."""
        analyzer = SemanticAnalyzer(
            api_provider=settings.get("api_provider", "openai"),
            model=settings.get("model"),
            resolution=tuple(settings.get("resolution", [1024, 1024])),
            num_views=settings.get("num_views", 6),
            use_blender=settings.get("use_blender", True),
            blender_path=settings.get("blender_path"),
            use_cache=settings.get("cache_enabled", True),
            cache_dir=settings.get("cache_dir"),
        )
        
        segmentation = analyzer.analyze(mesh)
        duration = time.time() - start
        
        num_segments = len(segmentation.segments)
        segment_types = [s.region_type.value for s in segmentation.segments]
        
        # Store for later tests
        self._last_segmentation = segmentation
        self._last_analyzer = analyzer
        
        if num_segments > 0:
            return TestResult(
                name="semantic_analysis",
                passed=True,
                duration=duration,
                message=f"Detected {num_segments} segments: {segment_types}",
                data={"segments": segment_types, "count": num_segments}
            )
        else:
            return TestResult(
                name="semantic_analysis",
                passed=False,
                duration=duration,
                message="No segments detected"
            )
    
    def _test_segment_validation(
        self,
        mesh: Mesh,
        mesh_config: dict,
        settings: dict,
        start: float
    ) -> TestResult:
        """Validate detected segments against expected regions."""
        if not hasattr(self, "_last_segmentation"):
            return TestResult(
                name="segment_validation",
                passed=False,
                duration=0,
                message="No segmentation available - run semantic_analysis first"
            )
        
        expected = set(mesh_config.get("expected_regions", []))
        detected = set(s.region_type.value for s in self._last_segmentation.segments)
        
        duration = time.time() - start
        
        missing = expected - detected
        
        if not missing:
            return TestResult(
                name="segment_validation",
                passed=True,
                duration=duration,
                message=f"All expected regions found: {expected}",
                data={"expected": list(expected), "detected": list(detected)}
            )
        else:
            return TestResult(
                name="segment_validation",
                passed=False,
                duration=duration,
                message=f"Missing regions: {missing}",
                data={"expected": list(expected), "detected": list(detected), "missing": list(missing)}
            )
    
    def _test_retopology(
        self,
        mesh: Mesh,
        mesh_config: dict,
        settings: dict,
        start: float
    ) -> TestResult:
        """Test retopology pipeline."""
        retopo_config = mesh_config.get("retopology", {})
        target_faces = retopo_config.get("target_faces", 5000)
        backend = retopo_config.get("backend", "hybrid")
        
        pipeline = RetopoPipeline(backend=backend)
        result = pipeline.process(mesh, target_faces=target_faces, evaluate=False)
        
        # Handle both tuple return (mesh, score) and single mesh return
        if isinstance(result, tuple):
            output = result[0]
        else:
            output = result
        
        duration = time.time() - start
        
        # Store for later tests
        self._last_retopo = output
        
        # Save retopo mesh artifact
        mesh_name = getattr(self, '_current_mesh_name', 'mesh')
        self._save_mesh(output, f"{mesh_name}_retopo")
        
        # Check output is valid
        if output.num_faces > 0:
            quad_count = sum(1 for f in output.faces if len(f) == 4)
            quad_pct = quad_count / output.num_faces * 100
            
            return TestResult(
                name="retopology",
                passed=True,
                duration=duration,
                message=f"{mesh.num_faces} → {output.num_faces} faces, {quad_pct:.1f}% quads",
                data={
                    "input_faces": mesh.num_faces,
                    "output_faces": output.num_faces,
                    "quad_percentage": quad_pct
                }
            )
        else:
            return TestResult(
                name="retopology",
                passed=False,
                duration=duration,
                message="Retopology produced empty mesh"
            )
    
    def _test_quality_evaluation(
        self,
        mesh: Mesh,
        mesh_config: dict,
        settings: dict,
        start: float
    ) -> TestResult:
        """Test quality evaluation."""
        if not hasattr(self, "_last_retopo"):
            return TestResult(
                name="quality_evaluation",
                passed=False,
                duration=0,
                message="No retopology output - run retopology first"
            )
        
        evaluator = MeshEvaluator()
        score = evaluator.evaluate(mesh, self._last_retopo)
        
        duration = time.time() - start
        
        thresholds = self.config.get("quality_thresholds", {})
        min_score = thresholds.get("min_overall_score", 40.0)
        
        passed = score.overall_score >= min_score
        
        return TestResult(
            name="quality_evaluation",
            passed=passed,
            duration=duration,
            message=f"Overall: {score.overall_score:.1f}/100 (min: {min_score})",
            data={
                "overall": score.overall_score,
                "quad_quality": score.quad_quality,
                "fidelity": score.fidelity_score,
                "topology": score.topology_score,
                "visual": score.visual_score
            }
        )
    
    def _test_ai_quality_assessment(
        self,
        mesh: Mesh,
        mesh_config: dict,
        settings: dict,
        start: float
    ) -> TestResult:
        """Test AI-powered visual quality assessment."""
        from meshretopo.evaluation.ai_quality import AIQualityAssessor
        from meshretopo.analysis.blender_render import BlenderRenderer
        
        if not hasattr(self, "_last_retopo"):
            return TestResult(
                name="ai_quality_assessment",
                passed=False,
                duration=0,
                message="No retopology output - run retopology first"
            )
        
        # Save renders as artifacts
        mesh_name = getattr(self, '_current_mesh_name', 'mesh')
        renders_dir = self.results_dir / "renders"
        renders_dir.mkdir(exist_ok=True)
        
        try:
            renderer = BlenderRenderer(
                blender_path=settings.get("blender_path"),
                resolution=tuple(settings.get("resolution", [1024, 1024])),
            )
            
            # Save retopo mesh to temp file for rendering
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
                temp_path = f.name
            self._last_retopo.to_file(temp_path)
            
            # Render with wireframe and save
            retopo_renders, meta = renderer.render_mesh(
                temp_path,
                output_dir=str(renders_dir / "retopo"),
                wireframe=True
            )
            
            # Clean up temp file
            os.unlink(temp_path)
            
            self.logger.info(f"Saved {len(retopo_renders)} render images to {renders_dir / 'retopo'}")
        except Exception as e:
            self.logger.warning(f"Failed to save render artifacts: {e}")
        
        assessor = AIQualityAssessor(
            api_provider=settings.get("api_provider", "openai"),
            model=settings.get("model"),
            resolution=tuple(settings.get("resolution", [1024, 1024])),
            num_views=settings.get("num_views", 4),
            use_blender=settings.get("use_blender", True),
            blender_path=settings.get("blender_path"),
            wireframe_mode=settings.get("wireframe_mode", True),
        )
        
        # Get mesh context from config
        context = mesh_config.get("description", None)
        
        report = assessor.assess(
            mesh=self._last_retopo,
            original_mesh=mesh,
            context=context,
        )
        
        duration = time.time() - start
        
        # Store for potential later use
        self._last_ai_report = report
        
        # Save AI report as JSON artifact
        ai_report_data = {
            "overall_score": report.overall_score,
            "animation_ready": report.animation_ready,
            "mesh_type_detected": report.mesh_type_detected,
            "summary": report.summary,
            "strengths": report.strengths,
            "recommended_actions": report.recommended_actions,
            "issues": [
                {
                    "category": issue.category.value,
                    "severity": issue.severity.value,
                    "description": issue.description,
                    "location_hint": issue.location_hint,
                    "recommendation": issue.recommendation,
                }
                for issue in report.issues
            ]
        }
        self._save_json(ai_report_data, f"{mesh_name}_ai_quality_report")
        
        # Determine pass/fail - only fail if score is below threshold
        # Issues are informational (AI can be overly strict with severity)
        thresholds = self.config.get("quality_thresholds", {})
        min_ai_score = thresholds.get("min_ai_quality_score", 50.0)
        
        passed = report.overall_score >= min_ai_score
        
        # Build message
        issue_counts = report.issue_count_by_severity
        issues_str = ", ".join(f"{k}:{v}" for k, v in issue_counts.items() if v > 0)
        
        return TestResult(
            name="ai_quality_assessment",
            passed=passed,
            duration=duration,
            message=f"AI Score: {report.overall_score:.1f}/100, Issues: {issues_str or 'none'}",
            data={
                "overall_score": report.overall_score,
                "animation_ready": report.animation_ready,
                "mesh_type": report.mesh_type_detected,
                "issue_counts": issue_counts,
                "strengths": report.strengths,
                "summary": report.summary,
                "recommendations": report.recommended_actions,
            }
        )
    
    def _print_summary(self):
        """Print test summary."""
        passed = sum(1 for r in self.results if r.passed)
        total = len(self.results)
        
        print("\n" + "=" * 60)
        print(f"  TEST SUMMARY: {passed}/{total} passed")
        print("=" * 60)
        
        for result in self.results:
            status = "✓" if result.passed else "✗"
            print(f"  {status} {result.name}: {result.message}")
        
        print("=" * 60)
        
        total_time = sum(r.duration for r in self.results)
        print(f"  Total time: {total_time:.2f}s")
        print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="MeshRepair Test Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python tests/test_harness.py                    # Run 'quick' suite
    python tests/test_harness.py --suite full      # Run 'full' suite  
    python tests/test_harness.py --config my.json  # Use custom config
    python tests/test_harness.py --list            # List available suites
        """
    )
    parser.add_argument(
        "--suite", "-s",
        default="quick",
        help="Test suite to run (default: quick)"
    )
    parser.add_argument(
        "--config", "-c",
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available test suites"
    )
    
    args = parser.parse_args()
    
    harness = TestHarness(config_path=args.config)
    
    if args.list:
        harness.list_suites()
        return
    
    success = harness.run_suite(args.suite)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
