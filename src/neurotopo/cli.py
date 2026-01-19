"""
Command-line interface for NeuroTopo.

Provides commands for processing, evaluation, and experimentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="neurotopo",
    help="AI-assisted retopology: neural guidance + deterministic mesh generation"
)
console = Console()


@app.command()
def process(
    input_path: Path = typer.Argument(..., help="Input mesh file (OBJ, PLY, STL)"),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", help="Output mesh file"),
    target_faces: Optional[int] = typer.Option(None, "-t", "--target-faces", help="Target face count"),
    backend: str = typer.Option("trimesh", "-b", "--backend", help="Remeshing backend"),
    evaluate: bool = typer.Option(True, "--evaluate/--no-evaluate", help="Run quality evaluation"),
    timing: bool = typer.Option(False, "--timing", help="Show detailed timing information"),
    verbose: bool = typer.Option(False, "-v", "--verbose", help="Enable verbose logging"),
    timeout: Optional[float] = typer.Option(None, "--timeout", help="Timeout in seconds (enforces hard limit)"),
):
    """
    Process a mesh through the retopology pipeline.
    """
    from neurotopo.pipeline import RetopoPipeline
    from neurotopo.core.mesh import Mesh
    from neurotopo.utils.timing import get_timing_log, configure_timeouts
    
    # Configure logging if verbose
    if verbose or timing:
        logging.basicConfig(
            level=logging.INFO if not verbose else logging.DEBUG,
            format='%(name)s - %(message)s'
        )
    
    console.print(f"[bold blue]Loading mesh:[/bold blue] {input_path}")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        disable=verbose or timing,  # Disable spinner when logging
    ) as progress:
        task = progress.add_task("Processing...", total=None)
        
        pipeline = RetopoPipeline(
            backend=backend,
            target_faces=target_faces
        )
        
        # Configure timeout if specified
        if timeout:
            configure_timeouts(
                curvature=timeout / 4,
                features=timeout / 4,
                remesh=timeout / 2,
                evaluation=timeout / 4
            )
            pipeline.enforce_timeouts = True
            pipeline.timeout_analysis = timeout / 4
            pipeline.timeout_features = timeout / 4
            pipeline.timeout_remesh = timeout / 2
            pipeline.timeout_evaluation = timeout / 4
        
        output_mesh, score = pipeline.process(
            input_path, 
            evaluate=evaluate,
            enable_timing=timing or verbose
        )
    
    # Show timing if requested
    if timing:
        timing_log = get_timing_log()
        console.print(f"\n[bold cyan]Timing Summary[/bold cyan]")
        for entry in timing_log.entries:
            status = "[green]OK[/green]" if entry.success else "[red]FAIL[/red]"
            console.print(f"  {entry.operation}: {entry.elapsed_seconds:.3f}s {status}")
        console.print(f"  [bold]Total: {timing_log.total_time():.3f}s[/bold]")
    
    # Show results
    if score:
        console.print(f"\n[bold green]Quality Score: {score.overall_score:.1f}/100[/bold green]")
        
        table = Table(title="Quality Breakdown")
        table.add_column("Metric", style="cyan")
        table.add_column("Score", justify="right")
        
        table.add_row("Quad Quality", f"{score.quad_score:.1f}")
        table.add_row("Geometric Fidelity", f"{score.fidelity_score:.1f}")
        table.add_row("Topology", f"{score.topology_score:.1f}")
        
        console.print(table)
    
    console.print(f"\n[bold]Output:[/bold] {output_mesh.num_vertices} vertices, {output_mesh.num_faces} faces")
    
    # Save output
    if output_path:
        output_mesh.to_file(output_path)
        console.print(f"[green]Saved to:[/green] {output_path}")
    else:
        # Default output name
        default_output = input_path.parent / f"{input_path.stem}_retopo.obj"
        output_mesh.to_file(default_output)
        console.print(f"[green]Saved to:[/green] {default_output}")


@app.command()
def evaluate(
    mesh_path: Path = typer.Argument(..., help="Mesh to evaluate"),
    original_path: Optional[Path] = typer.Option(None, "-r", "--reference", help="Reference mesh for comparison"),
):
    """
    Evaluate mesh quality metrics.
    """
    from neurotopo.core.mesh import Mesh
    from neurotopo.evaluation import evaluate_retopology
    
    console.print(f"[bold blue]Evaluating:[/bold blue] {mesh_path}")
    
    mesh = Mesh.from_file(mesh_path)
    original = Mesh.from_file(original_path) if original_path else None
    
    score = evaluate_retopology(mesh, original)
    
    console.print(score.summary())


@app.command()
def experiment(
    config_path: Path = typer.Option(..., "-c", "--config", help="Experiment config YAML"),
    name: Optional[str] = typer.Option(None, "-n", "--name", help="Experiment name override"),
    output_dir: Optional[Path] = typer.Option(None, "-o", "--output", help="Output directory"),
):
    """
    Run an experiment from configuration file.
    """
    from neurotopo.experiments import ExperimentConfig, ExperimentRunner
    
    console.print(f"[bold blue]Loading config:[/bold blue] {config_path}")
    
    config = ExperimentConfig.load(config_path)
    
    if name:
        config.name = name
    if output_dir:
        config.output_dir = str(output_dir)
    
    console.print(f"[bold]Running experiment:[/bold] {config.name}")
    
    runner = ExperimentRunner(config)
    results = runner.run_all_inputs()
    
    # Show summary
    console.print(f"\n[bold green]Completed {len(results)} runs[/bold green]")
    
    table = Table(title="Results Summary")
    table.add_column("Mesh", style="cyan")
    table.add_column("Score", justify="right")
    table.add_column("Faces", justify="right")
    table.add_column("Time", justify="right")
    
    for result in results:
        score_str = f"{result.score.overall_score:.1f}" if result.score else "N/A"
        faces_str = str(result.output_mesh.num_faces) if result.output_mesh else "N/A"
        table.add_row(
            result.input_mesh_name,
            score_str,
            faces_str,
            f"{result.total_time:.2f}s"
        )
    
    console.print(table)
    
    # Save results
    runner.save_results()
    console.print(f"\n[green]Results saved to:[/green] {config.output_dir}")


@app.command()
def iterate(
    input_path: Path = typer.Argument(..., help="Input mesh file"),
    output_path: Optional[Path] = typer.Option(None, "-o", "--output", help="Output mesh file"),
    target_score: float = typer.Option(80.0, "-s", "--score", help="Target quality score"),
    max_iterations: int = typer.Option(5, "-i", "--iterations", help="Maximum iterations"),
    target_faces: Optional[int] = typer.Option(None, "-t", "--target-faces", help="Target face count"),
):
    """
    Iteratively improve retopology until target score is reached.
    """
    from neurotopo.pipeline import RetopoPipeline
    
    console.print(f"[bold blue]Starting iterative retopology[/bold blue]")
    console.print(f"Target score: {target_score}, Max iterations: {max_iterations}")
    
    pipeline = RetopoPipeline(target_faces=target_faces)
    
    best_mesh, best_score = pipeline.iterate(
        input_path,
        target_score=target_score,
        max_iterations=max_iterations,
        target_faces=target_faces
    )
    
    console.print(f"\n[bold green]Best Score: {best_score.overall_score:.1f}/100[/bold green]")
    console.print(best_score.summary())
    
    # Save output
    if output_path:
        best_mesh.to_file(output_path)
        console.print(f"\n[green]Saved to:[/green] {output_path}")


@app.command()
def compare(
    experiment1: str = typer.Argument(..., help="First experiment directory"),
    experiment2: str = typer.Argument(..., help="Second experiment directory"),
):
    """
    Compare results from two experiment runs.
    """
    import json
    
    exp1_path = Path(experiment1) / "experiment_log.json"
    exp2_path = Path(experiment2) / "experiment_log.json"
    
    with open(exp1_path) as f:
        results1 = json.load(f)
    with open(exp2_path) as f:
        results2 = json.load(f)
    
    table = Table(title="Experiment Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column(Path(experiment1).name, justify="right")
    table.add_column(Path(experiment2).name, justify="right")
    table.add_column("Diff", justify="right")
    
    # Compare average scores
    scores1 = [r["score"]["overall_score"] for r in results1 if r["score"]]
    scores2 = [r["score"]["overall_score"] for r in results2 if r["score"]]
    
    avg1 = sum(scores1) / len(scores1) if scores1 else 0
    avg2 = sum(scores2) / len(scores2) if scores2 else 0
    
    diff = avg2 - avg1
    diff_style = "green" if diff > 0 else "red" if diff < 0 else "white"
    
    table.add_row(
        "Avg Overall Score",
        f"{avg1:.1f}",
        f"{avg2:.1f}",
        f"[{diff_style}]{diff:+.1f}[/{diff_style}]"
    )
    
    # More comparisons...
    quad1 = sum(r["score"]["quad_score"] for r in results1 if r["score"]) / len(scores1) if scores1 else 0
    quad2 = sum(r["score"]["quad_score"] for r in results2 if r["score"]) / len(scores2) if scores2 else 0
    diff_q = quad2 - quad1
    diff_style = "green" if diff_q > 0 else "red" if diff_q < 0 else "white"
    
    table.add_row(
        "Avg Quad Score",
        f"{quad1:.1f}",
        f"{quad2:.1f}",
        f"[{diff_style}]{diff_q:+.1f}[/{diff_style}]"
    )
    
    console.print(table)


@app.command()
def gen_test_meshes(
    output_dir: Path = typer.Option("test_meshes", "-o", "--output", help="Output directory"),
):
    """
    Generate test meshes for experimentation.
    """
    from neurotopo.test_meshes import save_test_meshes
    
    console.print(f"[bold blue]Generating test meshes to:[/bold blue] {output_dir}")
    save_test_meshes(str(output_dir))
    console.print("[green]Done![/green]")


@app.command()
def gen_config(
    output_path: Path = typer.Option("config.yaml", "-o", "--output", help="Output config file"),
    name: str = typer.Option("my_experiment", "-n", "--name", help="Experiment name"),
):
    """
    Generate a default experiment configuration file.
    """
    from neurotopo.experiments import create_default_config
    
    config = create_default_config()
    config.name = name
    config.save(output_path)
    
    console.print(f"[green]Config saved to:[/green] {output_path}")


@app.command()
def info(
    mesh_path: Path = typer.Argument(..., help="Mesh file to inspect"),
):
    """
    Show information about a mesh file.
    """
    from neurotopo.core.mesh import Mesh
    
    mesh = Mesh.from_file(mesh_path)
    
    table = Table(title=f"Mesh Info: {mesh_path.name}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Name", mesh.name)
    table.add_row("Vertices", str(mesh.num_vertices))
    table.add_row("Faces", str(mesh.num_faces))
    table.add_row("Type", "Quads" if mesh.is_quad else "Triangles")
    
    min_b, max_b = mesh.bounds
    table.add_row("Bounding Box Min", f"({min_b[0]:.3f}, {min_b[1]:.3f}, {min_b[2]:.3f})")
    table.add_row("Bounding Box Max", f"({max_b[0]:.3f}, {max_b[1]:.3f}, {max_b[2]:.3f})")
    table.add_row("Diagonal", f"{mesh.diagonal:.4f}")
    
    console.print(table)


if __name__ == "__main__":
    app()
