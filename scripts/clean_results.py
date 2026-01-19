#!/usr/bin/env python3
"""
Clean old test results from the results directory.

Usage:
    python scripts/clean_results.py                    # Interactive mode
    python scripts/clean_results.py --keep 5          # Keep last 5 runs
    python scripts/clean_results.py --older-than 7    # Delete runs older than 7 days
    python scripts/clean_results.py --all             # Delete all results
    python scripts/clean_results.py --dry-run         # Show what would be deleted
"""
import argparse
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path


def get_results_dir() -> Path:
    """Get the results directory path."""
    return Path(__file__).parent.parent / "results"


def parse_timestamp(dirname: str) -> datetime | None:
    """Parse timestamp from directory name (YYYYMMDD_HHMMSS format)."""
    try:
        return datetime.strptime(dirname, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def get_result_runs(results_dir: Path) -> list[tuple[Path, datetime]]:
    """Get all result run directories with their timestamps."""
    runs = []
    if not results_dir.exists():
        return runs
    
    for d in results_dir.iterdir():
        if d.is_dir():
            ts = parse_timestamp(d.name)
            if ts:
                runs.append((d, ts))
    
    # Sort by timestamp, newest first
    runs.sort(key=lambda x: x[1], reverse=True)
    return runs


def get_dir_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total = 0
    for f in path.rglob('*'):
        if f.is_file():
            total += f.stat().st_size
    return total


def format_size(size_bytes: int) -> str:
    """Format size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def format_age(ts: datetime) -> str:
    """Format age of a timestamp."""
    age = datetime.now() - ts
    if age.days > 0:
        return f"{age.days} days ago"
    elif age.seconds >= 3600:
        return f"{age.seconds // 3600} hours ago"
    elif age.seconds >= 60:
        return f"{age.seconds // 60} minutes ago"
    else:
        return "just now"


def list_runs(results_dir: Path):
    """List all result runs with details."""
    runs = get_result_runs(results_dir)
    
    if not runs:
        print("No result runs found.")
        return
    
    print(f"\nFound {len(runs)} result run(s) in {results_dir}:\n")
    print(f"{'#':<3} {'Timestamp':<20} {'Age':<15} {'Size':<10} {'Path'}")
    print("-" * 70)
    
    total_size = 0
    for i, (path, ts) in enumerate(runs, 1):
        size = get_dir_size(path)
        total_size += size
        print(f"{i:<3} {ts.strftime('%Y-%m-%d %H:%M:%S'):<20} {format_age(ts):<15} {format_size(size):<10} {path.name}")
    
    print("-" * 70)
    print(f"Total: {format_size(total_size)}")


def delete_runs(
    runs: list[tuple[Path, datetime]], 
    dry_run: bool = False
) -> tuple[int, int]:
    """Delete specified runs. Returns (count, bytes_freed)."""
    count = 0
    bytes_freed = 0
    
    for path, ts in runs:
        size = get_dir_size(path)
        if dry_run:
            print(f"  Would delete: {path.name} ({format_size(size)})")
        else:
            shutil.rmtree(path)
            print(f"  Deleted: {path.name} ({format_size(size)})")
        count += 1
        bytes_freed += size
    
    return count, bytes_freed


def clean_keep_n(results_dir: Path, keep: int, dry_run: bool = False):
    """Keep only the N most recent runs."""
    runs = get_result_runs(results_dir)
    
    if len(runs) <= keep:
        print(f"Only {len(runs)} run(s) found, nothing to delete (keeping {keep}).")
        return
    
    to_delete = runs[keep:]
    
    print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(to_delete)} old run(s), keeping {keep} most recent:")
    count, freed = delete_runs(to_delete, dry_run)
    
    if not dry_run:
        print(f"\nDeleted {count} run(s), freed {format_size(freed)}")


def clean_older_than(results_dir: Path, days: int, dry_run: bool = False):
    """Delete runs older than N days."""
    runs = get_result_runs(results_dir)
    cutoff = datetime.now() - timedelta(days=days)
    
    to_delete = [(p, ts) for p, ts in runs if ts < cutoff]
    
    if not to_delete:
        print(f"No runs older than {days} days found.")
        return
    
    print(f"\n{'Would delete' if dry_run else 'Deleting'} {len(to_delete)} run(s) older than {days} days:")
    count, freed = delete_runs(to_delete, dry_run)
    
    if not dry_run:
        print(f"\nDeleted {count} run(s), freed {format_size(freed)}")


def clean_all(results_dir: Path, dry_run: bool = False):
    """Delete all result runs."""
    runs = get_result_runs(results_dir)
    
    if not runs:
        print("No runs to delete.")
        return
    
    print(f"\n{'Would delete' if dry_run else 'Deleting'} ALL {len(runs)} run(s):")
    count, freed = delete_runs(runs, dry_run)
    
    if not dry_run:
        print(f"\nDeleted {count} run(s), freed {format_size(freed)}")


def interactive_clean(results_dir: Path):
    """Interactive cleaning mode."""
    runs = get_result_runs(results_dir)
    
    if not runs:
        print("No result runs found.")
        return
    
    list_runs(results_dir)
    
    print("\nOptions:")
    print("  1. Keep last N runs")
    print("  2. Delete runs older than N days")
    print("  3. Delete all")
    print("  4. Delete specific runs")
    print("  5. Cancel")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        n = input("Number of runs to keep: ").strip()
        try:
            clean_keep_n(results_dir, int(n))
        except ValueError:
            print("Invalid number")
    elif choice == "2":
        days = input("Delete runs older than (days): ").strip()
        try:
            clean_older_than(results_dir, int(days))
        except ValueError:
            print("Invalid number")
    elif choice == "3":
        confirm = input("Delete ALL results? (yes/no): ").strip().lower()
        if confirm == "yes":
            clean_all(results_dir)
        else:
            print("Cancelled")
    elif choice == "4":
        indices = input("Enter run numbers to delete (comma-separated): ").strip()
        try:
            to_delete = []
            for idx in indices.split(","):
                i = int(idx.strip()) - 1
                if 0 <= i < len(runs):
                    to_delete.append(runs[i])
            if to_delete:
                print(f"\nDeleting {len(to_delete)} run(s):")
                delete_runs(to_delete)
            else:
                print("No valid runs selected")
        except ValueError:
            print("Invalid input")
    else:
        print("Cancelled")


def main():
    parser = argparse.ArgumentParser(
        description="Clean old test results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/clean_results.py                    # Interactive mode
    python scripts/clean_results.py --list            # List all runs
    python scripts/clean_results.py --keep 5          # Keep last 5 runs
    python scripts/clean_results.py --older-than 7    # Delete runs older than 7 days
    python scripts/clean_results.py --all             # Delete all results
    python scripts/clean_results.py --keep 3 --dry-run  # Preview deletion
        """
    )
    parser.add_argument(
        "--keep", "-k",
        type=int,
        help="Keep only the N most recent runs"
    )
    parser.add_argument(
        "--older-than", "-o",
        type=int,
        dest="older_than",
        help="Delete runs older than N days"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Delete all result runs"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List all runs without deleting"
    )
    parser.add_argument(
        "--dry-run", "-n",
        action="store_true",
        dest="dry_run",
        help="Show what would be deleted without actually deleting"
    )
    
    args = parser.parse_args()
    results_dir = get_results_dir()
    
    if args.list:
        list_runs(results_dir)
    elif args.keep is not None:
        clean_keep_n(results_dir, args.keep, args.dry_run)
    elif args.older_than is not None:
        clean_older_than(results_dir, args.older_than, args.dry_run)
    elif args.all:
        if args.dry_run:
            clean_all(results_dir, dry_run=True)
        else:
            confirm = input("Delete ALL results? (yes/no): ").strip().lower()
            if confirm == "yes":
                clean_all(results_dir)
            else:
                print("Cancelled")
    else:
        # Interactive mode
        interactive_clean(results_dir)


if __name__ == "__main__":
    main()
