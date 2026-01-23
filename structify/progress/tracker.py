"""Progress tracking with rich progress bars and Jupyter support."""

from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.live import Live
from rich.table import Table


def _is_notebook() -> bool:
    """Check if running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        shell_name = shell.__class__.__name__
        if shell_name == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell_name == 'TerminalInteractiveShell':
            return False  # Terminal IPython
        return False
    except (ImportError, NameError):
        return False


@dataclass
class StageProgress:
    """Track progress for a single stage."""

    name: str
    total: int
    completed: int = 0
    current_item: str = ""
    records_found: int = 0
    errors: int = 0
    status: str = "pending"  # pending, in_progress, completed, error
    substeps: dict[str, Any] = field(default_factory=dict)

    def update(self, completed: int | None = None, current_item: str | None = None):
        """Update progress."""
        if completed is not None:
            self.completed = completed
        if current_item is not None:
            self.current_item = current_item

    def increment(self, records: int = 0):
        """Increment completed count."""
        self.completed += 1
        self.records_found += records

    @property
    def percentage(self) -> float:
        """Get completion percentage."""
        if self.total == 0:
            return 0.0
        return (self.completed / self.total) * 100


class ProgressTracker:
    """
    Progress tracker for pipeline stages.

    Automatically detects Jupyter notebooks and uses a simpler display mode.
    In terminal mode, uses Rich Live display with beautiful progress bars.
    In notebook mode, uses simple print statements for compatibility.
    """

    def __init__(
        self,
        show_eta: bool = True,
        show_speed: bool = True,
        log_to_file: str | None = None,
        simple_mode: bool | None = None,
    ):
        """
        Initialize the progress tracker.

        Args:
            show_eta: Show estimated time remaining
            show_speed: Show processing speed
            log_to_file: Optional file path for progress logs
            simple_mode: Use simple print mode (auto-detects Jupyter if None)
        """
        self.show_eta = show_eta
        self.show_speed = show_speed
        self.log_to_file = log_to_file

        # Auto-detect Jupyter if simple_mode not specified
        if simple_mode is None:
            self.simple_mode = _is_notebook()
        else:
            self.simple_mode = simple_mode

        # Use force_terminal=True in simple mode to avoid Jupyter issues
        self.console = Console(force_terminal=not self.simple_mode)

        self._stages: dict[str, StageProgress] = {}
        self._current_stage: str | None = None
        self._progress: Progress | None = None
        self._live: Live | None = None
        self._task_ids: dict[str, Any] = {}
        self._last_print_percent: int = -1

    def _create_progress(self) -> Progress:
        """Create a rich progress bar."""
        columns = [
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
        ]

        if self.show_eta:
            columns.append(TimeRemainingColumn())

        columns.append(TimeElapsedColumn())

        return Progress(*columns, console=self.console, expand=True)

    def add_stage(self, name: str, total: int) -> StageProgress:
        """
        Add a new stage to track.

        Args:
            name: Stage name
            total: Total items to process

        Returns:
            StageProgress object
        """
        stage = StageProgress(name=name, total=total)
        self._stages[name] = stage
        return stage

    def start_stage(self, name: str) -> StageProgress:
        """
        Start tracking a stage.

        Args:
            name: Stage name

        Returns:
            StageProgress object
        """
        if name not in self._stages:
            raise ValueError(f"Stage '{name}' not found. Add it first with add_stage()")

        self._current_stage = name
        stage = self._stages[name]
        stage.status = "in_progress"
        self._last_print_percent = -1

        # Print stage header
        self._print_stage_header(name, stage)

        if not self.simple_mode:
            # Create progress bar for terminal mode
            if self._progress is None:
                self._progress = self._create_progress()
                self._live = Live(self._progress, console=self.console, refresh_per_second=4)
                self._live.start()

            # Add task for this stage
            task_id = self._progress.add_task(
                description=name,
                total=stage.total,
                completed=stage.completed,
            )
            self._task_ids[name] = task_id

        return stage

    def _print_stage_header(self, name: str, stage: StageProgress) -> None:
        """Print a stage header."""
        stage_num = list(self._stages.keys()).index(name) + 1
        total_stages = len(self._stages)

        if self.simple_mode:
            # Simple header for Jupyter
            print(f"\n{'='*60}")
            print(f"  Stage {stage_num}/{total_stages}: {name}")
            print(f"{'='*60}")
        else:
            # Rich panel for terminal
            panel = Panel(
                f"[bold]Stage {stage_num}/{total_stages}: {name}[/bold]",
                title="[cyan]Structify Pipeline[/cyan]",
                border_style="cyan",
                expand=False,
            )
            self.console.print()
            self.console.print(panel)

    def _print_simple_progress(self, stage: StageProgress) -> None:
        """Print simple progress for Jupyter mode."""
        percent = int(stage.percentage)

        # Print at 5% intervals or on first/last item for responsiveness
        should_print = (
            percent >= self._last_print_percent + 5 or
            stage.completed == stage.total or
            stage.completed == 1
        )

        if not should_print:
            return

        bar_width = 30
        filled = int(bar_width * stage.completed / max(stage.total, 1))
        bar = "█" * filled + "░" * (bar_width - filled)

        line = f"  [{bar}] {stage.completed}/{stage.total} ({percent}%)"
        line += f" - {stage.records_found} records"

        if stage.completed == stage.total:
            print(f"\r{line}")  # Final line with newline
        else:
            print(f"\r{line}", end="", flush=True)

        self._last_print_percent = percent

    def update(
        self,
        stage_name: str | None = None,
        completed: int | None = None,
        current_item: str | None = None,
        records: int = 0,
    ) -> None:
        """
        Update progress for a stage.

        Args:
            stage_name: Stage to update (defaults to current)
            completed: New completed count
            current_item: Current item being processed
            records: Number of records found in this update
        """
        name = stage_name or self._current_stage
        if name is None or name not in self._stages:
            return

        stage = self._stages[name]

        if completed is not None:
            stage.completed = completed
        if current_item is not None:
            stage.current_item = current_item
        stage.records_found += records

        if self.simple_mode:
            self._print_simple_progress(stage)
        elif self._progress is not None and name in self._task_ids:
            self._progress.update(
                self._task_ids[name],
                completed=stage.completed,
                description=f"{name} ({stage.records_found} records)",
            )

    def increment(
        self,
        stage_name: str | None = None,
        records: int = 0,
        current_item: str | None = None,
    ) -> None:
        """
        Increment progress by one.

        Args:
            stage_name: Stage to update (defaults to current)
            records: Number of records found
            current_item: Current item being processed
        """
        name = stage_name or self._current_stage
        if name is None or name not in self._stages:
            return

        stage = self._stages[name]
        stage.increment(records)

        if current_item:
            stage.current_item = current_item

        if self.simple_mode:
            self._print_simple_progress(stage)
        elif self._progress is not None and name in self._task_ids:
            self._progress.update(
                self._task_ids[name],
                advance=1,
                description=f"{name} ({stage.records_found} records)",
            )

    def complete_stage(self, name: str | None = None) -> None:
        """
        Mark a stage as completed.

        Args:
            name: Stage name (defaults to current)
        """
        name = name or self._current_stage
        if name is None or name not in self._stages:
            return

        stage = self._stages[name]
        stage.status = "completed"
        stage.completed = stage.total

        if self.simple_mode:
            self._print_simple_progress(stage)
            print(f"  ✓ {name} complete: {stage.records_found} records extracted")
        else:
            # Update progress bar to 100%
            if self._progress is not None and name in self._task_ids:
                self._progress.update(self._task_ids[name], completed=stage.total)
            self.console.print(
                f"[green]✓ {name} complete: {stage.records_found} records extracted[/green]"
            )

    def error_stage(self, name: str | None = None, error: str = "") -> None:
        """
        Mark a stage as errored.

        Args:
            name: Stage name (defaults to current)
            error: Error message
        """
        name = name or self._current_stage
        if name is None or name not in self._stages:
            return

        stage = self._stages[name]
        stage.status = "error"
        stage.errors += 1

        if self.simple_mode:
            print(f"\n  ✗ {name} error: {error}")
        else:
            self.console.print(f"[red]✗ {name} error: {error}[/red]")

    def log_retry(self, attempt: int, max_attempts: int, wait_time: float) -> None:
        """
        Log a retry attempt.

        Args:
            attempt: Current attempt number
            max_attempts: Maximum attempts
            wait_time: Time to wait before retry
        """
        if self.simple_mode:
            print(f"\n  ⏳ Retrying ({attempt}/{max_attempts}), waiting {wait_time:.0f}s...")
        else:
            self.console.print(
                f"[yellow]  ⏳ Retrying ({attempt}/{max_attempts}), waiting {wait_time:.0f}s...[/yellow]"
            )

    def log_substep(self, message: str, style: str = "info") -> None:
        """
        Log a substep message.

        In simple mode (Jupyter), only show warnings and errors to reduce noise.

        Args:
            message: Message to log
            style: Style (info, success, warning, error)
        """
        # In simple mode, only show warnings and errors
        if self.simple_mode and style not in ("warning", "error"):
            return

        if self.simple_mode:
            icons = {"info": "→", "success": "✓", "warning": "⚠", "error": "✗"}
            icon = icons.get(style, "→")
            print(f"    {icon} {message}")
        else:
            styles = {
                "info": "[blue]  → {message}[/blue]",
                "success": "[green]  ✓ {message}[/green]",
                "warning": "[yellow]  ⚠ {message}[/yellow]",
                "error": "[red]  ✗ {message}[/red]",
            }
            template = styles.get(style, styles["info"])
            self.console.print(template.format(message=message))

    def finish(self) -> None:
        """Finish all progress tracking."""
        if self._live is not None:
            self._live.stop()
            self._live = None
        if self._progress is not None:
            self._progress = None

        # Print summary
        self._print_summary()

    def _print_summary(self) -> None:
        """Print a summary of all stages."""
        if self.simple_mode:
            print(f"\n{'='*60}")
            print("  Extraction Summary")
            print(f"{'='*60}")
            print(f"  {'Stage':<25} {'Status':<10} {'Items':<12} {'Records':<10}")
            print(f"  {'-'*25} {'-'*10} {'-'*12} {'-'*10}")
            for name, stage in self._stages.items():
                status_icon = {
                    "completed": "✓",
                    "error": "✗",
                    "in_progress": "→",
                    "pending": "○",
                }.get(stage.status, "○")
                print(f"  {name:<25} {status_icon:<10} {stage.completed}/{stage.total:<10} {stage.records_found:<10}")
            print(f"{'='*60}\n")
        else:
            self.console.print()
            table = Table(title="Extraction Summary", show_header=True, header_style="bold cyan")
            table.add_column("Stage", style="bold")
            table.add_column("Status", justify="center")
            table.add_column("Items", justify="right")
            table.add_column("Records", justify="right")

            for name, stage in self._stages.items():
                status_icon = {
                    "completed": "[green]✓[/green]",
                    "error": "[red]✗[/red]",
                    "in_progress": "[yellow]→[/yellow]",
                    "pending": "[dim]○[/dim]",
                }.get(stage.status, "○")

                table.add_row(
                    name,
                    status_icon,
                    f"{stage.completed}/{stage.total}",
                    str(stage.records_found),
                )

            self.console.print(table)

    def __enter__(self) -> "ProgressTracker":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.finish()
        return False

    def get_stage(self, name: str) -> StageProgress | None:
        """Get a stage by name."""
        return self._stages.get(name)

    @property
    def current_stage(self) -> StageProgress | None:
        """Get the current stage."""
        if self._current_stage:
            return self._stages.get(self._current_stage)
        return None
