#!/usr/bin/env python3
"""
NeuroTopo GUI - Tkinter interface for the retopology pipeline.

Run with: python neurotopo_gui.py
Or:       .venv/bin/python neurotopo_gui.py
"""
import logging
import queue
import sys
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from datetime import datetime

# Ensure we use the project's virtual environment packages
_project_root = Path(__file__).parent
_venv_site_packages = _project_root / ".venv" / "lib"

# Find the site-packages in the venv
for sp in _venv_site_packages.glob("python*/site-packages"):
    if str(sp) not in sys.path:
        sys.path.insert(0, str(sp))

# Add src directory to path for neurotopo imports
_src_path = _project_root / "src"
if _src_path.exists() and str(_src_path) not in sys.path:
    sys.path.insert(0, str(_src_path))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("neurotopo.gui")


class LogHandler(logging.Handler):
    """Custom log handler that sends logs to a queue for the GUI."""
    
    def __init__(self, log_queue: queue.Queue):
        super().__init__()
        self.log_queue = log_queue
        
    def emit(self, record):
        msg = self.format(record)
        self.log_queue.put(msg)


class NeuroTopoGUI:
    """Main GUI application for NeuroTopo."""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("NeuroTopo - AI-Assisted Retopology")
        self.root.geometry("900x700")
        self.root.minsize(700, 500)
        
        # State
        self.input_path = tk.StringVar()
        self.output_path = tk.StringVar()
        self.target_faces = tk.IntVar(value=5000)
        self.backend = tk.StringVar(value="hybrid")
        self.neural_weight = tk.DoubleVar(value=0.6)
        self.use_semantic = tk.BooleanVar(value=False)
        self.use_ai_quality = tk.BooleanVar(value=False)
        self.manifold_repair = tk.BooleanVar(value=True)
        
        # Processing state
        self.is_processing = False
        self.log_queue = queue.Queue()
        
        # Setup logging to GUI
        self._setup_logging()
        
        # Build UI
        self._create_menu()
        self._create_main_layout()
        
        # Start log polling
        self._poll_logs()
        
    def _setup_logging(self):
        """Configure logging to display in the GUI."""
        handler = LogHandler(self.log_queue)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        
        # Add handler to neurotopo loggers
        for name in ['neurotopo', 'neurotopo.pipeline', 'neurotopo.remesh', 'neurotopo.evaluation']:
            log = logging.getLogger(name)
            log.addHandler(handler)
            log.setLevel(logging.INFO)
            
    def _create_menu(self):
        """Create the menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Mesh...", command=self._browse_input, accelerator="Cmd+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        
    def _create_main_layout(self):
        """Create the main application layout."""
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # === Input/Output Section ===
        io_frame = ttk.LabelFrame(main_frame, text="Input / Output", padding="10")
        io_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input file
        ttk.Label(io_frame, text="Input Mesh:").grid(row=0, column=0, sticky=tk.W, pady=2)
        input_entry = ttk.Entry(io_frame, textvariable=self.input_path, width=60)
        input_entry.grid(row=0, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(io_frame, text="Browse...", command=self._browse_input).grid(row=0, column=2, pady=2)
        
        # Output file
        ttk.Label(io_frame, text="Output Mesh:").grid(row=1, column=0, sticky=tk.W, pady=2)
        output_entry = ttk.Entry(io_frame, textvariable=self.output_path, width=60)
        output_entry.grid(row=1, column=1, padx=5, pady=2, sticky=tk.EW)
        ttk.Button(io_frame, text="Browse...", command=self._browse_output).grid(row=1, column=2, pady=2)
        
        io_frame.columnconfigure(1, weight=1)
        
        # === Parameters Section ===
        params_frame = ttk.LabelFrame(main_frame, text="Parameters", padding="10")
        params_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Left column
        left_params = ttk.Frame(params_frame)
        left_params.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        # Target faces
        faces_frame = ttk.Frame(left_params)
        faces_frame.pack(fill=tk.X, pady=2)
        ttk.Label(faces_frame, text="Target Faces:").pack(side=tk.LEFT)
        faces_spin = ttk.Spinbox(faces_frame, from_=100, to=1000000, textvariable=self.target_faces, width=10)
        faces_spin.pack(side=tk.LEFT, padx=5)
        
        # Backend selection
        backend_frame = ttk.Frame(left_params)
        backend_frame.pack(fill=tk.X, pady=2)
        ttk.Label(backend_frame, text="Backend:").pack(side=tk.LEFT)
        backends = ["trimesh", "hybrid", "guided_quad", "isotropic", "feature_aware"]
        backend_combo = ttk.Combobox(backend_frame, textvariable=self.backend, values=backends, state="readonly", width=15)
        backend_combo.pack(side=tk.LEFT, padx=5)
        
        # Neural weight slider
        weight_frame = ttk.Frame(left_params)
        weight_frame.pack(fill=tk.X, pady=2)
        ttk.Label(weight_frame, text="Neural Weight:").pack(side=tk.LEFT)
        weight_scale = ttk.Scale(weight_frame, from_=0.0, to=1.0, variable=self.neural_weight, orient=tk.HORIZONTAL, length=150)
        weight_scale.pack(side=tk.LEFT, padx=5)
        self.weight_label = ttk.Label(weight_frame, text="0.6")
        self.weight_label.pack(side=tk.LEFT)
        weight_scale.configure(command=self._update_weight_label)
        
        # Right column - checkboxes
        right_params = ttk.Frame(params_frame)
        right_params.pack(side=tk.RIGHT, fill=tk.X, padx=(20, 0))
        
        ttk.Checkbutton(right_params, text="Semantic Guidance (GPT-4o)", variable=self.use_semantic).pack(anchor=tk.W)
        ttk.Checkbutton(right_params, text="AI Quality Assessment", variable=self.use_ai_quality).pack(anchor=tk.W)
        ttk.Checkbutton(right_params, text="Manifold Repair", variable=self.manifold_repair).pack(anchor=tk.W)
        
        # === Action Buttons ===
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.run_button = ttk.Button(button_frame, text="▶ Run Pipeline", command=self._run_pipeline, style="Accent.TButton")
        self.run_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(button_frame, text="⏹ Stop", command=self._stop_pipeline, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(button_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=10)
        
        # === Log Output ===
        log_frame = ttk.LabelFrame(main_frame, text="Log Output", padding="5")
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Log toolbar
        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(log_toolbar, text="📋 Copy to Clipboard", command=self._copy_log_to_clipboard).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_toolbar, text="💾 Save to File", command=self._save_log_to_file).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_toolbar, text="🗑 Clear", command=self._clear_log).pack(side=tk.LEFT)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, font=('Menlo', 10), wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)
        
        # Configure text tags for coloring
        self.log_text.tag_configure('INFO', foreground='black')
        self.log_text.tag_configure('WARNING', foreground='orange')
        self.log_text.tag_configure('ERROR', foreground='red')
        self.log_text.tag_configure('SUCCESS', foreground='green')
        
        # === Status Bar ===
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))
        
    def _update_weight_label(self, value):
        """Update the neural weight label."""
        self.weight_label.configure(text=f"{float(value):.2f}")
        
    def _browse_input(self):
        """Open file dialog for input mesh."""
        filetypes = [
            ("Mesh files", "*.obj *.fbx *.stl *.ply *.glb *.gltf"),
            ("OBJ files", "*.obj"),
            ("FBX files", "*.fbx"),
            ("STL files", "*.stl"),
            ("All files", "*.*")
        ]
        path = filedialog.askopenfilename(title="Select Input Mesh", filetypes=filetypes)
        if path:
            self.input_path.set(path)
            # Auto-generate output path
            p = Path(path)
            self.output_path.set(str(p.parent / f"{p.stem}_retopo{p.suffix}"))
            
    def _browse_output(self):
        """Open file dialog for output mesh."""
        filetypes = [
            ("OBJ files", "*.obj"),
            ("All files", "*.*")
        ]
        path = filedialog.asksaveasfilename(title="Save Output Mesh", filetypes=filetypes, defaultextension=".obj")
        if path:
            self.output_path.set(path)
            
    def _log(self, message: str, level: str = "INFO"):
        """Add a message to the log."""
        self.log_text.configure(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"{timestamp} [{level}] {message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
    def _clear_log(self):
        """Clear the log output."""
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state=tk.DISABLED)
        
    def _copy_log_to_clipboard(self):
        """Copy log contents to clipboard."""
        log_content = self.log_text.get(1.0, tk.END).strip()
        if log_content:
            self.root.clipboard_clear()
            self.root.clipboard_append(log_content)
            self.status_var.set("Log copied to clipboard")
        else:
            self.status_var.set("Log is empty")
            
    def _save_log_to_file(self):
        """Save log contents to a file."""
        log_content = self.log_text.get(1.0, tk.END).strip()
        if not log_content:
            messagebox.showinfo("Info", "Log is empty, nothing to save.")
            return
            
        # Generate default filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"neurotopo_log_{timestamp}.txt"
        
        path = filedialog.asksaveasfilename(
            title="Save Log File",
            defaultextension=".txt",
            initialfile=default_name,
            filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")]
        )
        if path:
            try:
                with open(path, 'w') as f:
                    f.write(log_content)
                self.status_var.set(f"Log saved to {Path(path).name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save log: {e}")
        
    def _poll_logs(self):
        """Poll the log queue and display messages."""
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                level = "INFO"
                if "[WARNING]" in msg:
                    level = "WARNING"
                elif "[ERROR]" in msg:
                    level = "ERROR"
                self.log_text.configure(state=tk.NORMAL)
                self.log_text.insert(tk.END, msg + "\n", level)
                self.log_text.see(tk.END)
                self.log_text.configure(state=tk.DISABLED)
            except queue.Empty:
                break
        self.root.after(100, self._poll_logs)
        
    def _run_pipeline(self):
        """Run the retopology pipeline in a background thread."""
        # Validate input
        if not self.input_path.get():
            messagebox.showerror("Error", "Please select an input mesh file.")
            return
            
        if not Path(self.input_path.get()).exists():
            messagebox.showerror("Error", f"Input file not found: {self.input_path.get()}")
            return
            
        if not self.output_path.get():
            messagebox.showerror("Error", "Please specify an output path.")
            return
            
        # Update UI state
        self.is_processing = True
        self.run_button.configure(state=tk.DISABLED)
        self.stop_button.configure(state=tk.NORMAL)
        self.progress.start(10)
        self.status_var.set("Processing...")
        
        # Clear log
        self._clear_log()
        self._log(f"Starting pipeline...", "INFO")
        self._log(f"Input: {self.input_path.get()}", "INFO")
        self._log(f"Output: {self.output_path.get()}", "INFO")
        self._log(f"Backend: {self.backend.get()}, Target: {self.target_faces.get()} faces", "INFO")
        
        # Run in background thread
        thread = threading.Thread(target=self._pipeline_worker, daemon=True)
        thread.start()
        
    def _pipeline_worker(self):
        """Background worker for running the pipeline via CLI subprocess."""
        import subprocess
        
        try:
            # Find the venv Python and CLI
            venv_python = _project_root / ".venv" / "bin" / "python"
            if not venv_python.exists():
                raise FileNotFoundError(f"Virtual environment Python not found at {venv_python}")
            
            # Build CLI command
            cmd = [
                str(venv_python), "-m", "neurotopo.cli", "process",
                self.input_path.get(),
                "-o", self.output_path.get(),
                "-t", str(self.target_faces.get()),
                "-b", self.backend.get(),
                "--evaluate",
                "-v",  # Verbose for detailed output
            ]
            
            self._log_threadsafe(f"Running: neurotopo process ...")
            
            # Run the CLI in subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Combine stderr with stdout
                text=True,
                cwd=str(_project_root),
                env={**dict(__import__('os').environ), 'PYTHONPATH': str(_src_path)},
            )
            
            # Read output line by line
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                
                # Parse and display output
                # Strip ANSI color codes for clean logging
                import re
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line)
                
                if clean_line:
                    # Determine log level from content
                    level = "INFO"
                    if "error" in clean_line.lower() or "fail" in clean_line.lower():
                        level = "ERROR"
                    elif "warning" in clean_line.lower():
                        level = "WARNING"
                    elif "score" in clean_line.lower() or "complete" in clean_line.lower() or "✓" in line:
                        level = "SUCCESS"
                    
                    self._log_threadsafe(clean_line, level)
            
            # Wait for completion
            process.wait()
            
            if process.returncode != 0:
                raise RuntimeError(f"Pipeline exited with code {process.returncode}")
            
            # AI Quality Assessment if requested
            if self.use_ai_quality.get():
                self._run_ai_assessment()
            
            self._finish_processing(success=True)
            
        except Exception as e:
            self._log_threadsafe(f"Error: {str(e)}", "ERROR")
            import traceback
            self._log_threadsafe(traceback.format_exc(), "ERROR")
            self._finish_processing(success=False, error=str(e))
    
    def _run_ai_assessment(self):
        """Run AI quality assessment via CLI."""
        import subprocess
        
        self._log_threadsafe("Running AI quality assessment...")
        
        venv_python = _project_root / ".venv" / "bin" / "python"
        
        # Use evaluate command with AI assessment
        cmd = [
            str(venv_python), "-m", "neurotopo.cli", "evaluate",
            self.output_path.get(),
            "-v",
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=str(_project_root),
                env={**dict(__import__('os').environ), 'PYTHONPATH': str(_src_path)},
                timeout=300,
            )
            
            # Strip ANSI and log output
            import re
            for line in result.stdout.split('\n'):
                clean_line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()
                if clean_line:
                    level = "SUCCESS" if "score" in clean_line.lower() else "INFO"
                    self._log_threadsafe(clean_line, level)
                    
            if result.returncode != 0 and result.stderr:
                self._log_threadsafe(f"AI assessment warning: {result.stderr}", "WARNING")
                
        except Exception as e:
            self._log_threadsafe(f"AI assessment failed: {e}", "WARNING")
            
    def _log_threadsafe(self, message: str, level: str = "INFO"):
        """Thread-safe logging."""
        self.root.after(0, lambda: self._log(message, level))
        
    def _finish_processing(self, success: bool, error: str = None):
        """Called when processing finishes."""
        def finish():
            self.is_processing = False
            self.run_button.configure(state=tk.NORMAL)
            self.stop_button.configure(state=tk.DISABLED)
            self.progress.stop()
            
            if success:
                self.status_var.set("Complete!")
                messagebox.showinfo("Success", f"Retopology complete!\nOutput saved to:\n{self.output_path.get()}")
            else:
                self.status_var.set("Failed")
                messagebox.showerror("Error", f"Pipeline failed:\n{error}")
                
        self.root.after(0, finish)
        
    def _stop_pipeline(self):
        """Stop the running pipeline (not fully implemented - just UI feedback)."""
        self._log("Stopping pipeline... (may take a moment)", "WARNING")
        self.is_processing = False
        self.status_var.set("Stopping...")
        
    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About NeuroTopo",
            "NeuroTopo v0.4\n\n"
            "AI-Assisted Retopology\n\n"
            "Neural-guided, deterministically-controlled\n"
            "mesh retopology for production pipelines\n"
            "with GPT-4o visual quality assessment."
        )


def main():
    """Main entry point."""
    root = tk.Tk()
    
    # Try to use a modern theme
    try:
        root.tk.call("source", "/opt/homebrew/share/tcl-tk/themes/azure-dark/azure-dark.tcl")
        root.tk.call("set_theme", "dark")
    except tk.TclError:
        # Fall back to default theme
        style = ttk.Style()
        available_themes = style.theme_names()
        if 'aqua' in available_themes:
            style.theme_use('aqua')
        elif 'clam' in available_themes:
            style.theme_use('clam')
    
    app = NeuroTopoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
