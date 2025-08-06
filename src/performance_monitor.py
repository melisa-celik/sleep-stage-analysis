
import threading
import time
import psutil
import pynvml
import pandas as pd
import os
from IPython.display import HTML, display

class PerformanceMonitor:
    """
    Monitors CPU, RAM, and GPU performance and generates a visual HTML report.
    """
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self._is_running = False
        self._thread = None
        self._records = []
        self.start_time = None
        self._gpu_monitoring_available = False

        try:
            pynvml.nvmlInit()
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._gpu_monitoring_available = True
            print("[PerformanceMonitor] NVIDIA GPU monitoring is available.")
        except pynvml.NVMLError:
            print("[PerformanceMonitor] NVIDIA GPU not found. GPU monitoring will be disabled.")

    def _monitor(self):
        """The internal method that runs in a loop to collect metrics."""
        while self._is_running:
            timestamp = time.time() - self.start_time
            cpu_percent = psutil.cpu_percent(interval=None)
            ram_stats = psutil.virtual_memory()
            ram_percent = ram_stats.percent
            ram_used_gb = ram_stats.used / (1024**3)

            gpu_percent, gpu_mem_percent, gpu_mem_used_gb = None, None, None
            if self._gpu_monitoring_available:
                gpu_util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                gpu_percent = gpu_util.gpu
                gpu_mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                gpu_mem_percent = (gpu_mem_info.used / gpu_mem_info.total) * 100
                gpu_mem_used_gb = gpu_mem_info.used / (1024**3)

            self._records.append({
                'elapsed_time_s': timestamp,
                'cpu_percent': cpu_percent, 'ram_percent': ram_percent, 'ram_used_gb': ram_used_gb,
                'gpu_percent': gpu_percent, 'gpu_mem_percent': gpu_mem_percent, 'gpu_mem_used_gb': gpu_mem_used_gb
            })
            time.sleep(self.interval)

    def start(self):
        print("[PerformanceMonitor] Starting...")
        self.start_time = time.time()
        self._is_running = True
        self._records = []
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
        print("[PerformanceMonitor] Monitor started.")

    def stop(self):
        if not self._is_running: return
        self._is_running = False
        self._thread.join()
        if self._gpu_monitoring_available:
            pynvml.nvmlShutdown()
        print("[PerformanceMonitor] Monitor stopped.")
        return self.get_report()

    def get_report(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def generate_html_report(self, task_name: str, report_path: str = './reports'):
        """Generates and saves a fancy HTML report with summary cards and an interactive chart."""
        df = self.get_report()
        if df.empty:
            print("[PerformanceMonitor] No data to generate a report.")
            return

        if not os.path.exists(report_path):
            os.makedirs(report_path)

        filename = os.path.join(report_path, f"performance_report_{task_name}_{int(time.time())}.html")

        total_duration = df['elapsed_time_s'].iloc[-1]
        avg_cpu = f"{df['cpu_percent'].mean():.2f}%"
        max_cpu = f"{df['cpu_percent'].max():.2f}%"
        avg_ram = f"{df['ram_used_gb'].mean():.2f} GB ({df['ram_percent'].mean():.2f}%)"
        max_ram = f"{df['ram_used_gb'].max():.2f} GB ({df['ram_percent'].max():.2f}%)"

        avg_gpu, max_gpu, avg_gpu_mem, max_gpu_mem = "N/A", "N/A", "N/A", "N/A"
        if self._gpu_monitoring_available and not df['gpu_percent'].isnull().all():
            avg_gpu = f"{df['gpu_percent'].mean():.2f}%"
            max_gpu = f"{df['gpu_percent'].max():.2f}%"
            avg_gpu_mem = f"{df['gpu_mem_used_gb'].mean():.2f} GB ({df['gpu_mem_percent'].mean():.2f}%)"
            max_gpu_mem = f"{df['gpu_mem_used_gb'].max():.2f} GB ({df['gpu_mem_percent'].max():.2f}%)"

        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['elapsed_time_s'], y=df['cpu_percent'], mode='lines', name='CPU Usage (%)'))
        fig.add_trace(go.Scatter(x=df['elapsed_time_s'], y=df['ram_percent'], mode='lines', name='RAM Usage (%)'))
        if self._gpu_monitoring_available:
            fig.add_trace(go.Scatter(x=df['elapsed_time_s'], y=df['gpu_percent'], mode='lines', name='GPU Usage (%)'))

        fig.update_layout(title=f'Performance Metrics for: {task_name}', xaxis_title='Time (seconds)', yaxis_title='Usage (%)', template='plotly_white')
        chart_html = fig.to_html(full_html=False, include_plotlyjs='cdn')

        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Performance Report: {task_name}</title>
            <script src="https://cdn.tailwindcss.com"></script>
        </head>
        <body class="bg-gray-100 font-sans p-8">
            <div class="max-w-6xl mx-auto bg-white rounded-lg shadow-xl p-8">
                <h1 class="text-4xl font-bold text-gray-800 mb-2">Performance Report</h1>
                <p class="text-lg text-gray-600 mb-6">Analysis for task: <span class="font-semibold text-indigo-600">{task_name}</span></p>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
                    <div class="bg-gray-50 p-6 rounded-lg shadow-sm"><p class="text-sm text-gray-500">Total Duration</p><p class="text-3xl font-bold text-gray-800">{total_duration:.2f}s</p></div>
                    <div class="bg-gray-50 p-6 rounded-lg shadow-sm"><p class="text-sm text-gray-500">Avg / Max CPU</p><p class="text-3xl font-bold text-gray-800">{avg_cpu} / {max_cpu}</p></div>
                    <div class="bg-gray-50 p-6 rounded-lg shadow-sm"><p class="text-sm text-gray-500">Avg / Max RAM</p><p class="text-3xl font-bold text-gray-800">{avg_ram} / {max_ram}</p></div>
                    <div class="bg-gray-50 p-6 rounded-lg shadow-sm"><p class="text-sm text-gray-500">Avg / Max GPU</p><p class="text-3xl font-bold text-gray-800">{avg_gpu} / {max_gpu}</p></div>
                </div>

                <div class="w-full">
                    {chart_html}
                </div>
            </div>
        </body>
        </html>
        """

        with open(filename, 'w') as f:
            f.write(html_content)

        print(f"[PerformanceMonitor] Fancy HTML report saved to: {filename}")
        display(HTML(f'<a href="{filename}" target="_blank">Click here to view the full report</a>'))
