"""
Streamlit Dashboard for CF Recommendation Service Monitoring.

This dashboard provides real-time monitoring for:
- Service health metrics (latency, error rate, fallback rate)
- Training run history and performance
- Data drift detection results
- Model comparison
- Scheduler management (NEW)

Usage:
    streamlit run service/dashboard.py
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
import json
import sys
import os
import requests
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ============================================================================
# Configuration
# ============================================================================

TRAINING_DB_PATH = "logs/training_metrics.db"
SERVICE_DB_PATH = "logs/service_metrics.db"
SCHEDULER_LOG_DIR = Path("logs/scheduler")
SCHEDULER_STATUS_FILE = SCHEDULER_LOG_DIR / "task_status.json"
SCHEDULER_CONFIG_PATH = Path("config/scheduler_config.json")

# API URL: Use Docker service name if running in container, else localhost
# In Docker: viecomrec-api:8000, Local: localhost:8000
API_BASE_URL = os.environ.get("API_URL", "http://viecomrec-api:8000")

# Page config
st.set_page_config(
    page_title="CF Recommendation Dashboard",
    page_icon="📊",
    layout="wide"
)


# ============================================================================
# Database Connections
# ============================================================================

def get_training_db():
    """Get training metrics database connection (fresh each time for real-time data)."""
    if not Path(TRAINING_DB_PATH).exists():
        return None
    return sqlite3.connect(TRAINING_DB_PATH, check_same_thread=False)


def get_service_db():
    """Get service metrics database connection (fresh each time for real-time data)."""
    if not Path(SERVICE_DB_PATH).exists():
        return None
    return sqlite3.connect(SERVICE_DB_PATH, check_same_thread=False)


# ============================================================================
# API Client Functions
# ============================================================================

def api_request(endpoint: str, method: str = "GET", json_data: dict = None) -> dict:
    """Make API request to the service."""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=json_data, timeout=30)
        elif method == "PUT":
            response = requests.put(url, json=json_data, timeout=10)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API service. Is it running?"}
    except requests.exceptions.Timeout:
        return {"error": "API request timed out"}
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_service_health(conn, minutes: int = 60) -> pd.DataFrame:
    """Load service health metrics."""
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = f"""
            SELECT * FROM service_health
            WHERE timestamp > datetime('now', '-{minutes} minutes')
            ORDER BY timestamp
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()


def load_requests(conn, minutes: int = 60) -> pd.DataFrame:
    """Load request logs."""
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = f"""
            SELECT * FROM requests
            WHERE timestamp > datetime('now', '-{minutes} minutes')
            ORDER BY timestamp DESC
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception:
        return pd.DataFrame()


def load_training_runs(conn, limit: int = 20) -> pd.DataFrame:
    """Load training run history."""
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = f"""
            SELECT * FROM training_runs
            ORDER BY started_at DESC
            LIMIT {limit}
        """
        df = pd.read_sql(query, conn)
        if not df.empty:
            df['started_at'] = pd.to_datetime(df['started_at'])
            if 'completed_at' in df.columns:
                df['completed_at'] = pd.to_datetime(df['completed_at'])
        return df
    except Exception:
        return pd.DataFrame()


def load_iteration_metrics(conn, run_id: str) -> pd.DataFrame:
    """Load iteration metrics for a training run."""
    if conn is None:
        return pd.DataFrame()
    
    try:
        query = f"""
            SELECT * FROM iteration_metrics
            WHERE run_id = '{run_id}'
            ORDER BY iteration
        """
        df = pd.read_sql(query, conn)
        return df
    except Exception:
        return pd.DataFrame()


# ============================================================================
# Dashboard Components
# ============================================================================

def render_header():
    """Render dashboard header."""
    st.title("📊 CF Recommendation Service Dashboard")
    st.markdown("Real-time monitoring for the Vietnamese cosmetics recommendation system")
    
    # Time selector
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        time_range = st.selectbox(
            "Time Range",
            ["Last 15 min", "Last 1 hour", "Last 6 hours", "Last 24 hours"],
            index=1
        )
    
    minutes_map = {
        "Last 15 min": 15,
        "Last 1 hour": 60,
        "Last 6 hours": 360,
        "Last 24 hours": 1440
    }
    
    with col2:
        auto_refresh = st.checkbox("Auto-refresh", value=False)
    
    return minutes_map.get(time_range, 60), auto_refresh


def render_service_overview(service_db, minutes: int):
    """Render service overview section."""
    st.header("🚀 Service Health")
    
    health_df = load_service_health(service_db, minutes)
    requests_df = load_requests(service_db, minutes)
    
    if health_df.empty and requests_df.empty:
        st.info("No service data available. Start the API server to collect metrics.")
        return
    
    # Summary metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if not requests_df.empty:
        total_requests = len(requests_df)
        avg_latency = requests_df['latency_ms'].mean()
        p95_latency = requests_df['latency_ms'].quantile(0.95)
        fallback_rate = requests_df['fallback'].mean() if 'fallback' in requests_df.columns else 0
        error_rate = (requests_df['error'].notna().sum() / len(requests_df)) if 'error' in requests_df.columns else 0
        
        col1.metric("Total Requests", f"{total_requests:,}")
        col2.metric("Avg Latency", f"{avg_latency:.1f} ms")
        col3.metric("P95 Latency", f"{p95_latency:.1f} ms")
        col4.metric("Fallback Rate", f"{fallback_rate:.1%}")
        col5.metric("Error Rate", f"{error_rate:.1%}")
    
    # Time series charts
    if not health_df.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Requests per Minute")
            st.line_chart(health_df.set_index('timestamp')['requests_per_minute'])
        
        with col2:
            st.subheader("Average Latency (ms)")
            st.line_chart(health_df.set_index('timestamp')['avg_latency_ms'])
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.subheader("Fallback Rate")
            if 'fallback_rate' in health_df.columns:
                st.line_chart(health_df.set_index('timestamp')['fallback_rate'])
        
        with col4:
            st.subheader("Error Rate")
            if 'error_rate' in health_df.columns:
                st.line_chart(health_df.set_index('timestamp')['error_rate'])
    
    # Recent requests table
    if not requests_df.empty:
        st.subheader("Recent Requests")
        display_cols = ['timestamp', 'user_id', 'topk', 'latency_ms', 'num_recommendations', 'fallback', 'model_id']
        display_cols = [c for c in display_cols if c in requests_df.columns]
        st.dataframe(requests_df[display_cols].head(20), use_container_width=True)


def render_training_history(training_db):
    """Render training history section."""
    st.header("🏋️ Training History")
    
    runs_df = load_training_runs(training_db)
    
    if runs_df.empty:
        st.info("No training runs recorded yet.")
        return
    
    # Summary
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Runs", len(runs_df))
    col2.metric("Successful", len(runs_df[runs_df['status'] == 'completed']))
    
    if 'recall_at_10' in runs_df.columns:
        best_recall = runs_df['recall_at_10'].max()
        col3.metric("Best Recall@10", f"{best_recall:.4f}" if pd.notna(best_recall) else "N/A")
    
    # Training runs table
    st.subheader("Recent Training Runs")
    display_cols = ['run_id', 'model_type', 'status', 'recall_at_10', 'ndcg_at_10', 'training_time_seconds', 'started_at']
    display_cols = [c for c in display_cols if c in runs_df.columns]
    st.dataframe(runs_df[display_cols], use_container_width=True)
    
    # Model comparison chart
    if 'recall_at_10' in runs_df.columns:
        st.subheader("Model Performance Comparison")
        
        completed_runs = runs_df[runs_df['status'] == 'completed'].copy()
        if not completed_runs.empty:
            completed_runs = completed_runs.sort_values('started_at')
            
            chart_data = completed_runs[['run_id', 'recall_at_10', 'ndcg_at_10']].set_index('run_id')
            st.bar_chart(chart_data)
    
    # Training curves for selected run
    if not runs_df.empty:
        st.subheader("Training Curves")
        selected_run = st.selectbox("Select Run", runs_df['run_id'].tolist())
        
        if selected_run:
            iter_df = load_iteration_metrics(training_db, selected_run)
            
            if not iter_df.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'loss' in iter_df.columns:
                        st.line_chart(iter_df.set_index('iteration')['loss'])
                        st.caption("Training Loss")
                
                with col2:
                    if 'validation_ndcg' in iter_df.columns:
                        st.line_chart(iter_df.set_index('iteration')['validation_ndcg'])
                        st.caption("Validation NDCG@10")
            else:
                st.info("No iteration metrics available for this run.")


def render_drift_detection():
    """Render drift detection section."""
    st.header("📈 Data Drift Detection")
    
    st.info("Click 'Run Drift Check' to analyze data for distribution shifts.")
    
    if st.button("Run Drift Check"):
        try:
            from recsys.cf.drift_detection import run_drift_detection
            
            # Load data
            interactions_path = Path("data/processed/interactions.parquet")
            embeddings_path = Path("data/processed/content_based_embeddings/product_embeddings.pt")
            
            if not interactions_path.exists():
                st.error("Interactions data not found. Run data processing first.")
                return
            
            historical_data = pd.read_parquet(interactions_path)
            
            # Use 80/20 split for demo (in production, use actual recent data)
            split_idx = int(len(historical_data) * 0.8)
            hist = historical_data.iloc[:split_idx]
            new = historical_data.iloc[split_idx:]
            
            # Run detection
            with st.spinner("Analyzing data drift..."):
                report = run_drift_detection(
                    hist, new,
                    str(embeddings_path) if embeddings_path.exists() else None
                )
            
            # Display results
            st.subheader("Drift Detection Results")
            
            # Overall status
            if report['overall']['drift_detected']:
                st.error("⚠️ Drift Detected - Consider retraining")
            else:
                st.success("✅ No significant drift detected")
            
            # Individual checks
            for check_name, result in report['checks'].items():
                with st.expander(f"📊 {check_name.replace('_', ' ').title()}"):
                    if 'error' in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        for key, value in result.items():
                            if isinstance(value, float):
                                st.write(f"**{key}**: {value:.4f}")
                            else:
                                st.write(f"**{key}**: {value}")
        
        except ImportError as e:
            st.error(f"Import error: {e}")
        except Exception as e:
            st.error(f"Error running drift detection: {e}")


def render_model_info():
    """Render current model information."""
    st.header("🤖 Current Model")
    
    registry_path = Path("artifacts/cf/registry.json")
    
    if not registry_path.exists():
        st.warning("No model registry found.")
        return
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    current_best = registry.get('current_best')
    if isinstance(current_best, dict):
        current_best = current_best.get('model_id')
    
    models_data = registry.get('models', [])
    if isinstance(models_data, dict):
        models = []
        for model_id, info in models_data.items():
            entry = dict(info) if isinstance(info, dict) else {}
            entry.setdefault('model_id', model_id)
            models.append(entry)
    else:
        models = models_data
    
    if current_best:
        st.subheader(f"Active Model: {current_best}")
        current_model = next((m for m in models if m.get('model_id') == current_best), None)
        if current_model:
            col1, col2, col3 = st.columns(3)
            
            col1.metric("Model Type", current_model.get('model_type', 'N/A'))
            
            metrics = current_model.get('metrics', {})
            col2.metric("Recall@10", f"{metrics.get('recall@10', 0):.4f}")
            col3.metric("NDCG@10", f"{metrics.get('ndcg@10', 0):.4f}")
            
            # Additional details
            with st.expander("Model Details"):
                st.json(current_model)
    
    # All models
    st.subheader("All Registered Models")
    if models:
        models_df = pd.DataFrame(models)
        st.dataframe(models_df, use_container_width=True)
    else:
        st.info("No models registered.")


def render_scheduler():
    """Render scheduler management section."""
    st.header("📅 Scheduler Management")
    
    # Try to get status from API first
    api_status = api_request("/scheduler/status")
    use_api = "error" not in api_status
    
    if use_api:
        # API is available - use real-time data
        st.success("🟢 Connected to API - Real-time data")
        
        # Status overview
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Scheduler", "Running" if api_status.get('running') else "Stopped")
        col2.metric("Total Jobs", api_status.get('total_jobs', 0))
        col3.metric("Enabled", api_status.get('enabled_jobs', 0))
        col4.metric("Disabled", api_status.get('disabled_jobs', 0))
        
        # Last health check
        if api_status.get('last_health_check'):
            st.caption(f"Last health check: {api_status['last_health_check']}")
        
        # Jobs list from API
        st.subheader("Scheduled Jobs")
        jobs_response = api_request("/scheduler/jobs")
        
        if "error" not in jobs_response:
            jobs = jobs_response.get('jobs', [])
            
            for job in jobs:
                job_id = job.get('job_id', 'unknown')
                enabled = job.get('enabled', False)
                description = job.get('description', '')
                last_status = job.get('last_status', 'N/A')
                last_run = job.get('last_run', 'Never')
                
                # Status indicator
                status_icon = "✅" if enabled else "⏸️"
                status_color = "green" if last_status == "success" else ("red" if last_status == "failed" else "gray")
                
                with st.expander(f"{status_icon} **{job_id}** - {description}"):
                    col1, col2, col3 = st.columns(3)
                    
                    col1.write(f"**Status:** {last_status}")
                    col2.write(f"**Last Run:** {last_run[:19] if last_run else 'Never'}")
                    col3.write(f"**Schedule:** {json.dumps(job.get('schedule', {}))}")
                    
                    # Action buttons
                    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)
                    
                    with btn_col1:
                        if st.button("▶️ Run Now", key=f"run_{job_id}"):
                            result = api_request(f"/scheduler/jobs/{job_id}/run", "POST")
                            if "error" in result:
                                st.error(result["error"])
                            else:
                                st.success(f"Job triggered! Log: {result.get('log_file', '')}")
                                st.rerun()
                    
                    with btn_col2:
                        if enabled:
                            if st.button("⏸️ Disable", key=f"disable_{job_id}"):
                                result = api_request(f"/scheduler/jobs/{job_id}/disable", "POST")
                                if "error" not in result:
                                    st.success("Job disabled")
                                    st.rerun()
                        else:
                            if st.button("▶️ Enable", key=f"enable_{job_id}"):
                                result = api_request(f"/scheduler/jobs/{job_id}/enable", "POST")
                                if "error" not in result:
                                    st.success("Job enabled")
                                    st.rerun()
                    
                    with btn_col3:
                        if st.button("📄 View Logs", key=f"logs_{job_id}"):
                            logs_response = api_request(f"/scheduler/logs/{job_id}?lines=50")
                            if "error" not in logs_response:
                                st.code("\n".join(logs_response.get('logs', ['No logs available'])), language="text")
                            else:
                                st.warning("No logs available")
                    
                    with btn_col4:
                        # Schedule editor
                        st.write("**Update Schedule:**")
                        new_hour = st.number_input("Hour", min_value=0, max_value=23, 
                                                    value=job.get('schedule', {}).get('hour', 0),
                                                    key=f"hour_{job_id}")
                        new_minute = st.number_input("Minute", min_value=0, max_value=59,
                                                      value=job.get('schedule', {}).get('minute', 0),
                                                      key=f"minute_{job_id}")
                        if st.button("💾 Save Schedule", key=f"save_{job_id}"):
                            new_schedule = {"hour": new_hour, "minute": new_minute}
                            result = api_request(f"/scheduler/jobs/{job_id}/schedule", "PUT", 
                                                {"schedule": new_schedule})
                            if "error" not in result:
                                st.success("Schedule updated! Restart scheduler to apply.")
                            else:
                                st.error(result["error"])
        else:
            st.error(f"Error fetching jobs: {jobs_response['error']}")
        
        # Task execution history
        st.subheader("📜 Execution History")
        history_response = api_request("/scheduler/history?page=1&page_size=10")
        
        if "error" not in history_response:
            history = history_response.get('history', [])
            if history:
                history_df = pd.DataFrame(history)
                display_cols = ['task_name', 'status', 'timestamp', 'exit_code']
                display_cols = [c for c in display_cols if c in history_df.columns]
                
                # Color status
                def color_status(val):
                    if val == 'success':
                        return 'background-color: #d4edda'
                    elif val == 'failed':
                        return 'background-color: #f8d7da'
                    return ''
                
                st.dataframe(
                    history_df[display_cols].style.applymap(color_status, subset=['status']),
                    use_container_width=True
                )
            else:
                st.info("No execution history available")
        
    else:
        # API not available - use local files
        st.warning(f"⚠️ Cannot connect to API ({api_status.get('error', 'Unknown error')}). Showing cached data.")
        
        # Load from local files
        if SCHEDULER_STATUS_FILE.exists():
            try:
                with open(SCHEDULER_STATUS_FILE, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                # Handle corrupted JSON (multiple JSON objects on separate lines)
                if content:
                    # Try parsing as single JSON first
                    try:
                        task_status = json.loads(content)
                    except json.JSONDecodeError:
                        # Parse line by line (JSONL format)
                        task_status = {}
                        for line in content.split('\n'):
                            line = line.strip()
                            if line:
                                try:
                                    entry = json.loads(line)
                                    task_name = entry.get('task', 'unknown')
                                    task_status[task_name] = entry
                                except json.JSONDecodeError:
                                    continue
                    
                    st.subheader("Task Status (from file)")
                    
                    for task_name, status in task_status.items():
                        if isinstance(status, dict):
                            status_val = status.get('status', 'unknown')
                            timestamp = status.get('timestamp', 'N/A')
                        else:
                            status_val = str(status)
                            timestamp = 'N/A'
                        status_icon = "✅" if status_val == 'success' else "❌"
                        st.write(f"{status_icon} **{task_name}**: {status_val} at {timestamp}")
                else:
                    st.info("Task status file is empty.")
            except Exception as e:
                st.error(f"Error reading task status: {e}")
        else:
            st.info("No scheduler status file found. Start the scheduler to generate status.")
        
        # Show config
        if SCHEDULER_CONFIG_PATH.exists():
            with open(SCHEDULER_CONFIG_PATH, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            st.subheader("Scheduler Configuration")
            st.json(config)
    
    # Manual refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Data"):
        st.rerun()


# ============================================================================
# Main Dashboard
# ============================================================================

def main():
    """Main dashboard function."""
    # Get database connections (fresh each request for real-time data)
    training_db = get_training_db()
    service_db = get_service_db()
    
    # Header
    minutes, auto_refresh = render_header()
    
    # Auto-refresh with configurable interval
    if auto_refresh:
        # Use session state to track refresh
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = time.time()
        
        # Refresh every 10 seconds
        if time.time() - st.session_state.last_refresh > 10:
            st.session_state.last_refresh = time.time()
            st.rerun()
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Service Health",
        "🏋️ Training History",
        "📅 Scheduler",
        "📈 Drift Detection",
        "🤖 Model Info"
    ])
    
    with tab1:
        render_service_overview(service_db, minutes)
    
    with tab2:
        render_training_history(training_db)
    
    with tab3:
        render_scheduler()
    
    with tab4:
        render_drift_detection()
    
    with tab5:
        render_model_info()
    
    # Footer
    st.markdown("---")
    col1, col2 = st.columns([3, 1])
    with col1:
        st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    with col2:
        if st.button("🔄 Refresh Now"):
            st.rerun()


if __name__ == "__main__":
    main()
