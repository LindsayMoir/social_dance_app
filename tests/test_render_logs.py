"""
Test script to verify Render logs download functionality.
"""
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import subprocess
import sys

# Load environment variables from .env file (in src directory, one level up from tests)
# Get the project root directory (parent of tests folder)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_root, 'src', '.env')
load_dotenv(env_path)

def test_render_logs_download():
    """Test downloading logs from Render cron job."""

    print("=" * 60)
    print("Testing Render Logs Download")
    print("=" * 60)

    # Check for RENDER_API_KEY
    render_api_key = os.getenv('RENDER_API_KEY')
    if not render_api_key:
        print("❌ RENDER_API_KEY not found in environment variables")
        print("Please set RENDER_API_KEY before running this test:")
        print("  export RENDER_API_KEY='your-api-key-here'")
        return False

    print("✓ RENDER_API_KEY found")

    # Service name
    service_name = os.getenv('RENDER_SERVICE_NAME', 'social_dance_app_cron')
    print(f"✓ Using service name: {service_name}")

    # Create logs directory (relative to project root)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(project_root, 'logs', 'render_logs')
    os.makedirs(log_dir, exist_ok=True)
    print(f"✓ Created log directory: {log_dir}")

    # Create timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'test_cron_log_{timestamp}.txt')
    print(f"✓ Will save to: {log_file}")

    # Download logs using Render API
    print(f"\nDownloading logs from Render...")

    try:
        import requests

        # First, get the service ID by name
        headers = {
            'Authorization': f'Bearer {render_api_key}',
            'Accept': 'application/json'
        }

        # List services to find the cron job
        services_url = 'https://api.render.com/v1/services'
        response = requests.get(services_url, headers=headers, params={'limit': 100})

        if response.status_code != 200:
            print(f"❌ Failed to list services: {response.status_code} - {response.text}")
            return False

        services = response.json()
        service_id = None
        owner_id = None

        # Debug: Show all available services
        print(f"\n📋 Available services:")
        for service in services:
            svc = service['service']
            print(f"  - Name: {svc['name']}")
            print(f"    Type: {svc.get('type', 'unknown')}")
            print(f"    ID: {svc['id']}")
            print()

        for service in services:
            if service['service']['name'] == service_name:
                service_id = service['service']['id']
                owner_id = service['service'].get('ownerId')
                break

        if not service_id:
            print(f"❌ Service '{service_name}' not found")
            return False

        print(f"✓ Found service ID: {service_id}")
        if owner_id:
            print(f"✓ Found owner ID: {owner_id}")

        # For cron jobs, we need to get job runs first, then their logs
        # Try to get recent job runs
        jobs_url = f'https://api.render.com/v1/services/{service_id}/jobs'
        jobs_response = requests.get(jobs_url, headers=headers, params={'limit': 10})

        if jobs_response.status_code == 200:
            jobs = jobs_response.json()
            print(f"✓ Found {len(jobs)} recent job runs")

            # Debug: Show all available jobs
            if jobs:
                print(f"\n📋 Available job runs:")
                for idx, job_wrapper in enumerate(jobs[:10], 1):  # Show up to 10 most recent
                    job = job_wrapper['job']
                    print(f"  {idx}. Job ID: {job.get('id')}")
                    print(f"     Status: {job.get('status', 'unknown')}")
                    print(f"     Started: {job.get('startedAt', 'N/A')}")
                    print(f"     Finished: {job.get('finishedAt', 'N/A')}")
                    print()

            if jobs:
                # Get the most recent job
                latest_job = jobs[0]['job']
                job_id = latest_job['id']
                print(f"✓ Getting logs for most recent job: {job_id}")
                print(f"  Status: {latest_job.get('status', 'unknown')}")
                print(f"  Started: {latest_job.get('startedAt', 'N/A')}")

                # Get logs for this specific job
                job_logs_url = f'https://api.render.com/v1/jobs/{job_id}/logs'
                log_response = requests.get(job_logs_url, headers=headers)

                if log_response.status_code != 200:
                    print(f"❌ Failed to get job logs: {log_response.status_code} - {log_response.text}")
                    return False

                logs_text = log_response.text
            else:
                print("⚠ No recent job runs found, trying alternative endpoints...")
                # Fall through to try alternative methods
                jobs_response.status_code = 404  # Trigger fallback

        if jobs_response.status_code != 200:
            print(f"⚠ Could not get job runs (status {jobs_response.status_code}), trying direct logs endpoint...")

            # Try the /v1/logs endpoint with service filter
            # Query parameters for filtering by service
            # Get logs from the last 24 hours
            # Render API expects RFC3339/ISO 8601 format timestamps
            end_time = datetime.utcnow()
            start_time = end_time - timedelta(hours=24)

            # Format as RFC3339 (ISO 8601 with Z for UTC)
            start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%SZ')
            end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%SZ')

            logs_url = 'https://api.render.com/v1/logs'

            # Try different parameter formats - Render API requires ownerId and resource
            # Based on error messages, build params with ownerId
            log_params = {
                'resource': service_id,
                'startTime': start_time_str,
                'endTime': end_time_str,
                'limit': 1000
            }

            # Add ownerId if we have it
            if owner_id:
                log_params['ownerId'] = owner_id

            attempts = [log_params]

            log_response = None
            for i, log_params in enumerate(attempts, 1):
                print(f"  Attempt {i}: Trying {logs_url} with params: {list(log_params.keys())}")
                log_response = requests.get(logs_url, headers=headers, params=log_params)

                if log_response.status_code == 200:
                    print(f"  ✓ Success with attempt {i}!")
                    break
                elif log_response.status_code != 400:  # Not a parameter error, might be worth breaking
                    print(f"  Got status {log_response.status_code}: {log_response.text[:200]}")
                    break
                else:
                    print(f"  ✗ Attempt {i} failed (400): {log_response.text[:200]}")

            if log_response.status_code != 200:
                print(f"❌ Failed to get logs: {log_response.status_code} - {log_response.text}")
                return False

            # The response should be JSON with log entries
            log_data = log_response.json()

            # Debug: Show the structure of the response
            print(f"\n📋 Log response structure:")
            if isinstance(log_data, list):
                print(f"  Type: list with {len(log_data)} entries")
                if log_data:
                    print(f"  First entry keys: {list(log_data[0].keys())}")
                    print(f"  First entry sample: {str(log_data[0])[:200]}")
            elif isinstance(log_data, dict):
                print(f"  Type: dict with keys: {list(log_data.keys())}")
                if 'logs' in log_data:
                    print(f"  'logs' key contains {len(log_data['logs'])} entries")
                if 'entries' in log_data:
                    print(f"  'entries' key contains {len(log_data['entries'])} entries")
            print()

            # Extract log text from the response
            # The exact format depends on Render's API response structure
            if isinstance(log_data, list):
                # If it's a list of log entries, concatenate them
                logs_text = '\n'.join([entry.get('text', entry.get('message', str(entry))) for entry in log_data])
            elif isinstance(log_data, dict):
                # If it's a dict, it might have a 'logs' key or similar
                logs_text = '\n'.join([entry.get('text', entry.get('message', str(entry)))
                                      for entry in log_data.get('logs', log_data.get('entries', []))])
            else:
                logs_text = str(log_data)

        # Check if we got any output
        if not logs_text:
            print("❌ No logs returned from Render")
            return False

        # Save logs to file
        with open(log_file, 'w') as f:
            f.write(logs_text)

        # Get stats
        num_chars = len(logs_text)
        num_lines = len(logs_text.splitlines())
        file_size_kb = num_chars / 1024

        print(f"\n✓ Logs downloaded successfully!")
        print(f"  - Characters: {num_chars:,}")
        print(f"  - Lines: {num_lines:,}")
        print(f"  - File size: {file_size_kb:.2f} KB")
        print(f"  - Saved to: {log_file}")

        # Show first few lines
        lines = logs_text.splitlines()
        if lines:
            print(f"\n📋 First 5 lines of log:")
            for i, line in enumerate(lines[:5], 1):
                print(f"  {i}. {line[:100]}{'...' if len(line) > 100 else ''}")

        # Show last few lines
        if len(lines) > 5:
            print(f"\n📋 Last 5 lines of log:")
            for i, line in enumerate(lines[-5:], len(lines)-4):
                print(f"  {i}. {line[:100]}{'...' if len(line) > 100 else ''}")

        return True

    except requests.exceptions.RequestException as e:
        print(f"\n❌ Failed to download logs:")
        print(f"  Error: {e}")
        return False

    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n")
    success = test_render_logs_download()
    print("\n" + "=" * 60)

    if success:
        print("✅ TEST PASSED: Render logs download working!")
        sys.exit(0)
    else:
        print("❌ TEST FAILED: Render logs download not working")
        sys.exit(1)
