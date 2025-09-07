import subprocess
import sys
import os
import time
import webbrowser

def main():
    print("🚀 Starting Ragineer-Test Application...")
    
    # Start FastAPI backend
    print("🔧 Starting backend server...")
    backend_cmd = [sys.executable, '-m', 'uvicorn', 'backend.api_endpoints.api_app:app', '--host', '0.0.0.0', '--port', '8000', '--reload']
    backend_proc = subprocess.Popen(backend_cmd)
    
    # Start frontend server (simple HTTP server)
    print("🎨 Starting frontend server...")
    frontend_dir = os.path.join(os.getcwd(), 'frontend')
    frontend_cmd = [sys.executable, '-m', 'http.server', '3000']
    frontend_proc = subprocess.Popen(frontend_cmd, cwd=frontend_dir)
    
    # Give servers time to start
    time.sleep(3)
    
    print("\n✅ Both servers are running!")
    print(f"🔗 Backend API: http://localhost:8000")
    print(f"🌐 Frontend UI: http://localhost:3000")
    print("\n� To clear cache: Press Ctrl+F5 or use incognito mode")
    print("⚠️  Press Ctrl+C to stop both servers.")
    
    # Auto-open browser with cache buster
    try:
        cache_busted_url = f"http://localhost:3000/?v={int(time.time())}"
        webbrowser.open(cache_busted_url)
        print(f"🌍 Opening: {cache_busted_url}")
    except Exception as e:
        print(f"Could not auto-open browser: {e}")
    
    try:
        backend_proc.wait()
        frontend_proc.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping servers...")
        backend_proc.terminate()
        frontend_proc.terminate()
        print("✅ Servers stopped.")

if __name__ == "__main__":
    main()
