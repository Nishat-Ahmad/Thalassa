import sys
import os
import json

def main():
    try:
        # Ensure repo root is on sys.path
        here = os.path.dirname(__file__)
        repo_root = os.path.abspath(os.path.join(here, ".."))
        if repo_root not in sys.path:
            sys.path.append(repo_root)
        from flows.flow import pipeline
    except Exception as e:
        print(json.dumps({"status":"error","message":str(e)}))
        sys.exit(1)
    out = pipeline("AAPL")
    print(json.dumps({"status":"ok","output":out}))

if __name__ == "__main__":
    main()
