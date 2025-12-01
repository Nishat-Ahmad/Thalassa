import sys
import json

def main():
    try:
        from flows.flow import pipeline
    except Exception as e:
        print(json.dumps({"status":"error","message":str(e)}))
        sys.exit(1)
    out = pipeline("AAPL")
    print(json.dumps({"status":"ok","output":out}))

if __name__ == "__main__":
    main()
