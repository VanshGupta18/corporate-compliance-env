# FastAPI app entry point for OpenEnv
# This re-exports the main app from app/server/app.py to satisfy
# OpenEnv's requirement for server/app.py location

from app.server.app import app
import uvicorn


def main():
    """Main entry point for the server."""
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
