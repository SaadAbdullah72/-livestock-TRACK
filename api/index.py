from main_app import app

# Vercel Python runtime expects a module-level `app` ASGI object.
# Importing the existing FastAPI app keeps the same API surface while
# allowing deployment on Vercel's serverless runtime.
