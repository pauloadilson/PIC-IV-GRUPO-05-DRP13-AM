from app_ml.main import app

# Export do ASGI app para o runtime de Serverless Functions do Vercel
# Referência: https://vercel.com/docs/frameworks/fastapi
app = app

