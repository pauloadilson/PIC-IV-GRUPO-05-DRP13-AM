import uvicorn
from app_ml.config import get_ml_settings

if __name__ == "__main__":
    settings = get_ml_settings()
    print("=================================================================")
    print("   API de Aprendizagem de Máquina - PIC-IV (Viveiro Bioterra)   ")
    print(f"   Porta configurada: {settings.porta_ml_api}")
    print(f"   Validação de chave (VALIDAR_ML_API_KEY): {settings.validar_ml_api_key}")
    print(f"   Documentação interativa: http://127.0.0.1:{settings.porta_ml_api}/docs")
    print("=================================================================")
    uvicorn.run(
        "app_ml.main:app",
        host="127.0.0.1",
        port=settings.porta_ml_api,
        reload=True,
    )

